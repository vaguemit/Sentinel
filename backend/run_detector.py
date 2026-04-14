"""
AI Sentinel Lite - Phase 5: Identity & Auto-Night Vision
------------------------------------------------------------------------------
- Face Recognizer identifies registered friends.
- Brightness detection automatically switches between RGB (1) and IR (0) cameras.
"""

import cv2
import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.detector import YoloDetector
from app.intelligence.scene_builder import SceneBuilder
from app.intelligence.ollama_client import OllamaClient
from app.memory.db_client import MemoryDB
from app.vision.face_recognizer import SentinelFaceRecognizer


def draw_scene_overlay(frame, scene, summary, is_night_vision=False):
    """Draw scene panel and AI summary on the frame."""
    h, w = frame.shape[:2]

    # --- Indicators ---
    if is_night_vision:
        cv2.putText(frame, "[ NIGHT VISION ACTIVE ]", (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    
    # --- Bottom-left: Scene data panel ---
    panel_x, panel_y = 10, h - 180
    panel_w, panel_h = 420, 170

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"  People   : {scene['people']}",
        f"  Identified: {', '.join(scene.get('identities', [])) if scene.get('identities') else 'None'}",
        f"  Density  : {scene['density']}",
        f"  Movement : {scene['movement']}",
    ]
    if scene['objects']:
        obj_str = ", ".join(f"{k}:{v}" for k, v in scene['objects'].items())
        lines.append(f"  Objects  : {obj_str}")

    cv2.putText(frame, "[ SCENE ]", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (panel_x + 10, panel_y + 48 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # --- Top: AI Summary bar ---
    summary_display = summary if summary else "Waiting for AI summary..."
    if len(summary_display) > 90:
        summary_display = summary_display[:87] + "..."

    bar_h = 40
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 0), (w, bar_h), (10, 10, 60), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "AI: " + summary_display, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 1)

    return frame


def main():
    print("Starting AI Sentinel Lite - Phase 5...")
    
    # Init Models
    detector = YoloDetector("yolov8s.pt")
    face_recognizer = SentinelFaceRecognizer()
    scene_builder = SceneBuilder(movement_threshold=15)
    ollama = OllamaClient()
    db = MemoryDB(persist_directory="chroma_db")

    if not ollama.is_running():
        print("[WARNING] Ollama is not running.")
        
    CAMERA_RGB = 1
    CAMERA_IR = 0
    current_camera = CAMERA_RGB

    cap = cv2.VideoCapture(current_camera)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened. Press 'Q' to quit.")

    current_summary = "Initializing AI..."
    last_summary_time = 0
    SUMMARY_INTERVAL = 4
    generating = False
    
    last_cam_switch_time = time.time()

    def generate_summary_async(scene):
        nonlocal current_summary, generating
        current_summary = "Thinking..."
        # Inject recognized faces into the prompt context via SceneBuilder modifications, 
        # but scene dict already holds it. We will ensure 'identities' is included.
        result = ollama.summarize(scene)
        current_summary = result
        db.save_event(result)
        generating = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. Auto-Night Vision Logic ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        now = time.time()
        
        # Don't switch cameras constantly (5s cooldown)
        if (now - last_cam_switch_time) > 5.0:
            if current_camera == CAMERA_RGB and brightness < 20:
                print(f"Low light detected (brightness: {brightness:.1f}). Switching to IR Night Vision.")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_IR)
                current_camera = CAMERA_IR
                last_cam_switch_time = now
                continue
                
            elif current_camera == CAMERA_IR and brightness > 120:
                print(f"Bright light detected (brightness: {brightness:.1f}). Switching to RGB Day Vision.")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_RGB)
                current_camera = CAMERA_RGB
                last_cam_switch_time = now
                continue

        # --- 2. YOLO Object Detection ---
        annotated_frame, counts, raw_result = detector.process_frame(frame)
        
        # --- 3. Face Identification ---
        recognized_names = set()
        # Find people bounding boxes, crop and identify
        for box in raw_result.boxes:
            cls_id = int(box.cls[0])
            if detector.model.names[cls_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Ensure valid crop
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size > 0 and person_crop.shape[0] > 50:
                    name = face_recognizer.identify(person_crop)
                    if name and name != "Unknown":
                        recognized_names.add(name)
                        # Draw name over person
                        cv2.putText(annotated_frame, name, (x1, max(y1-30, 20)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- 4. Scene Building ---
        scene = scene_builder.build(raw_result, detector.model.names)
        scene['identities'] = list(recognized_names)

        # --- 5. RAG LLM Call ---
        if not generating and (now - last_summary_time) >= SUMMARY_INTERVAL:
            generating = True
            last_summary_time = now
            t = threading.Thread(target=generate_summary_async, args=(scene,), daemon=True)
            t.start()

        # --- 6. Overlay & Show ---
        is_night = (current_camera == CAMERA_IR)
        annotated_frame = draw_scene_overlay(annotated_frame, scene, current_summary, is_night)
        cv2.imshow("AI Sentinel Lite", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
