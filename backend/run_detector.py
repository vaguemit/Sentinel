"""
AI Sentinel Lite - Phase 6: Full Stack Surveillance Engine
------------------------------------------------------------
- YOLOv8s object detection
- Centroid multi-object tracking with persistent IDs + radar
- MediaPipe skeletal pose estimation + action recognition
- OpenCV DNN face identification (friend vs foe)
- Qwen 2.5 LLM scene summarization via Ollama
- ChromaDB RAG memory
- Auto night vision (IR camera switching)

OPTIMIZATION STRATEGY (4GB VRAM / CPU-only):
  - YOLO runs every frame (lightweight on CPU)
  - Face recognition runs every 10th frame (expensive ONNX)
  - MediaPipe runs every 3rd frame (medium cost)
  - LLM runs every 5 seconds in a background thread
  - Radar + overlays are pure OpenCV drawing (zero cost)

Press 'Q' to quit.
"""

import cv2
import sys
import os
import time
import threading
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.detector import YoloDetector
from app.vision.tracker import CentroidTracker
from app.vision.action_recognizer import ActionRecognizer
from app.vision.face_recognizer import SentinelFaceRecognizer
from app.intelligence.scene_builder import SceneBuilder
from app.intelligence.ollama_client import OllamaClient
from app.memory.db_client import MemoryDB


# ─── DRAWING HELPERS ───────────────────────────────────────────────

def draw_radar(frame, tracked_objects, radar_size=180):
    """
    Draw a circular radar map in the bottom-right corner.
    Green dots = tracked people, trails show recent movement.
    """
    h, w = frame.shape[:2]
    margin = 15
    cx = w - margin - radar_size // 2
    cy = h - margin - radar_size // 2
    radius = radar_size // 2

    # Semi-transparent dark circle
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, (15, 25, 15), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Radar grid rings
    cv2.circle(frame, (cx, cy), radius, (0, 100, 0), 1)
    cv2.circle(frame, (cx, cy), radius // 2, (0, 60, 0), 1)
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (0, 60, 0), 1)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), (0, 60, 0), 1)
    cv2.putText(frame, "RADAR", (cx - 22, cy - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

    # Map each tracked person onto the radar
    for obj_id, obj in tracked_objects.items():
        # Normalize position to radar space
        norm_x = obj.centroid[0] / w
        norm_y = obj.centroid[1] / h
        rx = int(cx + (norm_x - 0.5) * radar_size * 0.85)
        ry = int(cy + (norm_y - 0.5) * radar_size * 0.85)

        # Draw trail
        for i, pt in enumerate(obj.trail[-15:]):
            pnx = pt[0] / w
            pny = pt[1] / h
            prx = int(cx + (pnx - 0.5) * radar_size * 0.85)
            pry = int(cy + (pny - 0.5) * radar_size * 0.85)
            alpha = int(40 + (i / 15) * 180)
            cv2.circle(frame, (prx, pry), 1, (0, alpha, 0), -1)

        # Draw dot
        cv2.circle(frame, (rx, ry), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"T{obj_id}", (rx + 6, ry - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    return frame


def draw_tracker_labels(frame, tracked_objects, face_names, actions_map):
    """
    Draw persistent Target IDs, names, speed, and actions above each person.
    """
    for obj_id, obj in tracked_objects.items():
        x1, y1, x2, y2 = obj.bbox
        name = face_names.get(obj_id, "")
        action_list = actions_map.get(obj_id, [])

        # Build label
        label = f"Target {obj_id}"
        if name:
            label = f"{name} (T{obj_id})"

        # Speed indicator
        speed_label = ""
        if obj.speed > 25:
            speed_label = " [FAST]"
        elif obj.speed > 8:
            speed_label = " [moving]"

        # Action label
        action_str = ", ".join(action_list) if action_list else ""

        # Draw target ID + name
        cv2.putText(frame, label + speed_label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw action label in red/orange
        if action_str:
            cv2.putText(frame, f"[ {action_str} ]", (x1, max(y1 - 35, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    return frame


def draw_scene_overlay(frame, scene, summary, is_night_vision=False):
    """Draw scene panel and AI summary on the frame."""
    h, w = frame.shape[:2]

    if is_night_vision:
        cv2.putText(frame, "[ NIGHT VISION ]", (w - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Bottom-left: Scene data panel
    panel_x, panel_y = 10, h - 200
    panel_w, panel_h = 440, 190

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"  Targets    : {scene.get('targets', 0)}",
        f"  Identified : {', '.join(scene.get('identities', [])) or 'None'}",
        f"  Density    : {scene['density']}",
        f"  Movement   : {scene['movement']}",
        f"  Actions    : {', '.join(scene.get('actions', [])) or 'None'}",
    ]
    if scene['objects']:
        obj_str = ", ".join(f"{k}:{v}" for k, v in scene['objects'].items())
        lines.append(f"  Objects    : {obj_str}")

    cv2.putText(frame, "[ SCENE ]", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (panel_x + 10, panel_y + 48 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # Top: AI Summary bar
    summary_display = summary if summary else "Waiting for AI summary..."
    if len(summary_display) > 100:
        summary_display = summary_display[:97] + "..."

    bar_h = 40
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 0), (w, bar_h), (10, 10, 60), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "AI: " + summary_display, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 150), 1)

    return frame


# ─── MAIN LOOP ─────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" AI SENTINEL LITE - Phase 6")
    print(" Tracking | Skeletons | Radar | Identity | Memory")
    print("=" * 55)

    # Init all engines
    detector = YoloDetector("yolov8s.pt")
    tracker = CentroidTracker(max_disappeared=20, max_distance=150)
    action_engine = ActionRecognizer()
    face_recognizer = SentinelFaceRecognizer()
    scene_builder = SceneBuilder(movement_threshold=15)
    ollama = OllamaClient()
    db = MemoryDB(persist_directory="chroma_db")

    if not ollama.is_running():
        print("[WARNING] Ollama not running. AI summaries disabled.")
    else:
        print("Ollama connected.")

    # Camera setup
    CAMERA_RGB = 1
    CAMERA_IR = 0
    current_camera = CAMERA_RGB

    cap = cv2.VideoCapture(current_camera)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened. Press 'Q' to quit.\n")

    # State
    current_summary = "Initializing AI..."
    last_summary_time = 0
    SUMMARY_INTERVAL = 5
    generating = False
    frame_count = 0
    last_cam_switch = time.time()

    # Cached results (to avoid running expensive models every frame)
    cached_face_names = {}      # obj_id -> name
    cached_actions = {}         # obj_id -> [action_strings]
    cached_landmarks = None     # for skeleton drawing

    def generate_summary_async(scene):
        nonlocal current_summary, generating
        current_summary = "Thinking..."
        result = ollama.summarize(scene)
        current_summary = result
        db.save_event(result)
        generating = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        # ── AUTO NIGHT VISION ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if (now - last_cam_switch) > 5.0:
            if current_camera == CAMERA_RGB and brightness < 20:
                print(f"[NV] Low light ({brightness:.0f}). Switching to IR.")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_IR)
                current_camera = CAMERA_IR
                last_cam_switch = now
                continue
            elif current_camera == CAMERA_IR and brightness > 120:
                print(f"[NV] Bright ({brightness:.0f}). Switching to RGB.")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_RGB)
                current_camera = CAMERA_RGB
                last_cam_switch = now
                continue

        # ── YOLO DETECTION (every frame) ──
        annotated_frame, counts, raw_result = detector.process_frame(frame)

        # ── TRACKING ──
        person_detections = []
        for box in raw_result.boxes:
            cls_id = int(box.cls[0])
            if detector.model.names[cls_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                person_detections.append((cx, cy, x1, y1, x2, y2))

        tracked = tracker.update(person_detections)

        # ── FACE RECOGNITION (every 10th frame, only if people exist) ──
        if frame_count % 10 == 0 and tracked:
            new_names = {}
            for obj_id, obj in tracked.items():
                x1, y1, x2, y2 = obj.bbox
                x1, y1 = max(0, x1), max(0, y1)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 50:
                    name = face_recognizer.identify(crop)
                    if name and name != "Unknown":
                        new_names[obj_id] = name
            cached_face_names.update(new_names)

        # ── SKELETAL ACTION RECOGNITION (every 3rd frame, only if people) ──
        if frame_count % 3 == 0 and tracked:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks, actions = action_engine.analyze(frame_rgb)
            cached_landmarks = landmarks

            if actions:
                # Assign actions to the closest tracked person
                # For simplicity, assign to the first tracked person
                first_id = next(iter(tracked))
                cached_actions[first_id] = actions
        else:
            landmarks = cached_landmarks

        # ── DRAW SKELETON ──
        if cached_landmarks:
            annotated_frame = action_engine.draw_skeleton(annotated_frame, cached_landmarks)

        # ── DRAW TRACKER LABELS ──
        annotated_frame = draw_tracker_labels(annotated_frame, tracked, cached_face_names, cached_actions)

        # ── DRAW RADAR ──
        annotated_frame = draw_radar(annotated_frame, tracked)

        # ── BUILD SCENE ──
        scene = scene_builder.build(raw_result, detector.model.names)
        scene['targets'] = len(tracked)
        scene['identities'] = list(set(cached_face_names.values()))
        all_actions = set()
        for a_list in cached_actions.values():
            all_actions.update(a_list)
        scene['actions'] = list(all_actions)

        # ── LLM SUMMARY (every N seconds, non-blocking) ──
        if not generating and (now - last_summary_time) >= SUMMARY_INTERVAL:
            generating = True
            last_summary_time = now
            t = threading.Thread(target=generate_summary_async, args=(scene,), daemon=True)
            t.start()

        # ── FINAL COMPOSITE ──
        is_night = (current_camera == CAMERA_IR)
        annotated_frame = draw_scene_overlay(annotated_frame, scene, current_summary, is_night)

        # FPS counter
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow("AI Sentinel Lite", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nShutting down Sentinel...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
