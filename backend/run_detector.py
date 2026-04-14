"""
AI Sentinel Lite - Phase 3: Live Detection + Scene Understanding + AI Summary
------------------------------------------------------------------------------
- YOLOv8s detects people and objects in real time
- SceneBuilder converts detections to structured JSON
- Gemma 2B (via Ollama) generates a 1-sentence natural language summary every 4 seconds
- Everything displayed in a clean OpenCV window

Press 'Q' to quit.
"""

import cv2
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.detector import YoloDetector
from app.intelligence.scene_builder import SceneBuilder
from app.intelligence.ollama_client import OllamaClient


def draw_scene_overlay(frame, scene, summary):
    """Draw scene panel and AI summary on the frame."""
    h, w = frame.shape[:2]

    # --- Bottom-left: Scene data panel ---
    panel_x, panel_y = 10, h - 160
    panel_w, panel_h = 420, 150

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"  People   : {scene['people']}",
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
    # Truncate if too long
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
    print("Starting AI Sentinel Lite - Phase 3...")
    detector = YoloDetector("yolov8s.pt")
    scene_builder = SceneBuilder(movement_threshold=15)
    ollama = OllamaClient()

    if not ollama.is_running():
        print("[WARNING] Ollama is not running. Start it with: ollama serve")
        print("          AI summaries will show a placeholder until Ollama is available.")
    else:
        print("Ollama is running. AI summaries enabled.")

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened. Press 'Q' to quit.")

    current_summary = "Initializing AI..."
    last_summary_time = 0
    SUMMARY_INTERVAL = 4  # seconds between LLM calls
    generating = False

    def generate_summary_async(scene):
        nonlocal current_summary, generating
        current_summary = "Thinking..."
        result = ollama.summarize(scene)
        current_summary = result
        generating = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, counts, raw_result = detector.process_frame(frame)
        scene = scene_builder.build(raw_result, detector.model.names)

        # Trigger a new LLM summary every SUMMARY_INTERVAL seconds (non-blocking)
        now = time.time()
        if not generating and (now - last_summary_time) >= SUMMARY_INTERVAL:
            generating = True
            last_summary_time = now
            t = threading.Thread(target=generate_summary_async, args=(scene,), daemon=True)
            t.start()

        annotated_frame = draw_scene_overlay(annotated_frame, scene, current_summary)
        cv2.imshow("AI Sentinel Lite", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
