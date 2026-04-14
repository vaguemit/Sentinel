"""
AI Sentinel Lite - Phase 2: Live Detection + Scene Understanding
--------------------------------------------------
Run this script to open a desktop window showing:
- Live webcam feed with YOLOv8n bounding boxes
- Structured scene data overlay (people count, density, movement)

Press 'Q' to quit.
"""

import cv2
import sys
import os

# Allow importing from the app package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.detector import YoloDetector
from app.intelligence.scene_builder import SceneBuilder


def draw_scene_overlay(frame, scene):
    """
    Draw a clean scene summary panel in the bottom-left corner.
    """
    h, w = frame.shape[:2]
    panel_x, panel_y = 10, h - 150
    panel_w, panel_h = 400, 140

    # Semi-transparent dark background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Scene data lines
    lines = [
        f"  People   : {scene['people']}",
        f"  Density  : {scene['density']}",
        f"  Movement : {scene['movement']}",
    ]
    if scene['objects']:
        obj_str = ", ".join(f"{k}:{v}" for k, v in scene['objects'].items())
        lines.append(f"  Objects  : {obj_str}")

    # Title
    cv2.putText(frame, "[ SCENE ]", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Content lines
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (panel_x + 10, panel_y + 45 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame


def main():
    print("Starting AI Sentinel Lite - Phase 2...")
    detector = YoloDetector("yolov8n.pt")
    scene_builder = SceneBuilder(movement_threshold=15)

    # Camera 0 = IR camera (Windows Hello), Camera 1 = real RGB webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLOv8n detection — returns annotated frame, counts, and raw result
        annotated_frame, counts, raw_result = detector.process_frame(frame)

        # Build structured scene data from raw result
        scene = scene_builder.build(raw_result, detector.model.names)

        # Overlay scene data panel on the frame
        annotated_frame = draw_scene_overlay(annotated_frame, scene)

        # Show the frame
        cv2.imshow("AI Sentinel Lite", annotated_frame)

        # Press 'Q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
