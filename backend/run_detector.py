"""
AI Sentinel Lite - Phase 1: Live Webcam Detection
--------------------------------------------------
Run this script directly to open a desktop window showing:
- Live webcam feed
- YOLOv8n bounding boxes
- Detection counts on screen

Press 'Q' to quit.
"""

import cv2
import sys
import os

# Allow importing from the app package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.detector import YoloDetector

def main():
    print("Starting AI Sentinel Lite...")
    detector = YoloDetector("yolov8n.pt")

    # Open webcam using DirectShow backend (avoids MSMF errors on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Make sure it's connected and not in use by another app.")
        return

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLOv8n detection
        annotated_frame, counts = detector.process_frame(frame)

        # Overlay detection summary in top-left corner
        y_offset = 30
        for label, count in counts.items():
            text = f"{label}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        # Show the frame in a native desktop window
        cv2.imshow("AI Sentinel Lite", annotated_frame)

        # Press 'Q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
