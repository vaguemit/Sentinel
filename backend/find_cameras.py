"""
AI Sentinel Lite - Camera Diagnostic Tool
------------------------------------------
Scans camera indices 0-4 and lists which ones work.
Run this first to find out which camera index to use.
"""
import cv2

def find_cameras():
    print("Scanning for available cameras...\n")
    available = []
    for index in range(5):
        # Try with DirectShow (Windows)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  [OK] Camera found at index {index}")
                available.append(index)
            else:
                print(f"  [??] Index {index} opened but no frames")
            cap.release()
        else:
                print(f"  [--] No camera at index {index}")

    print("\n--- Summary ---")
    if available:
        print(f"Working camera indices: {available}")
        print(f"Use index {available[0]} in run_detector.py")
    else:
        print("No cameras found. Check if webcam is plugged in or if another app is using it.")

if __name__ == "__main__":
    find_cameras()
