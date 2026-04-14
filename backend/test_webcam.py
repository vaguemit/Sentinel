"""
Minimal webcam test - no YOLO, just raw cv2.imshow
If you see yourself in the window, the webcam works.
Press Q to quit.
"""
import cv2

for idx in [1, 0]:
    print(f"Trying camera index {idx}...")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"  -> Could not open index {idx}")
        continue

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  -> Opened but no frame from index {idx}")
        cap.release()
        continue

    print(f"  -> SUCCESS: Camera {idx} is working! Showing feed... Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Webcam Test - Index {idx}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    break  # stop after first working camera
