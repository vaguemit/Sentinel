import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.face_recognizer import SentinelFaceRecognizer

def main():
    print("=====================================")
    print(" Sentinel - Face Registration Tool")
    print("=====================================")
    
    name = input("Enter the name of the person to register: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    print("Loading models...")
    recognizer = SentinelFaceRecognizer()
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam 1. Trying 0...")
        cap = cv2.VideoCapture(0)
        
    print(f"\nPlease look at the camera. Press SPACE to capture the face for '{name}'.")
    print("Press 'Q' to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw a guide
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//2 - 150, h//2 - 200), (w//2 + 150, h//2 + 200), (0, 255, 0), 2)
        cv2.putText(frame, "Align face inside box & press SPACE", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Registration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            print("Capturing...")
            success = recognizer.register_face(name, frame)
            if success:
                print(f"✅ Success! Face registered for {name}.")
                break
            else:
                print("❌ Could not detect a clear face. Try again.")
        elif key == ord('q'):
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
