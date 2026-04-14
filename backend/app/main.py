import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import os
import uvicorn
from app.vision.detector import YoloDetector

app = FastAPI(title="AI Sentinel Lite", version="1.0")

# Initialize the detector
# This will load the model into GPU memory once on startup
detector = YoloDetector("yolov8n.pt")

# Determine video source
# Fallback to local webcam (0) if sample.mp4 doesn't exist
VIDEO_SOURCE = "sample.mp4"
if not os.path.exists(VIDEO_SOURCE):
    print(f"[{VIDEO_SOURCE}] not found. Falling back to webcam (0).")
    VIDEO_SOURCE = 0

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            # If video ends, loop it (useful for sample.mp4)
            if isinstance(VIDEO_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
                
        # Run detection
        annotated_frame, counts = detector.process_frame(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield as multi-part for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def read_root():
    return {"message": "AI Sentinel Lite is running. Access /video_feed for the stream."}

@app.get("/video_feed")
def video_feed():
    """
    Returns an MJPEG stream of the annotated video.
    Can be viewed directly in a browser.
    """
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
