import cv2
import warnings
from ultralytics import YOLO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the YOLOv8n model.
        YOLOv8 nano is extremely lightweight and will easily fit in 4GB VRAM.
        When initialized, it will auto-download 'yolov8n.pt' if not present.
        """
        print("Loading YOLOv8n model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def process_frame(self, frame):
        """
        Perform object detection on a single frame.
        Draw bounding boxes and return the annotated frame along with raw class counts.
        """
        # Perform inference
        # verbose=False prevents YOLO from printing stats to console on every frame
        results = self.model(frame, verbose=False)
        
        # The results list contains output for the single batch image
        result = results[0]
        
        # Plot the predictions on the frame
        annotated_frame = result.plot()
        
        # Count the detections by class name
        # We'll use this later for the Scene Builder (Phase 2)
        detections_count = {}
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                detections_count[class_name] = detections_count.get(class_name, 0) + 1
                
        return annotated_frame, detections_count
