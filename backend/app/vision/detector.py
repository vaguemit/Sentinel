import cv2
import warnings
from ultralytics import YOLO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class YoloDetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.55):
        """
        Initialize the YOLOv8s model on GPU.
        yolov8s (small) is significantly more accurate than nano
        and runs comfortably on a 3050 Ti at 30+ FPS.
        conf: confidence threshold (0.0-1.0). Lower = more detections, Higher = fewer false positives.
        """
        self.conf = conf
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading YOLOv8s model on {device.upper()}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"Model loaded successfully on {device.upper()}.")

    def process_frame(self, frame):
        """
        Perform object detection on a single frame.
        Draw bounding boxes and return the annotated frame along with raw class counts.
        """
        # Perform inference
        # verbose=False prevents YOLO from printing stats to console on every frame
        results = self.model(frame, verbose=False, conf=self.conf)
        
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
                
        return annotated_frame, detections_count, result
