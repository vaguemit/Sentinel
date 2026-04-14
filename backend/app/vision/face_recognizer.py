import cv2
import numpy as np
import os
import json

class SentinelFaceRecognizer:
    def __init__(self, models_dir="models", db_path="faces.json"):
        detector_path = os.path.join(models_dir, "face_detection_yunet.onnx")
        recognizer_path = os.path.join(models_dir, "face_recognition_sface.onnx")
        
        self.db_path = db_path
        self.faces_db = self._load_db()

        # Initialize YuNet for face detection
        self.detector = cv2.FaceDetectorYN.create(
            model=detector_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000
        )
        
        # Initialize SFace for recognition
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=recognizer_path,
            config=""
        )

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    # Convert lists back to numpy arrays
                    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
            except Exception as e:
                print(f"Error loading face DB: {e}")
        return {}

    def save_db(self):
        data = {k: v.tolist() for k, v in self.faces_db.items()}
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=4)

    def register_face(self, name, frame):
        """Extract encoding from a frame and save it under 'name'."""
        encoding = self.get_encoding(frame)
        if encoding is not None:
            self.faces_db[name] = encoding
            self.save_db()
            return True
        return False

    def get_encoding(self, frame):
        """Detect face and return its 128D encoding using SFace."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        
        # YuNet expects BGR
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
            
        # Get the largest face
        # faces format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        
        # Align and extract
        aligned_face = self.recognizer.alignCrop(frame, largest_face)
        feature = self.recognizer.feature(aligned_face)
        return feature[0]

    def identify(self, frame):
        """Returns the name of the recognized person, or 'Unknown'."""
        if not self.faces_db:
            return "Unknown"
            
        encoding = self.get_encoding(frame)
        if encoding is None:
            return None
            
        best_match = "Unknown"
        best_score = 0.0  # Cosine similarity
        
        for name, db_encoding in self.faces_db.items():
            # SFace recommended distance threshold is 0.363 for Cosine
            score = self.recognizer.match(encoding, db_encoding, cv2.FaceRecognizerSF_FR_COSINE)
            if score > 0.363 and score > best_score:
                best_score = score
                best_match = name
                
        return best_match
