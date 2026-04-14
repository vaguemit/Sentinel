import urllib.request
import os

print("Downloading OpenCV ONNX Face Models...")

os.makedirs("models", exist_ok=True)

models = {
    "models/face_detection_yunet.onnx": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface.onnx": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

for path, url in models.items():
    if not os.path.exists(path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
    else:
        print(f"File {path} already exists. Skipping.")

print("All models downloaded successfully!")
