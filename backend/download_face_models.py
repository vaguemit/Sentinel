import urllib.request
import os

print("Downloading OpenCV ONNX Face Models...")

os.makedirs("models", exist_ok=True)

import requests

print("Downloading OpenCV ONNX Face Models...")

os.makedirs("models", exist_ok=True)

models = {
    "models/face_detection_yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

for path, url in models.items():
    print(f"Downloading {url}...")
    response = requests.get(url, allow_redirects=True)
    with open(path, 'wb') as f:
        f.write(response.content)
    print(f"Saved to {path} ({len(response.content)} bytes)")

print("All models downloaded successfully!")
