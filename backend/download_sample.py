import urllib.request
import os

# A lightweight royalty-free stock video of people walking
VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
OUTPUT_FILE = "sample.mp4"

def download_sample_video():
    if os.path.exists(OUTPUT_FILE):
        print(f"{OUTPUT_FILE} already exists. Skipping download.")
        return

    print(f"Downloading sample video from {VIDEO_URL}...")
    try:
        urllib.request.urlretrieve(VIDEO_URL, OUTPUT_FILE)
        print(f"Successfully downloaded {OUTPUT_FILE}.")
    except Exception as e:
        print(f"Failed to download video. Error: {e}")

if __name__ == "__main__":
    download_sample_video()
