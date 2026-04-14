# AI Sentinel Lite

AI Sentinel Lite is a real-time, offline AI surveillance system that detects people and objects from video feeds and generates concise, human-readable scene summaries using a local language model.

## Features (Phase 1)
- Real-time object detection using YOLOv8n.
- FastAPI backend serving MJPEG video feed.

## Setup Instructions

1. Install Python dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Download a sample video:
   ```bash
   python download_sample.py
   ```
3. Run the backend:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
