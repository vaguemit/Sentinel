# 🛡️ AI Sentinel Lite

> A fully offline, real-time AI surveillance system built for consumer hardware.

**AI Sentinel Lite** transforms your webcam into an intelligent security guard using YOLOv8 object detection, facial recognition, skeletal pose analysis, and local LLM scene summarization — all running entirely on your machine with **zero cloud dependencies**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **YOLOv8s Object Detection** | Real-time person and object detection with bounding boxes |
| 👤 **Facial Identity Recognition** | Register faces and identify known vs unknown individuals |
| 🦴 **Skeletal Pose Estimation** | MediaPipe-powered 33-point body landmark tracking |
| 🤸 **Action Recognition** | Detects sitting, standing, hands raised, waving, leaning |
| 📡 **Spatial Radar Map** | Live top-down radar showing tracked target positions |
| 🆔 **Persistent Object Tracking** | Centroid-based tracker assigns permanent Target IDs |
| 🧠 **Local LLM Summaries** | AI generates natural language scene descriptions via Ollama |
| 💾 **RAG Memory (ChromaDB)** | Vector database stores all observations for later querying |
| 💬 **AI Chat Interface** | Ask questions about past events: *"Did anyone carry a backpack?"* |
| 🔥 **Density Heatmap** | Thermal overlay showing high-traffic zones (toggle with `H`) |
| 🚧 **Virtual Restricted Zones** | Draw zones on-screen; intrusions trigger red alerts (press `Z`) |
| 📸 **Auto-Screenshot on Anomaly** | Captures evidence when unknown people or fast movement detected |
| 📊 **Live Analytics Dashboard** | Real-time graph: people count timeline, FPS, event counter |
| 🌙 **Auto Night Vision** | Switches to IR camera when lights go off |
| ⚡ **Multi-Threaded Pipeline** | Face/skeleton processing offloaded to background thread for higher FPS |

---

## 🏗️ Architecture

```
Webcam Feed
    │
    ├── [Thread: Main] ──► YOLOv8s Detection ──► Centroid Tracker ──► Overlays & Display
    │
    ├── [Thread: Worker] ──► Face Recognition (ONNX) ──► Skeleton Analysis (MediaPipe)
    │
    ├── [Thread: LLM] ──► Ollama (qwen2.5:1.5b) ──► Scene Summary
    │
    └── [Thread: LLM] ──► ChromaDB ──► RAG Memory Storage
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed with a model pulled (e.g., `ollama pull qwen2.5:1.5b`)
- A webcam

### Installation

```bash
git clone https://github.com/vaguemit/Sentinel.git
cd Sentinel
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r backend/requirements.txt
```

### Download Face Models

```bash
cd backend
python download_face_models.py
```

### Register Your Face

```bash
python register_face.py
# Enter your name, align face in the green box, press SPACE
```

### Run Sentinel

```bash
python run_detector.py
```

### Chat with Memory (separate terminal)

```bash
python chat.py
# Ask: "What did you see in the last 5 minutes?"
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `H` | Toggle density heatmap overlay |
| `Z` | Enter zone-drawing mode (click + drag to define restricted area) |

---

## 📂 Project Structure

```
Sentinel/
├── backend/
│   ├── app/
│   │   ├── intelligence/
│   │   │   ├── ollama_client.py     # LLM interface (Ollama)
│   │   │   └── scene_builder.py     # Structured scene JSON builder
│   │   ├── memory/
│   │   │   └── db_client.py         # ChromaDB vector storage
│   │   └── vision/
│   │       ├── detector.py          # YOLOv8s wrapper
│   │       ├── tracker.py           # Centroid multi-object tracker
│   │       ├── face_recognizer.py   # OpenCV DNN face ID (YuNet + SFace)
│   │       └── action_recognizer.py # MediaPipe pose + action classifier
│   ├── models/                      # ONNX face models + MediaPipe task
│   ├── captures/                    # Auto-saved anomaly screenshots
│   ├── chroma_db/                   # Persistent vector database
│   ├── run_detector.py              # Main surveillance engine
│   ├── chat.py                      # RAG chat CLI
│   ├── register_face.py             # Face enrollment tool
│   ├── download_face_models.py      # Model downloader
│   └── requirements.txt
└── README.md
```

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | Not required | 4 GB (for future CUDA acceleration) |
| CPU | 4 cores | 8+ cores |
| Webcam | Any USB webcam | 720p+ |

> **Note:** The system currently runs entirely on CPU. GPU acceleration via CUDA-enabled PyTorch is supported but requires Python 3.12 or earlier for stable wheels.

---

## 🛣️ Roadmap

- [x] Phase 1: YOLOv8 detection pipeline
- [x] Phase 2: Scene understanding (SceneBuilder)
- [x] Phase 3: LLM integration (Ollama)
- [x] Phase 4: RAG memory (ChromaDB)
- [x] Phase 5: Face recognition + Night vision
- [x] Phase 6: Tracking, skeletons, radar, actions
- [x] Phase 7: Heatmap, zones, analytics, threading
- [ ] Phase 8: React web dashboard
- [ ] Phase 9: Telegram/Email alerts
- [ ] Phase 10: Groq API integration for cloud-speed inference

---

## 📜 License

MIT License

---

Built with ❤️ by [vaguemit](https://github.com/vaguemit)
