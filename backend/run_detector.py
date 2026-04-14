"""
AI Sentinel Lite - Full Stack Surveillance Engine
------------------------------------------------------------
- YOLOv8s object detection
- Centroid multi-object tracking with persistent IDs + radar
- MediaPipe skeletal pose estimation + action recognition
- OpenCV DNN face identification (friend vs foe)
- Qwen 2.5 LLM scene summarization via Ollama
- ChromaDB RAG memory
- Auto night vision (IR camera switching)
- Real-time density heatmap (toggle with H)
- Virtual restricted zones with intrusion alerts
- Auto-screenshot on anomaly detection
- Live analytics graph (people count, FPS, events)

OPTIMIZATION STRATEGY (4GB VRAM / CPU-only):
  - YOLO runs every frame (lightweight on CPU)
  - Face recognition runs every 10th frame (expensive ONNX)
  - MediaPipe runs every 3rd frame (medium cost)
  - LLM runs every 5 seconds in a background thread
  - Radar + overlays + heatmap are pure OpenCV drawing (zero cost)

Press 'Q' to quit. Press 'H' to toggle heatmap. Press 'Z' to define a zone.
"""

import cv2
import sys
import os
import time
import threading
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vision.detector import YoloDetector
from app.vision.tracker import CentroidTracker
from app.vision.action_recognizer import ActionRecognizer
from app.vision.face_recognizer import SentinelFaceRecognizer
from app.intelligence.scene_builder import SceneBuilder
from app.intelligence.ollama_client import OllamaClient
from app.memory.db_client import MemoryDB


# ─── HEATMAP ENGINE ────────────────────────────────────────────────

class DensityHeatmap:
    """Accumulates person positions over time into a thermal heatmap."""
    def __init__(self, width=1280, height=720, decay=0.995):
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self.decay = decay  # Slow decay so old positions fade gradually

    def update(self, tracked_objects):
        """Add current positions to the accumulator."""
        # Decay old values slightly each frame
        self.accumulator *= self.decay

        for obj_id, obj in tracked_objects.items():
            cx, cy = int(obj.centroid[0]), int(obj.centroid[1])
            h, w = self.accumulator.shape
            if 0 <= cx < w and 0 <= cy < h:
                # Gaussian-like splat (fast approximation with cv2.circle)
                cv2.circle(self.accumulator, (cx, cy), 40, 1.0, -1)

    def render(self, frame):
        """Render the heatmap as a translucent overlay on the frame."""
        h, w = frame.shape[:2]
        acc = self.accumulator[:h, :w]

        # Normalize to 0-255
        max_val = acc.max()
        if max_val < 1:
            return frame

        normalized = (acc / max_val * 255).astype(np.uint8)
        # Apply colormap: COLORMAP_JET gives blue(cold) -> red(hot)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Only overlay where there's actual heat (avoid blue wash everywhere)
        mask = normalized > 10
        mask_3ch = np.stack([mask, mask, mask], axis=-1)

        blended = frame.copy()
        blended[mask_3ch] = cv2.addWeighted(frame, 0.5, colored, 0.5, 0)[mask_3ch]

        # Label
        cv2.putText(blended, "[ HEATMAP ON ]", (w - 180, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        return blended


# ─── ZONE ENGINE ───────────────────────────────────────────────────

class ZoneManager:
    """Manages virtual restricted zones drawn by the user."""
    def __init__(self):
        self.zones = []        # List of (x1, y1, x2, y2) tuples
        self.intrusions = {}   # zone_idx -> list of target IDs
        self.alert_flash = 0   # Countdown frames for red flash

    def add_zone(self, x1, y1, x2, y2):
        self.zones.append((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
        print(f"Zone {len(self.zones)} added: ({x1},{y1}) to ({x2},{y2})")

    def check_intrusions(self, tracked_objects, face_names):
        """Check if any tracked object is inside a restricted zone."""
        self.intrusions = {}
        for zone_idx, (zx1, zy1, zx2, zy2) in enumerate(self.zones):
            for obj_id, obj in tracked_objects.items():
                cx, cy = obj.centroid
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    if zone_idx not in self.intrusions:
                        self.intrusions[zone_idx] = []
                    name = face_names.get(obj_id, f"Target {obj_id}")
                    self.intrusions[zone_idx].append(name)
                    self.alert_flash = 15  # Flash for 15 frames

        return self.intrusions

    def draw_zones(self, frame):
        """Draw zone boundaries and flash red on intrusion."""
        for zone_idx, (zx1, zy1, zx2, zy2) in enumerate(self.zones):
            if zone_idx in self.intrusions:
                # INTRUSION: Red zone with flashing
                color = (0, 0, 255)
                thickness = 3
                cv2.putText(frame, f"!! INTRUSION Zone {zone_idx+1} !!",
                            (zx1, max(zy1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Normal: Green dashed boundary
                color = (0, 200, 0)
                thickness = 1
                cv2.putText(frame, f"Zone {zone_idx+1}",
                            (zx1, max(zy1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, thickness)

        # Red border flash on any intrusion
        if self.alert_flash > 0:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 4)
            self.alert_flash -= 1

        return frame


# ─── AUTO-SCREENSHOT ENGINE ────────────────────────────────────────

class AnomalyCapture:
    """Automatically saves screenshots when anomalies are detected."""
    def __init__(self, capture_dir="captures"):
        self.capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)
        self.last_capture_time = 0
        self.cooldown = 10  # seconds between captures

    def check_and_capture(self, frame, tracked, face_names, actions_map, zone_intrusions):
        """Check for anomaly conditions and save screenshot if triggered."""
        now = time.time()
        if now - self.last_capture_time < self.cooldown:
            return None

        reason = None

        # Anomaly 1: Unknown person detected
        for obj_id in tracked:
            if obj_id not in face_names:
                reason = f"Unknown_person_Target{obj_id}"
                break

        # Anomaly 2: Someone running fast
        for obj_id, obj in tracked.items():
            if obj.speed > 30:
                name = face_names.get(obj_id, f"Target{obj_id}")
                reason = f"Fast_movement_{name}"
                break

        # Anomaly 3: Zone intrusion
        if zone_intrusions:
            for zone_idx, intruders in zone_intrusions.items():
                reason = f"Zone{zone_idx+1}_intrusion_{'_'.join(intruders)}"
                break

        if reason:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{reason}.jpg"
            filepath = os.path.join(self.capture_dir, filename)
            cv2.imwrite(filepath, frame)
            self.last_capture_time = now
            print(f"[CAPTURE] Anomaly screenshot saved: {filepath}")
            return filepath

        return None


# ─── LIVE ANALYTICS ENGINE ─────────────────────────────────────────

class LiveAnalytics:
    """Tracks and renders real-time analytics: people count history, FPS, events."""
    def __init__(self, history_seconds=60):
        self.people_history = []     # (timestamp, count) tuples
        self.event_count = 0
        self.history_seconds = history_seconds
        self.fps_samples = []
        self.last_frame_time = time.time()

    def update(self, people_count):
        now = time.time()
        # FPS
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_samples.append(1.0 / dt)
        if len(self.fps_samples) > 30:
            self.fps_samples.pop(0)

        # People count history
        self.people_history.append((now, people_count))
        # Prune old entries
        cutoff = now - self.history_seconds
        self.people_history = [(t, c) for t, c in self.people_history if t >= cutoff]

    def log_event(self):
        self.event_count += 1

    def get_fps(self):
        if not self.fps_samples:
            return 0
        return sum(self.fps_samples) / len(self.fps_samples)

    def draw(self, frame):
        """Draw a mini analytics panel in the top-right corner."""
        h, w = frame.shape[:2]
        graph_w, graph_h = 220, 100
        margin = 10
        gx = w - graph_w - margin
        gy = 50  # Below the AI summary bar

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (gx - 5, gy - 20), (gx + graph_w + 5, gy + graph_h + 35),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        fps = self.get_fps()
        cv2.putText(frame, f"ANALYTICS  FPS: {fps:.0f}", (gx, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        # Draw people count graph
        if len(self.people_history) > 1:
            max_count = max(c for _, c in self.people_history)
            if max_count == 0:
                max_count = 1

            points = []
            now = time.time()
            for t, c in self.people_history:
                px = int(gx + ((t - (now - self.history_seconds)) / self.history_seconds) * graph_w)
                py = int(gy + graph_h - (c / max_count) * graph_h)
                points.append((px, py))

            # Draw the line
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 200, 255), 1)

            # Fill area under the line
            fill_pts = points + [(points[-1][0], gy + graph_h), (points[0][0], gy + graph_h)]
            fill_overlay = frame.copy()
            cv2.fillPoly(fill_overlay, [np.array(fill_pts)], (0, 80, 120))
            cv2.addWeighted(fill_overlay, 0.3, frame, 0.7, 0, frame)

        # Axis labels
        cv2.putText(frame, "People", (gx, gy + graph_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(frame, f"Events: {self.event_count}", (gx + 100, gy + graph_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(frame, f"60s", (gx + graph_w - 20, gy + graph_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # Graph border
        cv2.rectangle(frame, (gx, gy), (gx + graph_w, gy + graph_h), (50, 50, 50), 1)

        return frame


# ─── DRAWING HELPERS ───────────────────────────────────────────────

def draw_radar(frame, tracked_objects, radar_size=180):
    """
    Draw a circular radar map in the bottom-right corner.
    Green dots = tracked people, trails show recent movement.
    """
    h, w = frame.shape[:2]
    margin = 15
    cx = w - margin - radar_size // 2
    cy = h - margin - radar_size // 2
    radius = radar_size // 2

    # Semi-transparent dark circle
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, (15, 25, 15), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Radar grid rings
    cv2.circle(frame, (cx, cy), radius, (0, 100, 0), 1)
    cv2.circle(frame, (cx, cy), radius // 2, (0, 60, 0), 1)
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (0, 60, 0), 1)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), (0, 60, 0), 1)
    cv2.putText(frame, "RADAR", (cx - 22, cy - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

    # Map each tracked person onto the radar
    for obj_id, obj in tracked_objects.items():
        # Normalize position to radar space
        norm_x = obj.centroid[0] / w
        norm_y = obj.centroid[1] / h
        rx = int(cx + (norm_x - 0.5) * radar_size * 0.85)
        ry = int(cy + (norm_y - 0.5) * radar_size * 0.85)

        # Draw trail
        for i, pt in enumerate(obj.trail[-15:]):
            pnx = pt[0] / w
            pny = pt[1] / h
            prx = int(cx + (pnx - 0.5) * radar_size * 0.85)
            pry = int(cy + (pny - 0.5) * radar_size * 0.85)
            alpha = int(40 + (i / 15) * 180)
            cv2.circle(frame, (prx, pry), 1, (0, alpha, 0), -1)

        # Draw dot
        cv2.circle(frame, (rx, ry), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"T{obj_id}", (rx + 6, ry - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    return frame


def draw_tracker_labels(frame, tracked_objects, face_names, actions_map):
    """
    Draw persistent Target IDs, names, speed, and actions above each person.
    """
    for obj_id, obj in tracked_objects.items():
        x1, y1, x2, y2 = obj.bbox
        name = face_names.get(obj_id, "")
        action_list = actions_map.get(obj_id, [])

        # Build label
        label = f"Target {obj_id}"
        if name:
            label = f"{name} (T{obj_id})"

        # Speed indicator
        speed_label = ""
        if obj.speed > 25:
            speed_label = " [FAST]"
        elif obj.speed > 8:
            speed_label = " [moving]"

        # Action label
        action_str = ", ".join(action_list) if action_list else ""

        # Draw target ID + name
        cv2.putText(frame, label + speed_label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw action label in red/orange
        if action_str:
            cv2.putText(frame, f"[ {action_str} ]", (x1, max(y1 - 35, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    return frame


def draw_scene_overlay(frame, scene, summary, is_night_vision=False):
    """Draw scene panel and AI summary on the frame."""
    h, w = frame.shape[:2]

    if is_night_vision:
        cv2.putText(frame, "[ NIGHT VISION ]", (w - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Bottom-left: Scene data panel
    panel_x, panel_y = 10, h - 200
    panel_w, panel_h = 440, 190

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"  Targets    : {scene.get('targets', 0)}",
        f"  Identified : {', '.join(scene.get('identities', [])) or 'None'}",
        f"  Density    : {scene['density']}",
        f"  Movement   : {scene['movement']}",
        f"  Actions    : {', '.join(scene.get('actions', [])) or 'None'}",
    ]
    if scene['objects']:
        obj_str = ", ".join(f"{k}:{v}" for k, v in scene['objects'].items())
        lines.append(f"  Objects    : {obj_str}")

    cv2.putText(frame, "[ SCENE ]", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (panel_x + 10, panel_y + 48 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # Top: AI Summary bar
    summary_display = summary if summary else "Waiting for AI summary..."
    if len(summary_display) > 100:
        summary_display = summary_display[:97] + "..."

    bar_h = 40
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 0), (w, bar_h), (10, 10, 60), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "AI: " + summary_display, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 150), 1)

    return frame


# ─── MAIN LOOP (MULTI-THREADED PIPELINE) ───────────────────────────

def main():
    print("=" * 60)
    print(" AI SENTINEL LITE")
    print(" Tracking | Skeletons | Radar | Identity | Memory | Heatmap")
    print(" Zones | Auto-Capture | Analytics | Threaded Pipeline")
    print("=" * 60)

    # Init all engines
    detector = YoloDetector("yolov8s.pt")
    tracker = CentroidTracker(max_disappeared=20, max_distance=150)
    action_engine = ActionRecognizer()
    face_recognizer = SentinelFaceRecognizer()
    scene_builder = SceneBuilder(movement_threshold=15)
    ollama = OllamaClient()
    db = MemoryDB(persist_directory="chroma_db")

    if not ollama.is_running():
        print("[WARNING] Ollama not running. AI summaries disabled.")
    else:
        print("Ollama connected.")

    # Camera setup
    CAMERA_RGB = 1
    CAMERA_IR = 0
    current_camera = CAMERA_RGB

    cap = cv2.VideoCapture(current_camera)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened. Press 'Q' to quit. Press 'H' to toggle heatmap.\n")

    # State
    current_summary = "Initializing AI..."
    last_summary_time = 0
    SUMMARY_INTERVAL = 5
    generating = False
    frame_count = 0
    last_cam_switch = time.time()

    # Cached results (to avoid running expensive models every frame)
    cached_face_names = {}      # obj_id -> name
    cached_actions = {}         # obj_id -> [action_strings]
    cached_landmarks = None     # for skeleton drawing
    face_skeleton_lock = threading.Lock()

    # Heatmap
    heatmap = DensityHeatmap(width=1280, height=720)
    show_heatmap = False

    # Zones & Screenshots
    zone_mgr = ZoneManager()
    anomaly_capture = AnomalyCapture()
    drawing_zone = False
    zone_start = None

    # Analytics
    analytics = LiveAnalytics(history_seconds=60)

    # Mouse callback for zone drawing
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_zone, zone_start
        if event == cv2.EVENT_LBUTTONDOWN and drawing_zone:
            zone_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing_zone and zone_start:
            zone_mgr.add_zone(zone_start[0], zone_start[1], x, y)
            drawing_zone = False
            zone_start = None
            print("Zone defined! Press Z to add another.")

    # Background face + skeleton worker
    from collections import deque
    face_skel_queue = deque(maxlen=1)
    worker_running = True

    def face_skeleton_worker():
        nonlocal cached_face_names, cached_actions, cached_landmarks, worker_running
        local_count = 0
        while worker_running:
            if not face_skel_queue:
                time.sleep(0.01)
                continue
            work = face_skel_queue.pop()
            frame = work['frame']
            tracked = work['tracked']
            local_count += 1
            # Face recognition (every 5th worker cycle)
            if local_count % 5 == 0 and tracked:
                new_names = {}
                for obj_id, obj in tracked.items():
                    x1, y1, x2, y2 = obj.bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] > 50:
                        name = face_recognizer.identify(crop)
                        if name and name != "Unknown":
                            new_names[obj_id] = name
                with face_skeleton_lock:
                    cached_face_names.update(new_names)
            # Skeleton (every 2nd worker cycle)
            if local_count % 2 == 0 and tracked:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks, actions = action_engine.analyze(frame_rgb)
                with face_skeleton_lock:
                    cached_landmarks = landmarks
                    if actions and tracked:
                        first_id = next(iter(tracked))
                        cached_actions[first_id] = actions

    cv2.namedWindow("AI Sentinel Lite")
    cv2.setMouseCallback("AI Sentinel Lite", mouse_callback)

    def generate_summary_async(scene):
        nonlocal current_summary, generating
        current_summary = "Thinking..."
        result = ollama.summarize(scene)
        current_summary = result
        db.save_event(result)
        generating = False

    # Start background worker
    t_worker = threading.Thread(target=face_skeleton_worker, daemon=True)
    t_worker.start()
    print("[PIPELINE] Background face/skeleton worker started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        # ── AUTO NIGHT VISION ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if (now - last_cam_switch) > 5.0:
            if current_camera == CAMERA_RGB and brightness < 20:
                print(f"[NV] Low light ({brightness:.0f}). Switching to IR.")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_IR)
                current_camera = CAMERA_IR
                last_cam_switch = now
                continue
            elif current_camera == CAMERA_IR and brightness > 120:
                print(f"[NV] Bright ({brightness:.0f}). Switching to RGB.")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_RGB)
                current_camera = CAMERA_RGB
                last_cam_switch = now
                continue

        # ── YOLO DETECTION (every frame) ──
        annotated_frame, counts, raw_result = detector.process_frame(frame)

        # ── TRACKING ──
        person_detections = []
        for box in raw_result.boxes:
            cls_id = int(box.cls[0])
            if detector.model.names[cls_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                person_detections.append((cx, cy, x1, y1, x2, y2))

        tracked = tracker.update(person_detections)

        # ── OFFLOAD TO BACKGROUND WORKER ──
        if tracked:
            face_skel_queue.append({'frame': frame.copy(), 'tracked': dict(tracked)})

        # ── READ CACHED RESULTS (thread-safe) ──
        with face_skeleton_lock:
            local_landmarks = cached_landmarks
            local_face_names = dict(cached_face_names)
            local_actions = dict(cached_actions)

        # ── DRAW SKELETON ──
        if local_landmarks:
            annotated_frame = action_engine.draw_skeleton(annotated_frame, local_landmarks)

        # ── DRAW TRACKER LABELS ──
        annotated_frame = draw_tracker_labels(annotated_frame, tracked, local_face_names, local_actions)

        # ── DRAW RADAR ──
        annotated_frame = draw_radar(annotated_frame, tracked)

        # ── HEATMAP ──
        heatmap.update(tracked)
        if show_heatmap:
            annotated_frame = heatmap.render(annotated_frame)

        # ── ZONES ──
        zone_intrusions = zone_mgr.check_intrusions(tracked, cached_face_names)
        annotated_frame = zone_mgr.draw_zones(annotated_frame)

        # ── BUILD SCENE ──
        scene = scene_builder.build(raw_result, detector.model.names)
        scene['targets'] = len(tracked)
        scene['identities'] = list(set(local_face_names.values()))
        all_actions = set()
        for a_list in local_actions.values():
            all_actions.update(a_list)
        scene['actions'] = list(all_actions)
        if zone_intrusions:
            scene['zone_alert'] = True

        # ── AUTO SCREENSHOT ──
        captured = anomaly_capture.check_and_capture(
            annotated_frame, tracked, local_face_names, local_actions, zone_intrusions
        )
        if captured:
            analytics.log_event()

        # ── ANALYTICS ──
        analytics.update(len(tracked))
        annotated_frame = analytics.draw(annotated_frame)

        # ── LLM SUMMARY (every N seconds, non-blocking) ──
        if not generating and (now - last_summary_time) >= SUMMARY_INTERVAL:
            generating = True
            last_summary_time = now
            t = threading.Thread(target=generate_summary_async, args=(scene,), daemon=True)
            t.start()

        # ── FINAL COMPOSITE ──
        is_night = (current_camera == CAMERA_IR)
        annotated_frame = draw_scene_overlay(annotated_frame, scene, current_summary, is_night)

        cv2.imshow("AI Sentinel Lite", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nShutting down Sentinel...")
            break
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"Heatmap: {'ON' if show_heatmap else 'OFF'}")
        elif key == ord('z'):
            drawing_zone = True
            print("Click and drag on the video to define a restricted zone...")

    worker_running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
