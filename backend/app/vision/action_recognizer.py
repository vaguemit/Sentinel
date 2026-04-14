"""
AI Sentinel Lite - Skeletal Action Recognition (Tasks API)
------------------------------------------------------------
Uses MediaPipe Tasks PoseLandmarker to extract 33 body landmarks.
Calculates joint angles to classify physical actions.

Optimized: model_complexity=lite, only runs when YOLO detects a person.
"""

import math
import os
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Landmark indices (same as the old PoseLandmark enum)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
    (11, 23), (12, 24), (23, 24),                        # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # Legs
    (0, 11), (0, 12),                                     # Head to shoulders
]


class ActionRecognizer:
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        print("Loading MediaPipe Pose Landmarker (Tasks API)...")
        
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        print("MediaPipe Pose Landmarker loaded.")

    def _angle(self, a, b, c):
        """Calculate angle at point b given 3 landmarks."""
        v1 = np.array([a.x - b.x, a.y - b.y])
        v2 = np.array([c.x - b.x, c.y - b.y])
        dot = np.dot(v1, v2)
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        if mag == 0:
            return 0
        cos_angle = np.clip(dot / mag, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def _classify_action(self, landmarks):
        """Classify the action based on landmark positions and joint angles."""
        actions = []
        lm = landmarks

        nose = lm[NOSE]
        l_shoulder = lm[LEFT_SHOULDER]
        r_shoulder = lm[RIGHT_SHOULDER]
        l_elbow = lm[LEFT_ELBOW]
        r_elbow = lm[RIGHT_ELBOW]
        l_wrist = lm[LEFT_WRIST]
        r_wrist = lm[RIGHT_WRIST]
        l_hip = lm[LEFT_HIP]
        r_hip = lm[RIGHT_HIP]
        l_knee = lm[LEFT_KNEE]
        r_knee = lm[RIGHT_KNEE]

        # --- HANDS RAISED ---
        if l_wrist.y < nose.y and r_wrist.y < nose.y:
            actions.append("Hands Raised")
        elif l_wrist.y < l_shoulder.y and r_wrist.y < r_shoulder.y:
            actions.append("Arms Up")

        # --- SITTING ---
        l_hip_angle = self._angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = self._angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2
        if 60 < avg_hip_angle < 130:
            actions.append("Sitting")

        # --- LEANING ---
        shoulder_mid_x = (l_shoulder.x + r_shoulder.x) / 2
        hip_mid_x = (l_hip.x + r_hip.x) / 2
        lean = abs(shoulder_mid_x - hip_mid_x)
        if lean > 0.08:
            actions.append("Leaning")

        # --- WAVING ---
        l_elbow_angle = self._angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = self._angle(r_shoulder, r_elbow, r_wrist)
        if (l_wrist.y < nose.y and l_elbow_angle > 140) or \
           (r_wrist.y < nose.y and r_elbow_angle > 140):
            if "Hands Raised" not in actions:
                actions.append("Waving")

        # --- STANDING ---
        if avg_hip_angle > 150 and "Sitting" not in actions:
            actions.append("Standing")

        if not actions:
            actions.append("Idle")

        return actions

    def analyze(self, frame_rgb):
        """
        Run pose estimation on an RGB frame (numpy array).
        Returns: (list_of_landmark_lists, list_of_action_strings)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None, []

        # Use the first detected pose
        landmarks = result.pose_landmarks[0]
        actions = self._classify_action(landmarks)
        return landmarks, actions

    def draw_skeleton(self, frame, landmarks):
        """Draw neon-green skeleton overlay on the frame."""
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            # Only draw if both points are visible enough
            if start.visibility < 0.4 or end.visibility < 0.4:
                continue

            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 100), 2)

        # Draw landmark dots
        for lm in landmarks:
            if lm.visibility < 0.4:
                continue
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 3, (0, 200, 255), -1)

        return frame
