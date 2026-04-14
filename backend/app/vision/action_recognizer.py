"""
AI Sentinel Lite - Skeletal Action Recognition
------------------------------------------------
Uses MediaPipe Pose to extract 33 3D body landmarks from person crops.
Calculates joint angles to classify physical actions.

Optimized: Only runs when YOLO detects a person. Reuses a single
MediaPipe instance across frames to avoid re-initialization overhead.
"""

import math
import numpy as np

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class ActionRecognizer:
    def __init__(self):
        print("Loading MediaPipe Pose model...")
        self.pose = mp_pose.Pose(
            static_image_mode=False,    # Video mode = faster
            model_complexity=0,         # 0 = Lite (fastest, least CPU)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        print("MediaPipe Pose loaded.")

    def _angle(self, a, b, c):
        """Calculate angle at point b given 3 landmarks (a, b, c)."""
        v1 = np.array([a.x - b.x, a.y - b.y])
        v2 = np.array([c.x - b.x, c.y - b.y])
        dot = np.dot(v1, v2)
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        if mag == 0:
            return 0
        cos_angle = np.clip(dot / mag, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def _classify_action(self, landmarks):
        """
        Classify the action based on landmark positions and joint angles.
        Returns a list of detected actions.
        """
        actions = []
        lm = landmarks

        # Key landmarks
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # --- HANDS RAISED ---
        # Wrists are above nose
        if l_wrist.y < nose.y and r_wrist.y < nose.y:
            actions.append("Hands Raised")
        elif l_wrist.y < l_shoulder.y and r_wrist.y < r_shoulder.y:
            actions.append("Arms Up")

        # --- SITTING ---
        # Hip-Knee angle is roughly 90 degrees (between 60 and 130)
        l_hip_angle = self._angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = self._angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2
        if 60 < avg_hip_angle < 130:
            actions.append("Sitting")

        # --- LEANING ---
        # Shoulder midpoint is offset significantly from hip midpoint horizontally
        shoulder_mid_x = (l_shoulder.x + r_shoulder.x) / 2
        hip_mid_x = (l_hip.x + r_hip.x) / 2
        lean = abs(shoulder_mid_x - hip_mid_x)
        if lean > 0.08:
            actions.append("Leaning")

        # --- WAVING ---
        # One wrist above head AND elbow angle is open
        l_elbow_angle = self._angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = self._angle(r_shoulder, r_elbow, r_wrist)
        if (l_wrist.y < nose.y and l_elbow_angle > 140) or \
           (r_wrist.y < nose.y and r_elbow_angle > 140):
            if "Hands Raised" not in actions:
                actions.append("Waving")

        # --- STANDING ---
        # If not sitting and hip angle is wide open (standing upright)
        if avg_hip_angle > 150 and "Sitting" not in actions:
            actions.append("Standing")

        if not actions:
            actions.append("Idle")

        return actions

    def analyze(self, frame_rgb):
        """
        Run pose estimation on an RGB frame.
        Returns: (landmarks_for_drawing, list_of_action_strings) or (None, [])
        """
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None, []

        actions = self._classify_action(results.pose_landmarks.landmark)
        return results.pose_landmarks, actions

    def draw_skeleton(self, frame, landmarks):
        """Draw neon-green skeleton overlay on the frame."""
        if landmarks is None:
            return frame
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        return frame
