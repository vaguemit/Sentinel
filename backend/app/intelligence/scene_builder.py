"""
AI Sentinel Lite - Phase 2: Scene Builder
------------------------------------------
Converts raw YOLO detections into structured scene data:
{
    "people": 2,
    "objects": {"backpack": 1},
    "density": "low",
    "movement": "one moving, one stationary"
}

Uses centroid tracking across frames to determine movement.
"""

import math


class SceneBuilder:
    def __init__(self, movement_threshold=15, history_size=10):
        """
        movement_threshold: Minimum pixel distance between frames to count as "moving"
        history_size: Number of frames to keep centroid history for
        """
        self.movement_threshold = movement_threshold
        self.history_size = history_size
        # Stores previous centroids: list of (cx, cy) per frame
        self._centroid_history = []

    def _get_centroids(self, boxes, model_names):
        """Extract centroids for 'person' detections only."""
        centroids = []
        for box in boxes:
            class_id = int(box.cls[0])
            label = model_names[class_id]
            if label == "person":
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centroids.append((cx, cy))
        return centroids

    def _classify_movement(self, current_centroids, prev_centroids):
        """
        Compare current frame centroids to previous frame centroids.
        Returns a human-readable movement description.
        """
        if not prev_centroids or not current_centroids:
            return "unknown"

        moving = 0
        stationary = 0

        # Simple nearest-neighbour matching
        for curr in current_centroids:
            min_dist = float("inf")
            for prev in prev_centroids:
                dist = math.dist(curr, prev)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > self.movement_threshold:
                moving += 1
            else:
                stationary += 1

        if moving == 0:
            return "all stationary"
        elif stationary == 0:
            return "all moving"
        else:
            return f"{moving} moving, {stationary} stationary"

    def _classify_density(self, person_count):
        """Classify scene density based on number of people."""
        if person_count == 0:
            return "empty"
        elif person_count <= 2:
            return "low"
        elif person_count <= 5:
            return "medium"
        else:
            return "high"

    def build(self, result, model_names):
        """
        Main method. Takes a YOLOv8 result object and returns a structured scene dict.

        Args:
            result: ultralytics result object for a single frame
            model_names: dict of {class_id: class_name}

        Returns:
            dict: Structured scene data
        """
        boxes = result.boxes if result.boxes else []

        # Count all detected classes
        all_counts = {}
        for box in boxes:
            class_id = int(box.cls[0])
            label = model_names[class_id]
            all_counts[label] = all_counts.get(label, 0) + 1

        person_count = all_counts.pop("person", 0)
        # Everything else is an "object"
        other_objects = all_counts

        # Get current centroids for persons
        current_centroids = self._get_centroids(boxes, model_names)

        # Determine movement vs previous frame
        prev_centroids = self._centroid_history[-1] if self._centroid_history else []
        movement = self._classify_movement(current_centroids, prev_centroids)

        # Store current centroids in history
        self._centroid_history.append(current_centroids)
        if len(self._centroid_history) > self.history_size:
            self._centroid_history.pop(0)

        return {
            "people": person_count,
            "objects": other_objects,
            "density": self._classify_density(person_count),
            "movement": movement,
        }
