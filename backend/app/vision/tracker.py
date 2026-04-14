"""
AI Sentinel Lite - Centroid Multi-Object Tracker
-------------------------------------------------
Assigns persistent IDs to detected objects across frames using
Euclidean distance matching. Lightweight, no GPU needed.
"""

import math


class TrackedObject:
    def __init__(self, obj_id, centroid, bbox):
        self.id = obj_id
        self.centroid = centroid      # (cx, cy)
        self.bbox = bbox              # (x1, y1, x2, y2)
        self.frames_missing = 0
        self.trail = [centroid]       # History of positions for radar
        self.speed = 0.0              # Pixels per frame

    def update(self, centroid, bbox):
        if self.trail:
            self.speed = math.dist(centroid, self.trail[-1])
        self.centroid = centroid
        self.bbox = bbox
        self.frames_missing = 0
        self.trail.append(centroid)
        if len(self.trail) > 60:  # Keep ~2 seconds of trail at 30fps
            self.trail.pop(0)


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=120):
        """
        max_disappeared: frames before dropping a lost target
        max_distance: max pixel distance to match same target
        """
        self.next_id = 1
        self.objects = {}          # id -> TrackedObject
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        """
        detections: list of (cx, cy, x1, y1, x2, y2) tuples
        Returns: dict of {id: TrackedObject}
        """
        if not detections:
            # Mark all existing objects as missing
            lost = []
            for obj_id, obj in self.objects.items():
                obj.frames_missing += 1
                if obj.frames_missing > self.max_disappeared:
                    lost.append(obj_id)
            for obj_id in lost:
                del self.objects[obj_id]
            return self.objects

        input_centroids = [(d[0], d[1]) for d in detections]
        input_bboxes = [(d[2], d[3], d[4], d[5]) for d in detections]

        # If no existing objects, register all
        if not self.objects:
            for i, (cent, bbox) in enumerate(zip(input_centroids, input_bboxes)):
                self._register(cent, bbox)
            return self.objects

        # Match existing objects to new detections
        obj_ids = list(self.objects.keys())
        obj_centroids = [self.objects[oid].centroid for oid in obj_ids]

        # Build distance matrix
        distances = []
        for oc in obj_centroids:
            row = [math.dist(oc, ic) for ic in input_centroids]
            distances.append(row)

        # Greedy matching: pick closest pairs first
        used_objs = set()
        used_inputs = set()
        matches = []

        # Flatten and sort all distances
        all_pairs = []
        for i in range(len(obj_centroids)):
            for j in range(len(input_centroids)):
                all_pairs.append((distances[i][j], i, j))
        all_pairs.sort(key=lambda x: x[0])

        for dist, obj_idx, inp_idx in all_pairs:
            if obj_idx in used_objs or inp_idx in used_inputs:
                continue
            if dist > self.max_distance:
                break
            matches.append((obj_idx, inp_idx))
            used_objs.add(obj_idx)
            used_inputs.add(inp_idx)

        # Update matched objects
        for obj_idx, inp_idx in matches:
            oid = obj_ids[obj_idx]
            self.objects[oid].update(input_centroids[inp_idx], input_bboxes[inp_idx])

        # Mark unmatched existing objects as missing
        lost = []
        for i, oid in enumerate(obj_ids):
            if i not in used_objs:
                self.objects[oid].frames_missing += 1
                if self.objects[oid].frames_missing > self.max_disappeared:
                    lost.append(oid)
        for oid in lost:
            del self.objects[oid]

        # Register new unmatched detections
        for j in range(len(input_centroids)):
            if j not in used_inputs:
                self._register(input_centroids[j], input_bboxes[j])

        return self.objects

    def _register(self, centroid, bbox):
        self.objects[self.next_id] = TrackedObject(self.next_id, centroid, bbox)
        self.next_id += 1
