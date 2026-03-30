try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:  # pragma: no cover - runtime fallback when package is missing
    DeepSort = None


class TrackedObject:
    def __init__(self, track_id, class_name, confidence, bbox_xyxy):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_xyxy = bbox_xyxy


class DeepSortTracker:
    """Thin wrapper around deep_sort_realtime with a stable app-facing API."""

    def __init__(self, max_age=20, n_init=8, max_iou_distance=0.70):
        self.enabled = DeepSort is not None
        self._tracker = None

        if self.enabled:
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
            )

    def update(self, detections, frame=None):
        """
        Update tracker state using detections from the current frame.

        Expected detection format:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.92,
                "class_name": "person"
            },
            ...
        ]
        """
        if not self.enabled or self._tracker is None:
            # If the dependency is unavailable, fail gracefully instead of breaking the app.
            return []

        deep_sort_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            width = max(0.0, float(x2) - float(x1))
            height = max(0.0, float(y2) - float(y1))
            if width == 0.0 or height == 0.0:
                continue

            # DeepSORT expects [left, top, width, height] + confidence + class label.
            deep_sort_inputs.append(
                (
                    [float(x1), float(y1), width, height],
                    float(det["confidence"]),
                    str(det["class_name"]),
                )
            )

        # Main tracking step:
        # 1) Predict existing track positions with Kalman filtering.
        # 2) Associate new detections to predicted tracks (motion + appearance cues).
        # 3) Create new tracks for unmatched detections and age unmatched tracks.
        tracks = self._tracker.update_tracks(deep_sort_inputs, frame=frame)

        tracked_objects = []
        for track in tracks:
            # Use only confirmed tracks so IDs are stable (reduces one-frame false positives).
            if not track.is_confirmed():
                continue

            left, top, right, bottom = track.to_ltrb()
            tracked_objects.append(
                TrackedObject(
                    track_id=int(track.track_id),
                    class_name=str(track.get_det_class() or "object"),
                    confidence=float(track.get_det_conf() or 0.0),
                    bbox_xyxy=(int(left), int(top), int(right), int(bottom)),
                )
            )

        return tracked_objects
