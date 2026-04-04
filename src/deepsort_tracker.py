import math
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None


class TrackedObject:
    def __init__(self, track_id, class_name, confidence, bbox_xyxy):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_xyxy = bbox_xyxy


class DeepSortTracker:
    """
    Thin wrapper around deep_sort_realtime tuned for fine-tuned YOLO models.

    Key design decisions
    --------------------
    * n_init=3          – confirm tracks faster; fine-tuned YOLO is already precise
                          so we don't need 5 frames of evidence before trusting it.
    * max_age=90        – keep a predicted track alive for up to 90 frames (~3.6 s at
                          25 fps) before discarding. This absorbs occlusion and missed
                          detections without spawning a new ID on re-appearance.
    * max_cosine_distance=0.35
                        – relaxed from 0.2. Fine-tuned models can produce slightly
                          varying embeddings across frames; being too tight causes
                          good tracks to be rejected and re-assigned new IDs.
    * max_iou_distance=0.7
                        – unchanged; generous enough to handle partial occlusion.
    * Ghost filter: allow time_since_update <= 1 instead of == 0. This lets DeepSORT's
                    Kalman predictor bridge a single missed detection frame without
                    killing the track (the primary cause of ID flicker).
    """

    # Same-class duplicate suppression: bags closer than this (px) are the same object
    MERGE_DISTANCE_PX = 50

    def __init__(
        self,
        max_age=100,
        n_init=10,
        max_iou_distance=0.7,
        embedder_gpu=True,
        max_cosine_distance=0.45,
        min_confidence=0.25,
        min_dimension_px=8,
    ):
        self.enabled = DeepSort is not None
        self.min_confidence = float(min_confidence)
        self.min_dimension_px = int(min_dimension_px)
        self._tracker = None

        if self.enabled:
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                embedder_gpu=embedder_gpu,
                max_cosine_distance=max_cosine_distance,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections, frame=None):
        if not self.enabled or self._tracker is None:
            return []

        deep_sort_inputs = self._prepare_inputs(detections)
        raw_tracks = self._tracker.update_tracks(deep_sort_inputs, frame=frame)

        # Allow the Kalman predictor to bridge up to 1 missed frame.
        # time_since_update == 0  → YOLO matched this frame  (always keep)
        # time_since_update == 1  → Kalman prediction only   (keep to kill flicker)
        # time_since_update >= 2  → too stale, skip
        live_tracks = [
            t for t in raw_tracks
            if t.is_confirmed() and t.time_since_update <= 1
        ]

        # Older (more hits) tracks get priority in duplicate resolution
        live_tracks.sort(key=lambda t: t.hits, reverse=True)

        return self._resolve_duplicates(live_tracks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, detections):
        """Convert raw detections to the (ltwh, conf, class) format DeepSORT expects."""
        inputs = []
        for det in detections:
            if det["confidence"] <= self.min_confidence:
                continue
            x1, y1, x2, y2 = det["bbox"]
            w = float(x2) - float(x1)
            h = float(y2) - float(y1)
            if w <= self.min_dimension_px or h <= self.min_dimension_px:
                continue
            class_name = str(det.get("class_name", "object")).lower()
            inputs.append(([float(x1), float(y1), w, h], float(det["confidence"]), class_name))
        return inputs

    def _resolve_duplicates(self, tracks):
        """
        Per-class duplicate suppression using center distance.
        Persons and bags are tracked in separate buckets so a person bbox
        never suppresses a nearby bag box.
        """
        # claimed_spots maps class_name -> list of (cx, cy) already kept
        claimed_spots: dict[str, list[tuple]] = {}
        tracked_objects = []

        for track in tracks:
            l, t, r, b = track.to_ltrb()
            cls = str(track.get_det_class() or "object").lower()
            center = ((l + r) / 2.0, (t + b) / 2.0)

            bucket = claimed_spots.setdefault(cls, [])

            # Check if a senior track of the SAME class already owns this spot
            if any(
                math.hypot(center[0] - sc[0], center[1] - sc[1]) < self.MERGE_DISTANCE_PX
                for sc in bucket
            ):
                continue  # duplicate — discard the junior track

            bucket.append(center)

            det_conf = track.get_det_conf()
            tracked_objects.append(
                TrackedObject(
                    track_id=int(track.track_id),
                    class_name=cls,
                    confidence=float(det_conf if det_conf is not None else 0.0),
                    bbox_xyxy=(int(l), int(t), int(r), int(b)),
                )
            )

        return tracked_objects

    # ------------------------------------------------------------------
    # IoU utility (kept for reference / future NMS use)
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)