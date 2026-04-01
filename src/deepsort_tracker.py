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
    """Thin wrapper around deep_sort_realtime with built-in duplicate suppression."""

    def __init__(self, max_age=60, n_init=5, max_iou_distance=0.7,
                 embedder_gpu=True, max_cosine_distance=0.2):
        self.enabled = DeepSort is not None
        self._tracker = None
        # Internal threshold: If two boxes of the same class overlap more than this, kill one.
        self.INTERNAL_IOU_THRESHOLD = 0.5 

        if self.enabled:
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                embedder_gpu=embedder_gpu,
                max_cosine_distance=max_cosine_distance
            )
            
    def update(self, detections, frame=None):
            if not self.enabled or self._tracker is None:
                return []

            deep_sort_inputs = []
            for det in detections:
                if det["confidence"] <= 0.20: 
                    continue
                
                x1, y1, x2, y2 = det["bbox"]
                w, h = float(x2) - float(x1), float(y2) - float(y1)
                
                if w <= 5 or h <= 5: 
                    continue

                # Keep class names clean for the tracker
                c_name = str(det.get("class_name", "object")).lower()
                deep_sort_inputs.append(([float(x1), float(y1), w, h], float(det["confidence"]), c_name))

            # 1. Update DeepSORT
            tracks = self._tracker.update_tracks(deep_sort_inputs, frame=frame)
            
            # 2. GHOST FILTER: Only keep tracks YOLO actually confirmed this frame
            active_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]

            # 3. SENIORITY SORT: Older, more stable IDs get priority
            active_tracks.sort(key=lambda x: x.hits, reverse=True)

            tracked_objects = []
            # We track claimed spaces PER CLASS to allow person/bag overlap
            claimed_spots = {"person": [], "bag": [], "handbag": [], "backpack": [], "suitcase": []}
            
            # Distance to merge same-class duplicates (adjust based on resolution)
            MERGE_DISTANCE_PX = 50 

            for track in active_tracks:
                l, t, r, b = track.to_ltrb()
                center = ((l + r) / 2.0, (t + b) / 2.0)
                cls = str(track.get_det_class()).lower()

                # Ensure the class exists in our map (default to 'bag' logic)
                if cls not in claimed_spots:
                    cls = "bag"

                # Check if this specific class already has a senior ID in this spot
                is_duplicate = False
                for senior_center in claimed_spots[cls]:
                    dist = math.hypot(center[0] - senior_center[0], center[1] - senior_center[1])
                    if dist < MERGE_DISTANCE_PX:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue # Kills the flickering second box for the SAME class

                # If it's the first stable ID for this spot/class, keep it
                claimed_spots[cls].append(center)
                
                det_conf = track.get_det_conf()

                tracked_objects.append(
                    TrackedObject(
                        track_id=int(track.track_id),
                        class_name=cls,
                        confidence=float(det_conf if det_conf is not None else 0.0),
                        bbox_xyxy=(int(l), int(t), int(r), int(b))
                    )
                )
                
            return tracked_objects

    def _apply_internal_nms(self, tracks):
        """Standard NMS logic to ensure one ID per physical object."""
        if not tracks:
            return []
            
        indices = list(range(len(tracks)))
        keep = []
        
        while len(indices) > 0:
            current_idx = indices.pop(0)
            keep.append(current_idx)
            
            current_box = tracks[current_idx].to_ltrb()
            current_class = tracks[current_idx].get_det_class()
            
            remaining_indices = []
            for i in indices:
                # Only suppress if they are the SAME class (don't kill bag because of person)
                if tracks[i].get_det_class() == current_class:
                    iou = self._calculate_iou(current_box, tracks[i].to_ltrb())
                    if iou < self.INTERNAL_IOU_THRESHOLD:
                        remaining_indices.append(i)
                else:
                    remaining_indices.append(i)
            indices = remaining_indices
            
        return keep

    @staticmethod
    def _calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)