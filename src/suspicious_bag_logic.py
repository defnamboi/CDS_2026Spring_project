import math

class SuspiciousBagAnalyzer:
    """Track bag-to-person distance over time and flag unattended/abandoned bags."""
    def __init__(self, distance_threshold_px=150, abandonment_time_sec=5.0, fps=25.0, min_bag_track_frames=5):
        self.distance_threshold_px = float(distance_threshold_px)
        self.abandonment_time_sec = float(abandonment_time_sec)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.min_bag_track_frames = int(min_bag_track_frames)
        
        self.person_labels = {"person"}
        self.bag_labels = {"bag", "handbag", "backpack", "suitcase"}
        
        self.frame_index = 0
        self.bag_state = {}
        
        # Distance (px) to consider two bags as the same physical object
        self.PROXIMITY_MERGE_THRESHOLD = 50.0 

    def update(self, tracked_objects):
        self.frame_index += 1
        
        # 1. Separate objects by class immediately
        # This prevents the person's bounding box from suppressing the bag's box
        people = [obj for obj in tracked_objects if str(getattr(obj, "class_name", "")).lower() in self.person_labels]
        all_bags = [obj for obj in tracked_objects if str(getattr(obj, "class_name", "")).lower() in self.bag_labels]

        person_centers = {p.track_id: self._bbox_center(p.bbox_xyxy) for p in people}
        bag_status_by_id = {}
        events = []
        
        # 2. Filter duplicate bags (Strict Class-Aware Suppression)
        valid_bags = self._filter_duplicate_bags(all_bags)

        for bag in valid_bags:
            bag_id = bag.track_id
            bag_center = self._bbox_center(bag.bbox_xyxy)

            if bag_id not in self.bag_state:
                self._attempt_id_recovery(bag_id, bag_center)

            state = self.bag_state[bag_id]
            state["last_seen_frame"] = self.frame_index 
            state["active"] = True
            
            nearest_person_id, nearest_dist = self._get_nearest_person(bag_center, person_centers)

            state["last_center"] = bag_center
            state["seen_frames"] += 1

            status = self._calculate_status(state, nearest_dist)

            previous_status = state["status"]
            state["status"] = status
            state["distance_px"] = round(nearest_dist, 2) if nearest_dist != float("inf") else None
            
            bag_status_by_id[bag_id] = {
                "status": status,
                "nearest_person_id": nearest_person_id,
                "distance_px": state["distance_px"]
            }

            if status != previous_status:
                events.append({
                    "time": self._format_timestamp(self.frame_index / self.fps),
                    "bag_id": bag_id,
                    "person_id": nearest_person_id,
                    "status": status,
                    "previous_status": previous_status,
                    "distance": state["distance_px"]
                })

        self._cleanup_states(valid_bags)
        return bag_status_by_id, events

    def _filter_duplicate_bags(self, bags):
        """
        Prevents multiple bag boxes on the same object. 
        Prioritizes existing/senior tracks over new tracks.
        """
        # Sort by seen_frames (descending) then confidence
        # We want to keep the ID that has been around the longest
        sorted_bags = sorted(
            bags, 
            key=lambda x: (self.bag_state.get(x.track_id, {}).get("seen_frames", 0), getattr(x, "confidence", 0)), 
            reverse=True
        )
        
        unique_bags = []
        centers = []

        for bag in sorted_bags:
            current_center = self._bbox_center(bag.bbox_xyxy)
            # Only compare against other BAG centers
            if any(self._distance(current_center, c) < self.PROXIMITY_MERGE_THRESHOLD for c in centers):
                continue 
            unique_bags.append(bag)
            centers.append(current_center)
        return unique_bags

    def _calculate_status(self, state, nearest_dist):
        if state["seen_frames"] < self.min_bag_track_frames:
            state["unattended_start_frame"] = None
            return "warming_up"

        is_near_person = nearest_dist <= self.distance_threshold_px

        if is_near_person:
            state["unattended_start_frame"] = None
            state["abandon_confirm_buffer"] = 0 
            return "normal"
        else:
            if state["unattended_start_frame"] is None:
                state["unattended_start_frame"] = self.frame_index
            
            unattended_sec = (self.frame_index - state["unattended_start_frame"]) / self.fps

            # Grace period for flicker/occlusion
            if unattended_sec < 0.8:
                return "normal"
            
            if unattended_sec >= self.abandonment_time_sec:
                state["abandon_confirm_buffer"] += 1
                # Increase buffer to 15 frames for high-traffic stability
                return "abandoned" if state["abandon_confirm_buffer"] > 15 else "unattended"
            
            return "unattended"

    def _attempt_id_recovery(self, bag_id, bag_center):
        for old_id, old_state in self.bag_state.items():
            if old_state.get("active", False): continue
            
            frames_since_lost = self.frame_index - old_state.get("last_seen_frame", 0)
            if 0 < frames_since_lost < (self.fps * 4): # Look back 4 seconds
                if self._distance(bag_center, old_state["last_center"]) < 120:
                    self.bag_state[bag_id] = old_state.copy()
                    self.bag_state[bag_id]["active"] = True
                    del self.bag_state[old_id]
                    return
        
        self.bag_state[bag_id] = {
            "seen_frames": 0, "status": "normal", "unattended_start_frame": None,
            "distance_px": None, "last_center": bag_center, "abandon_confirm_buffer": 0, "active": True,
        }

    def _get_nearest_person(self, bag_center, person_centers):
        nearest_id, min_dist = None, float("inf")
        for p_id, p_center in person_centers.items():
            d = self._distance(bag_center, p_center)
            if d < min_dist:
                min_dist, nearest_id = d, p_id
        return nearest_id, min_dist

    def _cleanup_states(self, current_bags):
        active_ids = {b.track_id for b in current_bags}
        for bid, bstate in list(self.bag_state.items()):
            if bid not in active_ids:
                bstate["active"] = False
                # Keep stale IDs for 20 seconds to allow for re-entry/recovery
                if (self.frame_index - bstate.get("last_seen_frame", 0)) > (self.fps * 20):
                    del self.bag_state[bid]

    @staticmethod
    def _bbox_center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _format_timestamp(total_seconds):
        ts = int(max(total_seconds, 0))
        return f"{ts // 60:02d}:{ts % 60:02d}"