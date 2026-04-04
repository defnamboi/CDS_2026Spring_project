import math


class SuspiciousBagAnalyzer:
    """Track bag-to-person distance over time and flag unattended/abandoned bags."""

    def __init__(
        self,
        distance_threshold_px,
        abandonment_time_sec,
        min_bag_track_frames,
        fps=20.0
    ):
        self.distance_threshold_px = float(distance_threshold_px)
        self.abandonment_time_sec = float(abandonment_time_sec)
        #video is recorded in 30fps, assume compressed to 20fps after sending over telegram
        self.fps = float(fps) if fps > 0.0 else 20.0
        self.min_bag_track_frames = int(min_bag_track_frames)

        self.person_labels = {"person"}
        self.bag_labels = {"bag", "handbag", "backpack", "suitcase"}

        self.frame_index = 0
        self.bag_state = {}

    def reset(self):
        self.frame_index = 0
        self.bag_state = {}

    def update(self, tracked_objects):
        self.frame_index += 1

        people = []
        bags = []
        for obj in tracked_objects:
            class_name = str(getattr(obj, "class_name", "")).lower()
            track_id = getattr(obj, "track_id", None)
            bbox = getattr(obj, "bbox_xyxy", None)
            if track_id is None or bbox is None:
                continue

            if class_name in self.person_labels:
                people.append(obj)
            elif class_name in self.bag_labels:
                bags.append(obj)

        person_centers = {}
        for person in people:
            person_centers[person.track_id] = self._bbox_center(person.bbox_xyxy)

        bag_status_by_id = {}
        events = []

        for bag in bags:
            bag_id = bag.track_id
            bag_center = self._bbox_center(bag.bbox_xyxy)

            state = self.bag_state.setdefault(
                bag_id,
                {
                    "seen_frames": 0,
                    "status": "normal",
                    "unattended_start_frame": None,
                    "nearest_person_id": None,
                    "distance_px": None,
                },
            )
            state["seen_frames"] += 1

            nearest_person_id = None
            nearest_distance = float("inf")
            for person_id, center in person_centers.items():
                dist = self._distance(bag_center, center)
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_person_id = person_id

            if nearest_person_id is None:
                nearest_distance = float("inf")

            if state["seen_frames"] < self.min_bag_track_frames:
                status = "warming_up"
                state["unattended_start_frame"] = None
            else:
                if nearest_distance <= self.distance_threshold_px:
                    status = "normal"
                    state["unattended_start_frame"] = None
                else:
                    if state["unattended_start_frame"] is None:
                        state["unattended_start_frame"] = self.frame_index

                    unattended_frames = self.frame_index - state["unattended_start_frame"]
                    unattended_seconds = unattended_frames / self.fps

                    if unattended_seconds >= self.abandonment_time_sec:
                        status = "abandoned"
                    else:
                        status = "unattended"

            previous_status = state["status"]
            state["status"] = status
            state["nearest_person_id"] = nearest_person_id
            state["distance_px"] = None if nearest_distance == float("inf") else round(nearest_distance, 2)

            bag_status_by_id[bag_id] = {
                "status": status,
                "nearest_person_id": nearest_person_id,
                "distance_px": state["distance_px"],
                "seen_frames": state["seen_frames"],
            }

            if status != previous_status:
                events.append(
                    {
                        "time": self._format_timestamp(self.frame_index / self.fps),
                        "bag_id": bag_id,
                        "person_id": nearest_person_id,
                        "distance": state["distance_px"],
                        "status": status,
                        "previous_status": previous_status,
                    }
                )

        return bag_status_by_id, events

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _format_timestamp(total_seconds):
        total_seconds = int(max(total_seconds, 0))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
