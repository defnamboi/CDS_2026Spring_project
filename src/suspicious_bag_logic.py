import math
from collections import deque


class SuspiciousBagAnalyzer:
    """
    Track bag-to-person distance over time and flag unattended/abandoned bags.

    Status machine
    --------------
    warming_up  → bag is too new to judge (< min_bag_track_frames)
    normal      → bag is with its owner or being carried
    unattended  → bag has been separated from owner for > grace_period but
                  < abandonment_time_sec
    abandoned   → separation has persisted >= abandonment_time_sec AND confirmed
                  over a multi-frame buffer (avoids alerting on brief occlusion)

    A bag can only return to 'normal' if it physically disappears from the scene
    (picked up), which clears its state entirely. Walking past an alerted bag
    does NOT reset it.
    """

    def __init__(
        self,
        distance_threshold_px=150,
        abandonment_time_sec=5.0,
        fps=25.0,
        min_bag_track_frames=5,
        grace_period_sec=1.5,
        abandon_confirm_frames=5,
        history_len=16,
        movement_threshold_px=20.0,
        proximity_merge_threshold=50.0,
    ):
        self.distance_threshold_px  = float(distance_threshold_px)
        self.abandonment_time_sec   = float(abandonment_time_sec)
        self.fps                    = float(fps) if fps and fps > 0 else 25.0
        self.min_bag_track_frames   = int(min_bag_track_frames)
        self.grace_period_sec       = float(grace_period_sec)
        self.abandon_confirm_frames = int(abandon_confirm_frames)
        self.history_len            = int(history_len)
        self.movement_threshold_px  = float(movement_threshold_px)
        self.PROXIMITY_MERGE_THRESHOLD = float(proximity_merge_threshold)

        self.person_labels = {"person"}
        self.bag_labels    = {"bag", "handbag", "backpack", "suitcase"}

        self.frame_index = 0
        self.bag_state: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracked_objects):
        self.frame_index += 1

        people   = [o for o in tracked_objects if self._cls(o) in self.person_labels]
        all_bags = [o for o in tracked_objects if self._cls(o) in self.bag_labels]

        person_bboxes = {p.track_id: p.bbox_xyxy for p in people}

        valid_bags = self._filter_duplicate_bags(all_bags)

        bag_status_by_id = {}
        events = []

        for bag in valid_bags:
            bag_id     = bag.track_id
            bag_center = self._bbox_center(bag.bbox_xyxy)

            # Initialise or recover state
            if bag_id not in self.bag_state:
                self._attempt_id_recovery(bag_id, bag_center)

            state = self.bag_state[bag_id]
            state["last_seen_frame"] = self.frame_index
            state["active"]          = True
            state["seen_frames"]    += 1

            # Smooth the bag center with an EMA to reduce per-frame bbox jitter
            bag_center = self._ema_center(state, bag_center, alpha=0.6)
            state["last_center"] = bag_center

            # Rolling position history (O(1) append/pop via deque)
            state["center_history"].append(bag_center)

            nearest_person_id, nearest_dist = self._get_nearest_person(
                bag_center, person_bboxes
            )
            state["distance_px"] = (
                round(nearest_dist, 2) if nearest_dist != float("inf") else None
            )

            previous_status = state["status"]
            status = self._calculate_status(state, nearest_dist, nearest_person_id)
            state["status"] = status

            bag_status_by_id[bag_id] = {
                "status":            status,
                "nearest_person_id": nearest_person_id,
                "distance_px":       state["distance_px"],
            }

            if status != previous_status:
                events.append({
                    "time":            self._format_timestamp(self.frame_index / self.fps),
                    "bag_id":          bag_id,
                    "person_id":       nearest_person_id,
                    "status":          status,
                    "previous_status": previous_status,
                    "distance":        state["distance_px"],
                })

        self._cleanup_states(valid_bags)
        return bag_status_by_id, events

    # ------------------------------------------------------------------
    # Status logic
    # ------------------------------------------------------------------

    def _calculate_status(self, state, nearest_dist, nearest_person_id):
        # Not enough frames yet — hold off judging
        if state["seen_frames"] < self.min_bag_track_frames:
            state["unattended_start_frame"] = None
            return "warming_up"

        current_status  = state["status"]
        already_alerted = current_status in ("unattended", "abandoned")

        # Once alerted, NOTHING resets the bag — not proximity, not movement.
        # Only physically disappearing from scene (picked up) clears the state.
        if not already_alerted:
            is_near_person   = nearest_dist <= self.distance_threshold_px
            is_being_carried = self._is_bag_moving(state)
            is_attended      = is_near_person or is_being_carried

            if is_attended:
                if state["owner_id"] is None:
                    state["owner_id"] = nearest_person_id
                state["unattended_start_frame"] = None
                state["abandon_confirm_buffer"] = 0
                return "normal"

        # ---- UNATTENDED / ABANDONED path ----
        # Once we reach here, the bag is unattended.
        # Guard: never allow unattended_start_frame to be None once alerted —
        # ID recovery can shift it to None via the occlusion gap correction,
        # which would restart the clock from zero and drop back to "normal".
        if state["unattended_start_frame"] is None:
            state["unattended_start_frame"] = self.frame_index

        unattended_sec = (self.frame_index - state["unattended_start_frame"]) / self.fps

        # Grace period only applies before the first alert — skip it if already alerted
        if not already_alerted and unattended_sec < self.grace_period_sec:
            return "normal"

        if unattended_sec >= self.abandonment_time_sec:
            state["abandon_confirm_buffer"] += 1
            if state["abandon_confirm_buffer"] >= self.abandon_confirm_frames:
                return "abandoned"
            return "unattended"

        return "unattended"

    # ------------------------------------------------------------------
    # Movement / carried-bag detection
    # ------------------------------------------------------------------

    def _is_bag_moving(self, state):
        """
        True only if the bag has sustained directional movement — not just a
        single-frame bbox jitter spike.

        Strategy: split the history into thirds and check that EACH consecutive
        segment has moved meaningfully from the previous one. This filters out:
          - bbox jitter from a person walking past (one-frame displacement)
          - camera shake (affects all detections equally, not directional)
          - DeepSORT bbox interpolation artifacts
        A genuinely carried bag will show consistent displacement across all
        segments, not just a spike at one point.
        """
        history = state["center_history"]
        if len(history) < 9:   # need enough points to split into thirds
            return False

        third = len(history) // 3
        seg_a = history[0]           # start of window
        seg_b = history[third]       # 1/3 through
        seg_c = history[third * 2]   # 2/3 through
        seg_d = history[-1]          # end of window

        d1 = math.hypot(seg_b[0] - seg_a[0], seg_b[1] - seg_a[1])
        d2 = math.hypot(seg_c[0] - seg_b[0], seg_c[1] - seg_b[1])
        d3 = math.hypot(seg_d[0] - seg_c[0], seg_d[1] - seg_c[1])

        # ALL three segments must show movement above threshold.
        # A jitter spike only appears in one segment; real carrying shows all three.
        half_thresh = self.movement_threshold_px * 0.5
        return d1 > half_thresh and d2 > half_thresh and d3 > half_thresh

    # ------------------------------------------------------------------
    # Center smoothing (EMA)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_center(state, new_center, alpha=0.6):
        """
        Exponential moving average on the bag center.
        alpha close to 1.0 → mostly raw detection (responsive).
        alpha close to 0.0 → heavy smoothing (lag).
        0.6 balances jitter removal with responsiveness.
        """
        prev = state.get("ema_center")
        if prev is None:
            state["ema_center"] = new_center
            return new_center
        sx = alpha * new_center[0] + (1 - alpha) * prev[0]
        sy = alpha * new_center[1] + (1 - alpha) * prev[1]
        state["ema_center"] = (sx, sy)
        return (sx, sy)

    # ------------------------------------------------------------------
    # Duplicate suppression
    # ------------------------------------------------------------------

    def _filter_duplicate_bags(self, bags):
        """
        Keep at most one bag track per physical location.
        Senior tracks (more seen_frames, then higher confidence) always win.
        """
        sorted_bags = sorted(
            bags,
            key=lambda x: (
                self.bag_state.get(x.track_id, {}).get("seen_frames", 0),
                getattr(x, "confidence", 0.0),
            ),
            reverse=True,
        )

        unique_bags = []
        centers = []
        for bag in sorted_bags:
            c = self._bbox_center(bag.bbox_xyxy)
            if any(self._distance(c, ec) < self.PROXIMITY_MERGE_THRESHOLD for ec in centers):
                continue
            unique_bags.append(bag)
            centers.append(c)
        return unique_bags

    # ------------------------------------------------------------------
    # ID recovery on re-appearance
    # ------------------------------------------------------------------

    def _attempt_id_recovery(self, bag_id, bag_center):
        """
        When a new track ID appears, check if it matches a recently lost track
        at roughly the same position. If so, inherit its full state (including
        alert status and unattended timer) so occlusion + ID reassignment does
        not reset the abandonment counter.

        Recovery window is 10 s (generous) because occlusion behind a slow-
        moving crowd can last that long on a static bag.
        The unattended_start_frame is shifted forward by the occlusion gap so
        dead time is not counted as unattended time.
        """
        recovery_window_frames = self.fps * 10  # 10-second look-back (was 4)
        recovery_distance_px   = 150            # slightly wider than before (was 120)

        best_old_id     = None
        best_frames_ago = float("inf")

        for old_id, old_state in self.bag_state.items():
            if old_id == bag_id:
                continue
            if old_state.get("active", False):
                continue
            frames_ago = self.frame_index - old_state.get("last_seen_frame", 0)
            if frames_ago <= 0 or frames_ago >= recovery_window_frames:
                continue
            if self._distance(bag_center, old_state["last_center"]) >= recovery_distance_px:
                continue
            if frames_ago < best_frames_ago:
                best_frames_ago = frames_ago
                best_old_id     = old_id

        if best_old_id is not None:
            recovered = self.bag_state.pop(best_old_id)
            recovered["active"]     = True
            recovered["ema_center"] = bag_center  # reset smoother to avoid jump

            # Shift unattended_start_frame forward by the occlusion gap so we
            # do not count invisible frames as unattended time.
            if recovered.get("unattended_start_frame") is not None:
                recovered["unattended_start_frame"] += best_frames_ago

            self.bag_state[bag_id] = recovered
            return

        # Brand-new bag — start from scratch
        self.bag_state[bag_id] = {
            "seen_frames":            0,
            "status":                 "normal",
            "unattended_start_frame": None,
            "distance_px":            None,
            "last_center":            bag_center,
            "ema_center":             bag_center,
            "center_history":         deque(maxlen=self.history_len),
            "abandon_confirm_buffer": 0,
            "active":                 True,
            "owner_id":               None,
            "owner_lost_frame":       None,
        }

    # ------------------------------------------------------------------
    # Nearest person (bbox-edge distance)
    # ------------------------------------------------------------------

    def _get_nearest_person(self, bag_center, person_bboxes):
        """
        Distance from bag center to the nearest EDGE of each person bbox.
        A carried bag overlapping its carrier returns 0 px — correctly attended.
        Center-to-center would falsely flag backpacks / suitcases being dragged.
        """
        nearest_id, min_dist = None, float("inf")
        for p_id, bbox in person_bboxes.items():
            cx = max(bbox[0], min(bag_center[0], bbox[2]))
            cy = max(bbox[1], min(bag_center[1], bbox[3]))
            d  = self._distance(bag_center, (cx, cy))
            if d < min_dist:
                min_dist, nearest_id = d, p_id
        return nearest_id, min_dist

    # ------------------------------------------------------------------
    # State cleanup
    # ------------------------------------------------------------------

    def _cleanup_states(self, current_bags):
        active_ids       = {b.track_id for b in current_bags}
        stale_threshold  = self.fps * 20  # drop state after 20 s off-screen

        for bid in list(self.bag_state.keys()):
            bstate = self.bag_state[bid]
            if bid not in active_ids:
                bstate["active"] = False
                frames_gone = self.frame_index - bstate.get("last_seen_frame", 0)
                if frames_gone > stale_threshold:
                    del self.bag_state[bid]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cls(obj):
        return str(getattr(obj, "class_name", "")).lower()

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