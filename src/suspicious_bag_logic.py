import math
import logging
from collections import deque

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[BAG %(levelname)s f=%(frame)s] %(message)s"))
    logger.addHandler(handler)


class _FrameAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})["frame"] = self.extra.get("frame", "?")
        return msg, kwargs


class SuspiciousBagAnalyzer:
    """
    Track bag-to-person distance over time and flag unattended/abandoned bags.

    Status machine
    --------------
    warming_up  → bag is too new to judge (< min_bag_track_frames)
    normal      → bag is with its owner or being carried
    unattended  → bag has been separated from owner for > grace_period_sec
    abandoned   → unattended for >= abandonment_time_sec AND confirmed over
                  a multi-frame buffer

    Occlusion persistence
    ---------------------
    Both 'unattended' AND 'abandoned' are sticky — once reached they cannot
    be reset by proximity or movement. Only physically disappearing from the
    scene for longer than the stale threshold resets the state (assumed
    picked up). This applies equally through occlusion: a person walking past
    that causes a brief track loss will not reset either alert level.

    Spatial position memory
    -----------------------
    Abandoned (and unattended) bag locations are stored in a grid-bucketed
    spatial index (_alert_positions). If DeepSORT loses and reassigns the
    bag's track ID — including recycling an old numeric ID — the new track
    immediately inherits the correct alert status the moment it reappears at
    the same location.
    """

    # DeepSORT recycles track IDs. If a bag_id reappears after this many frames
    # of absence we treat it as a new appearance and run spatial-memory lookup.
    RECYCLE_GAP_FRAMES = 5

    def __init__(
        self,
        distance_threshold_px=150,
        abandonment_time_sec=5.0,
        fps=25.0,
        min_bag_track_frames=5,
        grace_period_sec=0.5,        # reduced: faster normal → unattended
        unattended_time_sec=1.0,     # new: how long before first 'unattended' alert
        abandon_confirm_frames=5,
        history_len=16,
        movement_threshold_px=20.0,
        proximity_merge_threshold=50.0,
        alert_position_grid_size=60,
    ):
        self.distance_threshold_px      = float(distance_threshold_px)
        self.abandonment_time_sec       = float(abandonment_time_sec)
        self.unattended_time_sec        = float(unattended_time_sec)
        self.fps                        = float(fps) if fps and fps > 0 else 25.0
        self.min_bag_track_frames       = int(min_bag_track_frames)
        self.grace_period_sec           = float(grace_period_sec)
        self.abandon_confirm_frames     = int(abandon_confirm_frames)
        self.history_len                = int(history_len)
        self.movement_threshold_px      = float(movement_threshold_px)
        self.PROXIMITY_MERGE_THRESHOLD  = float(proximity_merge_threshold)
        self._GRID_SIZE                 = int(alert_position_grid_size)

        self.person_labels = {"person"}
        self.bag_labels    = {"bag", "handbag", "backpack", "suitcase"}

        self.frame_index = 0
        self.bag_state: dict = {}

        # Spatial memory of alert bag positions (unattended OR abandoned).
        # Keyed by (grid_x, grid_y) bucket — independent of track IDs.
        # Value includes the status so we know what level to restore.
        self._alert_positions: dict[tuple, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracked_objects):
        self.frame_index += 1
        log = _FrameAdapter(logger, {"frame": self.frame_index})

        people   = [o for o in tracked_objects if self._cls(o) in self.person_labels]
        all_bags = [o for o in tracked_objects if self._cls(o) in self.bag_labels]

        person_bboxes = {p.track_id: p.bbox_xyxy for p in people}

        log.debug(
            f"persons={[p.track_id for p in people]}  "
            f"raw_bag_ids={[b.track_id for b in all_bags]}  "
            f"alert_buckets={list(self._alert_positions.keys())}"
        )

        valid_bags = self._filter_duplicate_bags(all_bags)

        bag_status_by_id = {}
        events = []

        for bag in valid_bags:
            bag_id     = bag.track_id
            bag_center = self._bbox_center(bag.bbox_xyxy)

            existing = self.bag_state.get(bag_id)
            is_new   = existing is None
            recycled = (
                existing is not None
                and not existing.get("active", False)
                and (self.frame_index - existing.get("last_seen_frame", 0)) > self.RECYCLE_GAP_FRAMES
            )

            if is_new or recycled:
                if recycled:
                    log.debug(
                        f"RECYCLED bag_id={bag_id}  old_status={existing['status']}  "
                        f"gap={self.frame_index - existing.get('last_seen_frame', 0)}f  "
                        f"center={bag_center} -- running spatial lookup"
                    )
                    
                else:
                    log.debug(f"NEW bag_id={bag_id} center={bag_center}  checking spatial memory...")
                self._attempt_id_recovery(bag_id, bag_center, log)

            state = self.bag_state[bag_id]

            if is_new or recycled:
                log.debug(
                    f"bag_id={bag_id} initialised with status={state['status']}  "
                    f"seen_frames={state['seen_frames']}  "
                    f"unattended_start={state['unattended_start_frame']}  "
                    f"confirm_buffer={state['abandon_confirm_buffer']}"
                )

            state["last_seen_frame"] = self.frame_index
            state["active"]          = True
            state["seen_frames"]    += 1

            bag_center = self._ema_center(state, bag_center, alpha=0.6)
            state["last_center"] = bag_center
            state["center_history"].append(bag_center)

            nearest_person_id, nearest_dist = self._get_nearest_person(
                bag_center, person_bboxes
            )
            state["distance_px"] = (
                round(nearest_dist, 2) if nearest_dist != float("inf") else None
            )

            previous_status = state["status"]
            status = self._calculate_status(state, nearest_dist, nearest_person_id, bag_center, log)
            state["status"] = status

            if status != previous_status:
                log.debug(
                    f"STATUS CHANGE bag_id={bag_id}  "
                    f"{previous_status} -> {status}  "
                    f"nearest_person={nearest_person_id}  "
                    f"dist={state['distance_px']}px  "
                    f"seen={state['seen_frames']}f  "
                    f"unattended_start={state['unattended_start_frame']}"
                )
                events.append({
                    "time":            self._format_timestamp(self.frame_index / self.fps),
                    "bag_id":          bag_id,
                    "person_id":       nearest_person_id,
                    "status":          status,
                    "previous_status": previous_status,
                    "distance":        state["distance_px"],
                })

            bag_status_by_id[bag_id] = {
                "status":            status,
                "nearest_person_id": nearest_person_id,
                "distance_px":       state["distance_px"],
            }

        self._cleanup_states(valid_bags, log)
        return bag_status_by_id, events

    # ------------------------------------------------------------------
    # Status logic
    # ------------------------------------------------------------------

    def _calculate_status(self, state, nearest_dist, nearest_person_id, bag_center, log):

        
        if state["seen_frames"] < self.min_bag_track_frames:
            state["unattended_start_frame"] = None
            return "warming_up"

        current_status  = state["status"]
        already_alerted = current_status in ("unattended", "abandoned")

        # Abandoned is terminal — never downgrade it under any circumstances
        if current_status == "abandoned":
            self._register_alert_position(state["last_center"], "abandoned", log)
            return "abandoned"

        if not already_alerted:
            is_near_person   = nearest_dist <= self.distance_threshold_px
            is_being_carried = self._is_bag_moving(state)

            if is_near_person or is_being_carried:
                if state["owner_id"] is None:
                    state["owner_id"] = nearest_person_id
                state["unattended_start_frame"] = None
                state["abandon_confirm_buffer"] = 0
                return "normal"

        if state["unattended_start_frame"] is None:
            state["unattended_start_frame"] = self.frame_index

        unattended_sec = (self.frame_index - state["unattended_start_frame"]) / self.fps

        if not already_alerted and unattended_sec < self.grace_period_sec:
            return "normal"

        if unattended_sec < self.unattended_time_sec:
            return "unattended"

        if unattended_sec >= self.abandonment_time_sec:
            state["abandon_confirm_buffer"] += 1
            if state["abandon_confirm_buffer"] >= self.abandon_confirm_frames:
                self._register_alert_position(bag_center, "abandoned", log)
                return "abandoned"
            return "unattended"

        self._register_alert_position(bag_center, "unattended", log)
        return "unattended"

    # ------------------------------------------------------------------
    # Spatial position memory
    # ------------------------------------------------------------------

    def _get_position_bucket(self, center):
        return (int(center[0] // self._GRID_SIZE), int(center[1] // self._GRID_SIZE))

    def _check_alert_position(self, center):
        bx, by = self._get_position_bucket(center)

        found_unattended = False

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                entry = self._alert_positions.get((bx + dx, by + dy))
                if not entry:
                    continue

                if entry["status"] == "abandoned":
                    return "abandoned"   # 🔥 ALWAYS PRIORITIZE

                if entry["status"] == "unattended":
                    found_unattended = True

        return "unattended" if found_unattended else None

    def _register_alert_position(self, center, status, log=None):
        bucket = self._get_position_bucket(center)
        existing = self._alert_positions.get(bucket)
        # Only upgrade status (unattended → abandoned), never downgrade
        if existing and existing["status"] == "abandoned" and status == "unattended":
            return
        self._alert_positions[bucket] = {
            "center": center,
            "status": status,
            "frame":  self.frame_index,
        }
        if log:
            log.debug(f"REGISTERED alert position bucket={bucket} status={status} center={center}")

    def _clear_alert_position(self, center, log=None):
        bx, by = self._get_position_bucket(center)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (bx + dx, by + dy)
                if key in self._alert_positions:
                    if log:
                        log.debug(f"CLEARED alert position bucket={key}")
                    del self._alert_positions[key]

    # ------------------------------------------------------------------
    # Movement / carried-bag detection
    # ------------------------------------------------------------------

    def _is_bag_moving(self, state):
        history = state["center_history"]
        if len(history) < 9:
            return False

        third = len(history) // 3
        seg_a = history[0]
        seg_b = history[third]
        seg_c = history[third * 2]
        seg_d = history[-1]

        d1 = math.hypot(seg_b[0] - seg_a[0], seg_b[1] - seg_a[1])
        d2 = math.hypot(seg_c[0] - seg_b[0], seg_c[1] - seg_b[1])
        d3 = math.hypot(seg_d[0] - seg_c[0], seg_d[1] - seg_c[1])

        half_thresh = self.movement_threshold_px * 0.5
        return d1 > half_thresh and d2 > half_thresh and d3 > half_thresh

    # ------------------------------------------------------------------
    # Center smoothing (EMA)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_center(state, new_center, alpha=0.6):
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

    def _attempt_id_recovery(self, bag_id, bag_center, log=None):
        recovery_window_frames = self.fps * 10
        recovery_distance_px   = 150

        best_old_id     = None
        best_frames_ago = float("inf")

        if log:
            inactive = {
                k: v for k, v in self.bag_state.items()
                if not v.get("active", False)
            }
            log.debug(
                f"  Recovery candidates for bag_id={bag_id}: "
                + str({
                    k: {
                        "status":      v["status"],
                        "last_center": v["last_center"],
                        "frames_ago":  self.frame_index - v.get("last_seen_frame", 0),
                    }
                    for k, v in inactive.items()
                })
            )

        for old_id, old_state in self.bag_state.items():
            if old_id == bag_id:
                continue
            if old_state.get("active", False):
                continue
            frames_ago = self.frame_index - old_state.get("last_seen_frame", 0)
            if frames_ago <= 0 or frames_ago >= recovery_window_frames:
                continue
            dist = self._distance(bag_center, old_state["last_center"])
            if log:
                log.debug(
                    f"  Candidate old_id={old_id}: frames_ago={frames_ago} "
                    f"dist={dist:.1f}px (threshold={recovery_distance_px}px) "
                    f"status={old_state['status']}"
                )
            if dist >= recovery_distance_px:
                continue
            if frames_ago < best_frames_ago:
                best_frames_ago = frames_ago
                best_old_id     = old_id
                
        if bag_id in self.bag_state:
            old_state = self.bag_state[bag_id]
            dist = self._distance(bag_center, old_state["last_center"])
            if dist < 150: # Standard recovery threshold
                if log:
                    log.debug(f"  SELF-RECOVERY: bag_id={bag_id} was recycled and is still at the same spot. Keeping status={old_state['status']}")
                # Just mark it active and update center, don't delete/re-init
                old_state["active"] = True
                old_state["ema_center"] = bag_center
                return

        # ---- Case 1: matched a lost track ----
        if best_old_id is not None:
            recovered = self.bag_state.pop(best_old_id)
            recovered["active"]     = True
            recovered["ema_center"] = bag_center

            # Never shift the timer for alerted bags — both unattended and
            # abandoned must keep their original start frame through occlusion.
            was_alerted = recovered.get("status") in ("unattended", "abandoned")
            if not was_alerted and recovered.get("unattended_start_frame") is not None:
                recovered["unattended_start_frame"] += best_frames_ago

            if log:
                log.debug(
                    f"  RECOVERED: old_id={best_old_id} -> new_id={bag_id}  "
                    f"status={recovered['status']}  was_alerted={was_alerted}"
                )
            self.bag_state[bag_id] = recovered
            return

        # ---- Case 2: spatial memory hit ----
        restored_status = self._check_alert_position(bag_center)
        if log:
            log.debug(
                f"  No track match. spatial_hit={restored_status}  "
                f"center={bag_center}  "
                f"bucket={self._get_position_bucket(bag_center)}  "
                f"all_alert_buckets={list(self._alert_positions.keys())}"
            )

        if restored_status:
            if log:
                log.debug(
                    f"  SPATIAL RESTORE: bag_id={bag_id} at alert position -> "
                    f"restoring status={restored_status} immediately"
                )
            # Set the unattended start far enough back to satisfy whichever
            # threshold applies to the restored status.
            if restored_status == "abandoned":
                unattended_start = self.frame_index - int(self.abandonment_time_sec * self.fps) - 1
                confirm_buffer   = self.abandon_confirm_frames
            else:  # unattended
                unattended_start = self.frame_index - int(self.unattended_time_sec * self.fps) - 1
                confirm_buffer   = 0

            self.bag_state[bag_id] = {
                "seen_frames":            self.min_bag_track_frames,
                "status":                 restored_status,
                "unattended_start_frame": unattended_start,
                "abandon_confirm_buffer": confirm_buffer,
                "distance_px":            None,
                "last_center":            bag_center,
                "ema_center":             bag_center,
                "center_history":         deque(maxlen=self.history_len),
                "active":                 True,
                "owner_id":               None,
                "owner_lost_frame":       None,
            }
            return

        # ---- Case 3: brand new bag ----
        if log:
            log.debug(f"  NEW bag_id={bag_id}: no match, no spatial hit -> starting fresh")
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

    def _cleanup_states(self, current_bags, log=None):
        active_ids      = {b.track_id for b in current_bags}
        stale_threshold = self.fps * 20

        for bid in list(self.bag_state.keys()):
            bstate = self.bag_state[bid]
            if bid not in active_ids:
                bstate["active"] = False
                frames_gone = self.frame_index - bstate.get("last_seen_frame", 0)
                if frames_gone > stale_threshold:
                    if bstate.get("status") in ("unattended", "abandoned"):
                        if log:
                            log.debug(
                                f"STALE alerted bag_id={bid} status={bstate['status']} "
                                f"gone {frames_gone}f -- clearing spatial memory"
                            )
                        self._clear_alert_position(bstate["last_center"], log)
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