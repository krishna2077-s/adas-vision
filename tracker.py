"""
tracker.py — Module 4: lightweight multi-object tracker.

Module 2 detects objects one frame at a time; it has no memory, so the "same"
car is an unrelated box each frame and the closest in-path object can flip
between vehicles in dense traffic. This module gives each object a stable
identity across frames and maintains its OWN smoothed kinematics, so the
decision engine (Module 3) can reason about specific objects' trajectories
instead of re-deriving everything from a flickering nearest box.

Design — a deliberately small, dependency-free tracker (a stripped-down SORT):

    - Association: greedy IoU matching of this frame's detections to existing
      tracks (no Hungarian solver, no scipy — object counts are small).
    - Motion: an optional constant-velocity model (TRACK_PREDICT_ON_COAST).
      Each track carries a per-corner pixel velocity; a coasting (unmatched)
      track is associated against its PREDICTED box, and a known closing hazard
      keeps a live, bounded distance/TTC between detections instead of freezing.
      This is what lets perception run below the video frame rate (async / high
      fps) without a tracked threat going stale. Disable it to fall back to the
      original constant-position behaviour.
    - Per-track kinematics: EMA-smoothed monocular distance (with the same
      spike rejection as before, but now per-object so a jump is real object
      motion, not an identity swap), a derived closing speed, and a real-seconds
      time-to-collision.
    - Lifecycle: a track is 'tentative' until it has been seen TRACK_MIN_HITS
      times (this alone rejects one-frame ghosts — they never confirm), then
      'confirmed'. A confirmed track coasts for up to TRACK_MAX_AGE frames
      without a detection (holding through brief dropouts) before it is dropped.

Pure Python + cv2 for drawing; adds negligible CPU.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

import config as cfg

logger = logging.getLogger(__name__)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """One tracked object, carrying its own smoothed kinematics across frames."""
    id:    int
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    risk:    str  = "LOW"
    in_path: bool = False

    smoothed_distance_m: Optional[float] = None
    closing_speed_mps:   float           = 0.0
    ttc_s:               Optional[float] = None

    hits:              int = 0     # total matched detections
    time_since_update: int = 0     # frames since last matched (0 = matched this frame)
    confirmed:         bool = False

    # Constant-velocity motion model — per-corner pixel velocity (px/s), zero
    # until the track has been matched a couple of times. Enables predicted-box
    # association after a dropout and honest coasting between detections.
    vx1: float = 0.0
    vy1: float = 0.0
    vx2: float = 0.0
    vy2: float = 0.0
    coast_predicted:   int = 0     # consecutive coast frames whose TTC we've extrapolated

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def _vel_ready(self) -> bool:
        """Velocity is trustworthy for prediction only after a few matches."""
        return self.hits >= cfg.TRACK_VEL_MIN_HITS

    def predict_box(self, dt: float) -> Tuple[int, int, int, int]:
        """Box extrapolated one step forward by the current per-corner velocity."""
        return (int(self.x1 + self.vx1 * dt), int(self.y1 + self.vy1 * dt),
                int(self.x2 + self.vx2 * dt), int(self.y2 + self.vy2 * dt))

    @property
    def center(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    @property
    def bottom_center(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int(self.y2))

    # -- internal updates ------------------------------------------------

    def _update_box(self, det, dt: float) -> None:
        a = cfg.TRACK_BBOX_EMA
        nx1 = a * det.x1 + (1 - a) * self.x1
        ny1 = a * det.y1 + (1 - a) * self.y1
        nx2 = a * det.x2 + (1 - a) * self.x2
        ny2 = a * det.y2 + (1 - a) * self.y2
        # Per-corner velocity (px/s), EMA-smoothed. Derived from the smoothed-box
        # displacement, which at steady state tracks the true per-frame motion.
        if dt > 1e-6:
            b = cfg.TRACK_VEL_EMA
            self.vx1 = b * (nx1 - self.x1) / dt + (1 - b) * self.vx1
            self.vy1 = b * (ny1 - self.y1) / dt + (1 - b) * self.vy1
            self.vx2 = b * (nx2 - self.x2) / dt + (1 - b) * self.vx2
            self.vy2 = b * (ny2 - self.y2) / dt + (1 - b) * self.vy2
        self.x1, self.y1, self.x2, self.y2 = nx1, ny1, nx2, ny2

    def _update_kinematics(self, det, dt: float) -> None:
        """EMA the per-object distance, derive closing speed + TTC."""
        raw_d = det.distance_m
        if self.smoothed_distance_m is None:
            self.smoothed_distance_m = raw_d
            self.closing_speed_mps = 0.0
            self.ttc_s = None
            return

        prev = self.smoothed_distance_m
        jump = abs(raw_d - prev)
        if jump <= cfg.MAX_PLAUSIBLE_JUMP_M:
            self.smoothed_distance_m = (
                cfg.DIST_EMA_ALPHA * raw_d + (1 - cfg.DIST_EMA_ALPHA) * prev
            )
            inst_v = (prev - self.smoothed_distance_m) / dt        # + = approaching
            self.closing_speed_mps = (
                cfg.CLOSING_EMA_ALPHA * inst_v
                + (1 - cfg.CLOSING_EMA_ALPHA) * self.closing_speed_mps
            )
            self.closing_speed_mps = max(
                -cfg.VCLOSE_CLAMP_MPS, min(cfg.VCLOSE_CLAMP_MPS, self.closing_speed_mps)
            )
        else:
            # A large jump for a SAME-identity track is bbox/monocular noise:
            # step toward the reading and don't trust closing this frame.
            step = min(cfg.MAX_DIST_STEP_M, jump)
            self.smoothed_distance_m = prev + step if raw_d > prev else prev - step
            self.closing_speed_mps = 0.0

        self.ttc_s = (
            self.smoothed_distance_m / self.closing_speed_mps
            if self.closing_speed_mps > cfg.MIN_CLOSING_MPS else None
        )

    def _coast(self, dt: float) -> None:
        """No detection this frame.

        With the motion model on, advance the box by its velocity and — for a
        track that was already CLOSING — keep distance/TTC live for a bounded
        number of coast frames (then freeze). Erring toward keeping a lost-but-
        closing threat 'live' between detections is the safe direction and is
        what makes below-frame-rate perception safe. Never fabricates closing
        for a track that wasn't already closing. With the model off, this is the
        original behaviour: hold position, drop closing/TTC.
        """
        predict = cfg.TRACK_PREDICT_ON_COAST and self._vel_ready()
        if predict:
            self.x1 += self.vx1 * dt
            self.y1 += self.vy1 * dt
            self.x2 += self.vx2 * dt
            self.y2 += self.vy2 * dt

        if (cfg.TRACK_PREDICT_ON_COAST
                and self.coast_predicted < cfg.TRACK_COAST_PREDICT_MAX
                and self.smoothed_distance_m is not None
                and self.closing_speed_mps > cfg.MIN_CLOSING_MPS):
            # extrapolate the approach; closing speed is held (not re-derived)
            self.smoothed_distance_m = max(
                0.0, self.smoothed_distance_m - self.closing_speed_mps * dt)
            self.ttc_s = (self.smoothed_distance_m / self.closing_speed_mps
                          if self.closing_speed_mps > cfg.MIN_CLOSING_MPS else 0.0)
            self.coast_predicted += 1
        else:
            # not closing, or the extrapolation budget is spent -> stop trusting it
            self.closing_speed_mps = 0.0
            self.ttc_s = None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class MultiObjectTracker:
    """
    Greedy-IoU multi-object tracker with per-track kinematics.

    Usage::

        tracker = MultiObjectTracker()
        tracks  = tracker.update(obj_result.detections)   # per frame
        annotated = tracker.annotate(annotated, tracks)
    """

    def __init__(self) -> None:
        self.tracks: List[Track] = []
        self._next_id = 1
        self._prev_tick = cv2.getTickCount()
        logger.info("MultiObjectTracker initialised.")

    def reset(self) -> None:
        """Drop all tracks (e.g. when a video loops back to the start)."""
        self.tracks = []
        self._next_id = 1

    def _measure_dt(self) -> float:
        tick = cv2.getTickCount()
        dt = (tick - self._prev_tick) / cv2.getTickFrequency()
        self._prev_tick = tick
        return max(cfg.DT_CLAMP_MIN_S, min(cfg.DT_CLAMP_MAX_S, dt))

    def update(self, detections: List) -> List[Track]:
        """
        Advance the tracker one frame and return the currently active tracks.

        Args:
            detections: this frame's Detection list from Module 2.

        Returns:
            All active tracks (confirmed + tentative). Consumers that make
            decisions should use only tracks with ``.confirmed`` True.
        """
        dt = self._measure_dt()
        detections = detections or []

        # ── Greedy IoU association (highest overlap first) ────────────
        # A track that missed the previous frame is matched against its PREDICTED
        # box (constant-velocity) so a fast object re-acquires its ID instead of
        # spawning a new track; a continuously-tracked object uses its measured box.
        pairs = []
        for ti, trk in enumerate(self.tracks):
            if (trk.time_since_update > 0 and cfg.TRACK_PREDICT_ON_COAST
                    and trk._vel_ready()):
                ref = trk.predict_box(dt)
            else:
                ref = trk.bbox
            for di, det in enumerate(detections):
                iou = _iou(ref, (det.x1, det.y1, det.x2, det.y2))
                if iou >= cfg.TRACK_IOU_MIN:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        matched_t, matched_d, matches = set(), set(), []
        for _iou_val, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            matches.append((ti, di))

        # ── Update matched tracks ─────────────────────────────────────
        for ti, di in matches:
            trk, det = self.tracks[ti], detections[di]
            trk._update_box(det, dt)
            trk._update_kinematics(det, dt)
            trk.label = det.label
            trk.confidence = det.confidence
            trk.risk = det.risk
            trk.in_path = det.in_path
            trk.hits += 1
            trk.time_since_update = 0
            trk.coast_predicted = 0            # a real detection resets the extrapolation budget
            if not trk.confirmed and trk.hits >= cfg.TRACK_MIN_HITS:
                trk.confirmed = True

        # ── Age unmatched tracks (coast, then drop) ───────────────────
        for ti, trk in enumerate(self.tracks):
            if ti not in matched_t:
                trk.time_since_update += 1
                trk._coast(dt)
        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.TRACK_MAX_AGE]

        # ── Spawn new tracks for unmatched detections ─────────────────
        for di, det in enumerate(detections):
            if di in matched_d:
                continue
            trk = Track(
                id=self._next_id, label=det.label,
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                confidence=det.confidence, risk=det.risk, in_path=det.in_path,
                hits=1, time_since_update=0,
            )
            trk._update_kinematics(det, dt)          # seed distance
            self._next_id += 1
            self.tracks.append(trk)

        return list(self.tracks)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def annotate(self, frame, tracks: List[Track]):
        """
        Tag each confirmed track with its stable ID (Module 2 already drew the
        risk-coloured box; this makes the tracking itself visible — the same
        number should stay on the same object across frames). The live tracked
        count is shown in the decision HUD's telemetry line.
        """
        for t in tracks:
            if not t.confirmed:
                continue
            x1, y1, _, _ = t.bbox
            cv2.putText(frame, f"#{t.id}", (x1 + 2, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.COLOR_TRACK_ID, 2, cv2.LINE_AA)
        return frame
