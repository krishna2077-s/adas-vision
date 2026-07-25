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
    - Motion: none. The last smoothed box is the prediction. At these frame
      rates a constant-position assumption associates reliably and can't drift.
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

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @property
    def center(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    @property
    def bottom_center(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int(self.y2))

    # -- internal updates ------------------------------------------------

    def _update_box(self, det) -> None:
        a = cfg.TRACK_BBOX_EMA
        self.x1 = a * det.x1 + (1 - a) * self.x1
        self.y1 = a * det.y1 + (1 - a) * self.y1
        self.x2 = a * det.x2 + (1 - a) * self.x2
        self.y2 = a * det.y2 + (1 - a) * self.y2

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

    def _coast(self) -> None:
        """No detection this frame: hold position, don't fabricate closing/TTC."""
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
        pairs = []
        for ti, trk in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou = _iou(trk.bbox, (det.x1, det.y1, det.x2, det.y2))
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
            trk._update_box(det)
            trk._update_kinematics(det, dt)
            trk.label = det.label
            trk.confidence = det.confidence
            trk.risk = det.risk
            trk.in_path = det.in_path
            trk.hits += 1
            trk.time_since_update = 0
            if not trk.confirmed and trk.hits >= cfg.TRACK_MIN_HITS:
                trk.confirmed = True

        # ── Age unmatched tracks (coast, then drop) ───────────────────
        for ti, trk in enumerate(self.tracks):
            if ti not in matched_t:
                trk.time_since_update += 1
                trk._coast()
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
