"""
forward_collision_warning.py — Module 10: Forward Collision Warning (FCW).

A visible, escalating collision alert layered on top of the decision engine.
Real cars show this on the cluster/windshield: an amber "risk" cue that ramps to
a red "BRAKE" flash with a chime as a lead vehicle closes. This module renders
exactly that, driven by kinematics the system already computes.

Single source of truth — it does NOT re-select a hazard or invent a second
time-to-collision. It consumes the ONE nearest in-path hazard the decision
engine (Module 3) already arbitrated (`decision.hazard_id / ttc_s /
smoothed_distance_m / closing_speed_mps`), so the banner and the brain can never
disagree about which object matters or how close it is.

Four driver-facing stages, escalating on the hazard's TTC (and a hard distance
floor for the imminent case):

    0 CLEAR      no closing in-path threat — no banner
    1 CAUTION    TTC <= FCW_TTC_CAUTION_S   — amber "COLLISION RISK"
    2 WARNING    TTC <= FCW_TTC_WARN_S      — orange "COLLISION WARNING"
    3 IMMINENT   TTC <= FCW_TTC_IMMINENT_S  — red flashing "BRAKE", full-frame
                 or lead <= FCW_DIST_IMMINENT_M   border + chime

Like the decision engine, it escalates instantly and releases slowly (a stage is
held FCW_HOLD_FRAMES frames before it may step down one level), so a single noisy
frame can neither raise a false alarm that sticks nor blink the banner off during
a real threat.

⚠️ Advisory only. FCW warns a human — it actuates nothing. It never touches a
vehicle's brakes, throttle, or steering. Research / learning use only.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import cv2

import config as cfg

logger = logging.getLogger(__name__)

CLEAR, CAUTION, WARNING, IMMINENT = 0, 1, 2, 3
STAGE_NAMES = ["CLEAR", "CAUTION", "WARNING", "IMMINENT"]
_STAGE_TITLES = {
    CAUTION:  "COLLISION RISK",
    WARNING:  "COLLISION WARNING",
    IMMINENT: "BRAKE",
}


@dataclass
class FCWState:
    """One frame of forward-collision-warning output (for drawing / logging)."""
    level:       int = CLEAR
    name:        str = "CLEAR"
    ttc_s:       Optional[float] = None
    distance_m:  Optional[float] = None
    closing_mps: float = 0.0
    label:       Optional[str] = None
    hazard_id:   Optional[int] = None


class ForwardCollisionWarning:
    """Staged, hysteretic collision alert. Consumes the engine's hazard; warns only."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self.stage = CLEAR
        self._down = 0                 # consecutive calmer frames (release debounce)
        self._last_chime_s = -1e9
        logger.info(f"ForwardCollisionWarning initialised ({frame_width}x{frame_height}).")

    def reset(self) -> None:
        """Clear latched state — call when a video loops back to the start."""
        self.stage = CLEAR
        self._down = 0

    # ------------------------------------------------------------------
    # Per-frame stage decision
    # ------------------------------------------------------------------

    def _raw_stage(self, ttc, dist, closing, label) -> int:
        """Instantaneous stage from this frame's hazard kinematics (pre-hysteresis)."""
        if label is None or closing <= cfg.FCW_MIN_CLOSING_MPS or ttc is None:
            return CLEAR
        if ttc <= cfg.FCW_TTC_IMMINENT_S or (dist is not None and dist <= cfg.FCW_DIST_IMMINENT_M):
            return IMMINENT
        if ttc <= cfg.FCW_TTC_WARN_S:
            return WARNING
        if ttc <= cfg.FCW_TTC_CAUTION_S:
            return CAUTION
        return CLEAR

    def process(self, decision, tracks: Optional[List] = None) -> FCWState:
        """
        Fold the engine's current hazard into a debounced FCW stage.

        `decision` is the DrivingDecision from Module 3; `tracks` is unused for
        the stage decision (the engine already chose the hazard) but is accepted
        so callers can pass it uniformly — draw() uses it to bracket the threat.
        """
        ttc     = getattr(decision, "ttc_s", None)
        dist    = getattr(decision, "smoothed_distance_m", None)
        closing = getattr(decision, "closing_speed_mps", 0.0) or 0.0
        label   = getattr(decision, "hazard_label", None)

        raw = self._raw_stage(ttc, dist, closing, label)

        # Fast escalation, slow one-level-at-a-time release (mirror the engine).
        if raw > self.stage:
            self.stage = raw
            self._down = 0
        elif raw < self.stage:
            self._down += 1
            if self._down >= cfg.FCW_HOLD_FRAMES:
                self.stage -= 1
                self._down = 0
        else:
            self._down = 0

        if self.stage >= WARNING:
            self._chime(self.stage)

        return FCWState(
            level=self.stage,
            name=STAGE_NAMES[self.stage],
            ttc_s=ttc,
            distance_m=dist,
            closing_mps=closing,
            label=label,
            hazard_id=getattr(decision, "hazard_id", None),
        )

    # ------------------------------------------------------------------
    # Optional audible chime (opt-in, non-blocking, platform best-effort)
    # ------------------------------------------------------------------

    def _chime(self, level: int) -> None:
        if not cfg.FCW_AUDIO:
            return
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if now - self._last_chime_s < cfg.FCW_AUDIO_COOLDOWN_S:
            return
        self._last_chime_s = now
        freq = 1200 if level >= IMMINENT else 800
        dur = 300 if level >= IMMINENT else 160
        try:
            import threading
            threading.Thread(target=self._beep, args=(freq, dur), daemon=True).start()
        except Exception:
            pass

    @staticmethod
    def _beep(freq: int, dur_ms: int) -> None:
        try:
            import winsound
            winsound.Beep(int(freq), int(dur_ms))
        except Exception:
            try:
                print("\a", end="", flush=True)   # terminal bell fallback
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _stage_color(self, level: int):
        return {
            CAUTION:  cfg.COLOR_FCW_CAUTION,
            WARNING:  cfg.COLOR_FCW_WARNING,
            IMMINENT: cfg.COLOR_FCW_IMMINENT,
        }.get(level, (200, 200, 200))

    def draw(self, frame, state: FCWState, tracks: Optional[List] = None):
        """
        Top-centre staged banner + live TTC, a warning bracket on the threat
        object, and (IMMINENT) a flashing full-frame border. Drawn last, so a
        collision warning is never occluded by another overlay.
        """
        if state.level == CLEAR:
            return frame

        color = self._stage_color(state.level)
        blink_on = not (state.level == IMMINENT and (self._tick() % 2 == 0))

        # ── warning bracket around the threat's bounding box ──────────
        if tracks and state.hazard_id is not None and state.level >= WARNING:
            self._bracket_hazard(frame, tracks, state.hazard_id, color)

        # ── full-frame red border while imminent (blinks) ─────────────
        if state.level == IMMINENT and blink_on:
            cv2.rectangle(frame, (2, 2), (self.w - 3, self.h - 3), color, 6)

        # ── top-centre banner ─────────────────────────────────────────
        bw, bh = 480, 44
        x0 = self.w // 2 - bw // 2
        y0 = 8
        if blink_on:
            self._filled(frame, x0, y0, bw, bh, color, 0.78)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), color, 2)

        # warning glyph (triangle + '!') on the left
        self._triangle(frame, x0 + 26, y0 + bh // 2, 14)

        title = _STAGE_TITLES.get(state.level, "")
        cv2.putText(frame, title, (x0 + 52, y0 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.82, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, title, (x0 + 52, y0 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 1, cv2.LINE_AA)

        # live TTC + distance on the right of the banner
        if state.ttc_s is not None:
            metric = f"TTC {state.ttc_s:.1f}s"
            if state.distance_m is not None:
                metric += f"  {state.distance_m:.0f}m"
            (tw, _), _ = cv2.getTextSize(metric, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, metric, (x0 + bw - tw - 14, y0 + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _tick(self) -> int:
        """A slow frame counter for blinking (independent of caller state)."""
        # cv2 tick count in ~quarter-second buckets -> ~2 Hz blink
        return int((cv2.getTickCount() / cv2.getTickFrequency()) * 4)

    @staticmethod
    def _filled(frame, x, y, w, h, color, alpha) -> None:
        sub = frame[y:y + h, x:x + w]
        if sub.size == 0:
            return
        overlay = sub.copy()
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)

    @staticmethod
    def _triangle(frame, cx, cy, r) -> None:
        import numpy as np
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillConvexPoly(frame, pts, (0, 0, 0), cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r + 5), (cx, cy + r - 8), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy + r - 4), 1, (255, 255, 255), -1)

    def _bracket_hazard(self, frame, tracks, hazard_id, color) -> None:
        trk = next((t for t in tracks if getattr(t, "id", None) == hazard_id), None)
        if trk is None:
            return
        x1, y1, x2, y2 = trk.bbox
        L = max(12, int(0.25 * (x2 - x1)))          # corner-bracket arm length
        for (px, py, dx, dy) in (
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ):
            cv2.line(frame, (px, py), (px + dx * L, py), color, 3, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * L), color, 3, cv2.LINE_AA)
