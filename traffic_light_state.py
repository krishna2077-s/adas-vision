"""
traffic_light_state.py — Module 11a: traffic-light STATE recognition.

YOLO (Module 2) already localises "traffic light" boxes; what it does NOT tell
us is the colour — and a light's colour is the whole point. This module reads
the state (RED / AMBER / GREEN) of each detected light from the box crop, picks
the one controlling our lane, smooths it over time, and exposes a plain advisory
("RED AHEAD — prepare to stop") for the planner and the HUD.

How the state is read (no training needed, honest and cheap):

    1. Take the light's box crop, convert to HSV.
    2. Keep only *illuminated* pixels (high value + saturation) — a lit bulb is
       bright and vivid; the dark housing and sky are rejected.
    3. Split those into red / amber / green by hue, and add a mild geometric
       prior (in a vertical head red sits high, green low) to break ties on
       small or blurry lights.
    4. The dominant colour is the state; its share is the confidence. Too few
       lit pixels -> UNKNOWN (honest: we don't guess in the dark).

    5. Temporal vote: the controlling light's state is the mode of the last
       few frames, so one noisy frame can't flip RED<->GREEN.

⚠️ Advisory only. This informs a human and the advisory planner — it actuates
nothing. Colour estimation from a single camera is imperfect (sun glare, LED
flicker, occlusion); treat it as a cue, not ground truth.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)

RED, AMBER, GREEN, UNKNOWN = "RED", "AMBER", "GREEN", "UNKNOWN"

_STATE_COLORS = {                    # BGR for drawing
    RED:     (0, 0, 255),
    AMBER:   (0, 200, 255),
    GREEN:   (0, 220, 0),
    UNKNOWN: (150, 150, 150),
}


@dataclass
class TrafficLight:
    """One detected light with its estimated state."""
    x1: int
    y1: int
    x2: int
    y2: int
    state: str = UNKNOWN
    confidence: float = 0.0
    in_path: bool = False
    distance_m: Optional[float] = None


class TrafficLightReader:
    """Reads RED/AMBER/GREEN from YOLO traffic-light boxes; votes over time."""

    def __init__(self) -> None:
        self._vote: "deque[str]" = deque(maxlen=cfg.TL_VOTE_WINDOW)
        self.controlling_state: Optional[str] = None
        logger.info("TrafficLightReader initialised.")

    def reset(self) -> None:
        self._vote.clear()
        self.controlling_state = None

    # ------------------------------------------------------------------
    def _classify(self, crop: np.ndarray) -> tuple:
        """Return (state, confidence) for one light crop."""
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 4:
            return UNKNOWN, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        lit = (v >= cfg.TL_MIN_VALUE) & (s >= cfg.TL_MIN_SAT)
        lit_count = int(lit.sum())
        if lit_count < max(6, 0.02 * crop.shape[0] * crop.shape[1]):
            return UNKNOWN, 0.0

        red   = lit & (((h <= cfg.TL_RED_HUE_HI) | (h >= cfg.TL_RED_HUE_LO2)))
        amber = lit & (h > cfg.TL_RED_HUE_HI) & (h <= cfg.TL_AMBER_HUE_HI)
        green = lit & (h > cfg.TL_AMBER_HUE_HI) & (h <= cfg.TL_GREEN_HUE_HI)

        counts = {RED: int(red.sum()), AMBER: int(amber.sum()), GREEN: int(green.sum())}

        # Geometric prior: red high, green low in a vertical head. Nudge the
        # colour whose lit pixels sit in the expected third (mild, tie-break only).
        H = crop.shape[0]
        for state, mask in ((RED, red), (GREEN, green)):
            m = int(mask.sum())
            if m == 0:
                continue
            ys = np.nonzero(mask)[0]
            frac = ys.mean() / max(1, H)          # 0 top .. 1 bottom
            if state == RED and frac < 0.5:
                counts[RED] = int(counts[RED] * 1.25)
            if state == GREEN and frac > 0.5:
                counts[GREEN] = int(counts[GREEN] * 1.25)

        total = sum(counts.values())
        if total < 4:
            return UNKNOWN, 0.0
        state = max(counts, key=counts.get)
        return state, counts[state] / total

    # ------------------------------------------------------------------
    def read(self, frame: np.ndarray, detections: List) -> List[TrafficLight]:
        """Classify every 'traffic light' detection; update the controlling vote."""
        lights: List[TrafficLight] = []
        for d in detections or []:
            if getattr(d, "label", None) != "traffic light":
                continue
            x1, y1 = max(0, d.x1), max(0, d.y1)
            x2, y2 = min(frame.shape[1], d.x2), min(frame.shape[0], d.y2)
            state, conf = self._classify(frame[y1:y2, x1:x2])
            lights.append(TrafficLight(
                x1, y1, x2, y2, state=state, confidence=conf,
                in_path=getattr(d, "in_path", False),
                distance_m=getattr(d, "distance_m", None)))

        controlling = self._pick_controlling(lights)
        if controlling is not None and controlling.state != UNKNOWN:
            self._vote.append(controlling.state)
        elif controlling is None:
            # No light in view — let the vote decay so a stale RED doesn't persist.
            if self._vote:
                self._vote.popleft()

        self.controlling_state = (
            max(set(self._vote), key=self._vote.count) if self._vote else None
        )
        return lights

    @staticmethod
    def _pick_controlling(lights: List[TrafficLight]) -> Optional[TrafficLight]:
        """The light governing our lane: prefer in-path; else the largest box."""
        if not lights:
            return None
        in_path = [l for l in lights if l.in_path]
        pool = in_path or lights
        return max(pool, key=lambda l: (l.x2 - l.x1) * (l.y2 - l.y1))

    # ------------------------------------------------------------------
    def advisory(self) -> Optional[str]:
        """Plain-language advisory from the smoothed controlling state (or None)."""
        s = self.controlling_state
        if s == RED:
            return "RED light ahead -- prepare to stop"
        if s == AMBER:
            return "AMBER -- light changing"
        if s == GREEN:
            return "GREEN -- proceed with care"
        return None

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, lights: List[TrafficLight]):
        """Ring each light in its state colour; show a controlling-state chip."""
        for l in lights:
            col = _STATE_COLORS.get(l.state, _STATE_COLORS[UNKNOWN])
            cv2.rectangle(frame, (l.x1, l.y1), (l.x2, l.y2), col, 2)
            tag = l.state if l.state != UNKNOWN else "?"
            cv2.putText(frame, tag, (l.x1, max(12, l.y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

        s = self.controlling_state
        if s:
            col = _STATE_COLORS.get(s, _STATE_COLORS[UNKNOWN])
            x, y = 14, 30
            cv2.rectangle(frame, (x - 6, y - 20), (x + 150, y + 8), cfg.COLOR_HUD_BG, -1)
            cv2.circle(frame, (x + 6, y - 6), 7, col, -1)
            cv2.putText(frame, f"LIGHT: {s}", (x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        return frame
