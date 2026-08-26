"""
road_detection.py — Module 1b: drivable-road-surface following (no markings).

Painted-line detection (lane_detection.py) is precise but only works where
paint exists. Most rural / hill roads in India have none — so this module
finds the DRIVABLE ROAD SURFACE itself and steers along its centreline. It is
the fallback guidance source: markings first, road surface second, honest
"no guidance" third.

How it works (all classical CV, done on a downscaled frame — a few ms on CPU):

    1. Texture gate      — asphalt is SMOOTH; vegetation/gravel is busy.
                           Keep pixels whose local intensity std is low.
    2. Soft colour gate  — road is low-chroma (grey-ish); drop strongly
                           coloured pixels (grass, soil, sky-blue).
    3. Seeded connectivity — flood only the region CONNECTED to the patch
                           directly in front of the car. A smooth hazy hill in
                           the distance is not connected to the bonnet except
                           through the road, so it gets cut off.
    4. Centreline        — per-row centre of the road mask gives a CURVED
                           path, so turns and hills are followed naturally.
    5. Look-ahead steering — the centreline at ~2/3 of the visible road ahead
                           vs the frame centre -> offset -> steering word.
    6. Honest confidence — from row coverage; below a floor the module says
                           NO GUIDANCE instead of guessing.

Output is a LaneDetectionResult (same dataclass as Module 1), so the rest of
the system consumes it unchanged; confidence is capped at 0.5 ("estimated")
because surface-following is inherently less precise than paint.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from lane_detection import LaneDetectionResult

logger = logging.getLogger(__name__)


class RoadDetector:
    """
    Drivable-road-surface detector for unmarked roads.

    Usage::

        road = RoadDetector(frame_width=1280, frame_height=720)
        result, annotated = road.process(frame)      # LaneDetectionResult
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self._smooth_look_x: Optional[float] = None   # EMA of the look-ahead centre
        logger.info(f"RoadDetector initialised ({frame_width}x{frame_height})")

    def reset(self) -> None:
        self._smooth_look_x = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        draw_on: Optional[np.ndarray] = None,
    ) -> Tuple[LaneDetectionResult, np.ndarray]:
        """
        Detects the drivable road surface and returns steering guidance.

        Args:
            frame:   RAW BGR frame — detection runs on clean pixels.
            draw_on: optional frame to draw the overlay onto (e.g. the frame
                     already annotated by other modules). Defaults to `frame`.

        Returns:
            (LaneDetectionResult, annotated_frame). confidence is 0.5 when the
            road is followed (estimated guidance), 0.0 when no reliable road
            region was found.
        """
        mask, centre, coverage = self._detect_surface(frame)

        result = LaneDetectionResult(frame_center_x=self.w // 2)
        canvas = draw_on if draw_on is not None else frame
        annotated = canvas

        if coverage >= cfg.ROAD_MIN_COVERAGE and len(centre) >= cfg.ROAD_MIN_ROWS:
            # Look-ahead point ~2/3 up the visible road ahead
            look = centre[min(len(centre) - 1, int(len(centre) * cfg.ROAD_LOOKAHEAD_FRAC))]
            look_x = float(look[1])

            # Temporal smoothing so the steering target doesn't jitter
            if self._smooth_look_x is None:
                self._smooth_look_x = look_x
            else:
                a = cfg.ROAD_SMOOTHING_ALPHA
                self._smooth_look_x = a * look_x + (1 - a) * self._smooth_look_x

            cx = int(self._smooth_look_x)
            offset = cx - result.frame_center_x
            result.lane_center_x = cx
            result.offset_px = offset
            result.confidence = 0.5                    # estimated, not paint-precise
            result.steering = self._steering(offset)
            annotated = self._annotate(canvas, mask, centre, cx, offset)
        else:
            # Not enough road visible — say so instead of guessing.
            self._smooth_look_x = None
            result.confidence = 0.0
            result.steering = "UNKNOWN -- no road surface lock"

        return result, annotated

    # ------------------------------------------------------------------
    # Surface detection
    # ------------------------------------------------------------------

    def _detect_surface(self, frame: np.ndarray):
        """Returns (full-res road mask, centreline [(y, x, width)...], coverage 0-1)."""
        ds = cfg.ROAD_DOWNSCALE
        small = cv2.resize(frame, (self.w // ds, self.h // ds), interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2)

        # 1. Texture: local std of intensity (asphalt smooth, vegetation busy)
        mu = cv2.blur(gray, (7, 7))
        mu2 = cv2.blur(gray * gray, (7, 7))
        std = np.sqrt(np.maximum(mu2 - mu * mu, 0))

        cand = ((std < cfg.ROAD_TEXTURE_STD_MAX)
                & (chroma < cfg.ROAD_CHROMA_MAX)
                & (gray > cfg.ROAD_MIN_BRIGHTNESS)).astype(np.uint8) * 255

        cand[:int(cfg.ROAD_HORIZON_FRAC * h), :] = 0    # sky / far hills
        cand[int(cfg.ROAD_BONNET_FRAC * h):, :] = 0     # bonnet (also smooth+grey)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=2)

        # 2. Seeded connectivity: keep only components touching the seed strip
        #    directly in front of the car.
        n, lbl = cv2.connectedComponents(cand, 8)
        seed_y = int(cfg.ROAD_SEED_Y_FRAC * h)
        seeds = set(lbl[seed_y, int(0.30 * w):int(0.70 * w)].tolist()) - {0}
        if not seeds:   # widen once (car may sit at the road edge)
            seeds = set(lbl[int(0.85 * h), int(0.20 * w):int(0.80 * w)].tolist()) - {0}
        if not seeds:
            return np.zeros((self.h, self.w), np.uint8), [], 0.0

        road_s = np.isin(lbl, list(seeds)).astype(np.uint8) * 255
        road_s = cv2.morphologyEx(road_s, cv2.MORPH_OPEN, k)

        # 3. Per-row centreline (in full-res coordinates)
        centre: List[Tuple[int, int, int]] = []
        top = int(cfg.ROAD_HORIZON_FRAC * h)
        for y in range(h - 1, top, -3):
            xs = np.where(road_s[y] > 0)[0]
            if len(xs) > 8:
                centre.append((y * ds, int((xs.min() + xs.max()) / 2) * ds,
                               int(xs.max() - xs.min()) * ds))
        coverage = len(centre) / max(1, (h - top) // 3)

        mask = cv2.resize(road_s, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return mask, centre, coverage

    # ------------------------------------------------------------------
    # Steering + drawing
    # ------------------------------------------------------------------

    def _steering(self, offset: int) -> str:
        a = abs(offset)
        direction = "RIGHT" if offset > 0 else "LEFT"
        if a < cfg.STEER_THRESHOLD_SLIGHT:
            return "STRAIGHT (road)"
        elif a < cfg.STEER_THRESHOLD_MODERATE:
            return f"SLIGHT {direction} (road)"
        elif a < cfg.STEER_THRESHOLD_HARD:
            return f"MODERATE {direction} (road)"
        return f"HARD {direction} (road)"

    def _annotate(self, frame, mask, centre, look_x, offset) -> np.ndarray:
        out = frame.copy()

        # Translucent road-surface fill
        green = np.zeros_like(out)
        green[mask > 0] = cfg.ROAD_FILL_COLOR
        out = cv2.addWeighted(out, 1.0, green, cfg.ROAD_FILL_ALPHA, 0)

        # Curved centreline
        pts = [(x, y) for (y, x, _wd) in centre]
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], cfg.COLOR_CENTER_LINE, 3, cv2.LINE_AA)

        # Look-ahead marker + frame-centre reference
        cx = self.w // 2
        cv2.line(out, (cx, self.h), (cx, int(0.5 * self.h)), (255, 255, 255), 1, cv2.LINE_AA)
        y_marker = int(0.62 * self.h)
        cv2.circle(out, (look_x, y_marker), 8, cfg.COLOR_CENTER_LINE, -1)
        cv2.line(out, (cx, y_marker), (look_x, y_marker), cfg.COLOR_CENTER_LINE, 2)

        # Mode tag so the driver knows this is surface-following, not paint
        cv2.putText(out, "ROAD-FOLLOW (estimated)", (14, self.h - 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cfg.COLOR_CENTER_LINE, 1, cv2.LINE_AA)
        return out
