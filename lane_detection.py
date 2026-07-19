"""
lane_detection.py — Module 1: Real-time lane detection using OpenCV.

Pipeline per frame:
    1. Grayscale + Gaussian blur          (reduce noise)
    2. Canny edge detection               (find edges)
    3. ROI mask                           (ignore sky / bonnet / sides)
    4. Hough Line Transform               (find straight lines in edges)
    5. Line averaging                     (merge raw lines → left + right lane)
    6. Exponential smoothing              (stabilise across frames)
    7. Lane centre + offset calculation   (how far are we from centre?)
    8. Steering decision                  (what correction is needed?)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Lane:
    """Represents one detected lane line as two endpoints."""
    x1: int
    y1: int
    x2: int
    y2: int

    def slope(self) -> float:
        dx = self.x2 - self.x1
        if dx == 0:
            return float("inf")
        return (self.y2 - self.y1) / dx

    def midpoint(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class LaneDetectionResult:
    """All outputs from one frame of lane detection."""
    left_lane:      Optional[Lane]   = None
    right_lane:     Optional[Lane]   = None
    lane_center_x:  Optional[int]    = None   # X pixel of midpoint between lanes
    frame_center_x: int              = 0      # X pixel of frame centre
    offset_px:      int              = 0      # lane_center - frame_center (+ = drift right)
    steering:       str              = "UNKNOWN"
    confidence:     float            = 0.0    # 0.0 – 1.0
    fps:            float            = 0.0


# ---------------------------------------------------------------------------
# Core detector class
# ---------------------------------------------------------------------------

class LaneDetector:
    """
    Stateful lane detector that smooths detections across frames using
    an exponential moving average on the lane line endpoints.

    Usage::

        detector = LaneDetector(frame_width=1280, frame_height=720)
        result, annotated_frame = detector.process(frame)
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height

        # Smoothed lane endpoints (floats, cast to int for drawing)
        self._smooth_left:  Optional[np.ndarray] = None   # [x1,y1,x2,y2]
        self._smooth_right: Optional[np.ndarray] = None

        # FPS tracking
        self._prev_tick = cv2.getTickCount()

        # Build ROI polygon once (depends only on frame size)
        self._roi_polygon = self._build_roi()

        logger.info(f"LaneDetector initialised ({frame_width}x{frame_height})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Tuple[LaneDetectionResult, np.ndarray]:
        """
        Runs the full lane detection pipeline on one BGR frame.

        Args:
            frame: BGR numpy array from cv2.VideoCapture.read().

        Returns:
            (LaneDetectionResult, annotated_frame)
        """
        # ── 1. Pre-process ────────────────────────────────────────────
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, cfg.BLUR_KERNEL_SIZE, 0)
        edges   = cv2.Canny(blurred, cfg.CANNY_LOW, cfg.CANNY_HIGH)

        # ── 2. ROI mask ───────────────────────────────────────────────
        masked = self._apply_roi(edges)

        # ── 3. Hough lines ────────────────────────────────────────────
        raw_lines = cv2.HoughLinesP(
            masked,
            rho       = cfg.HOUGH_RHO,
            theta     = np.deg2rad(cfg.HOUGH_THETA),
            threshold = cfg.HOUGH_THRESHOLD,
            minLineLength = cfg.HOUGH_MIN_LENGTH,
            maxLineGap    = cfg.HOUGH_MAX_GAP,
        )

        # ── 4. Separate + average left / right ────────────────────────
        left_raw, right_raw = self._split_lines(raw_lines)
        left_lane  = self._average_lane(left_raw,  side="left")
        right_lane = self._average_lane(right_raw, side="right")

        # ── 5. Smooth across frames ───────────────────────────────────
        left_lane  = self._smooth("left",  left_lane)
        right_lane = self._smooth("right", right_lane)

        # ── 6. Compute offset + steering ─────────────────────────────
        result = self._compute_result(left_lane, right_lane)

        # ── 7. Annotate frame ─────────────────────────────────────────
        annotated = self._annotate(frame.copy(), masked, raw_lines, result)

        # ── 8. FPS ───────────────────────────────────────────────────
        tick = cv2.getTickCount()
        result.fps = cv2.getTickFrequency() / (tick - self._prev_tick)
        self._prev_tick = tick

        return result, annotated

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    def _build_roi(self) -> np.ndarray:
        """Builds the trapezoidal ROI polygon in pixel coordinates."""
        w, h = self.w, self.h
        return np.array([[
            (int(cfg.ROI_BOTTOM_LEFT_X  * w), int(cfg.ROI_BOTTOM_Y * h)),
            (int(cfg.ROI_TOP_LEFT_X     * w), int(cfg.ROI_TOP_Y    * h)),
            (int(cfg.ROI_TOP_RIGHT_X    * w), int(cfg.ROI_TOP_Y    * h)),
            (int(cfg.ROI_BOTTOM_RIGHT_X * w), int(cfg.ROI_BOTTOM_Y * h)),
        ]], dtype=np.int32)

    def _apply_roi(self, edges: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self._roi_polygon, 255)
        return cv2.bitwise_and(edges, mask)

    # ------------------------------------------------------------------
    # Line processing
    # ------------------------------------------------------------------

    def _split_lines(
        self,
        raw_lines: Optional[np.ndarray],
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Classifies raw Hough lines into left-lane and right-lane candidates
        based on slope sign and position.

        Left  lane: negative slope (going up-right), x < centre
        Right lane: positive slope (going up-left),  x > centre
        """
        left, right = [], []
        if raw_lines is None:
            return left, right

        cx = self.w / 2

        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            slope = (y2 - y1) / dx
            if abs(slope) < cfg.MIN_SLOPE or abs(slope) > cfg.MAX_SLOPE:
                continue

            mid_x = (x1 + x2) / 2
            if slope < 0 and mid_x < cx:
                left.append((slope, x1, y1, x2, y2))
            elif slope > 0 and mid_x > cx:
                right.append((slope, x1, y1, x2, y2))

        return left, right

    def _average_lane(
        self,
        lines: List[Tuple],
        side: str,
    ) -> Optional[Lane]:
        """
        Averages a list of (slope, x1,y1,x2,y2) into a single full-height lane line.
        Extrapolates to the bottom of the ROI and to the top of the ROI.
        """
        if not lines:
            return None

        slopes = [l[0] for l in lines]
        x1s    = [l[1] for l in lines]
        y1s    = [l[2] for l in lines]
        x2s    = [l[3] for l in lines]
        y2s    = [l[4] for l in lines]

        avg_slope = float(np.mean(slopes))
        avg_x1    = float(np.mean(x1s))
        avg_y1    = float(np.mean(y1s))

        # Extrapolate: y = avg_y1 + avg_slope * (x - avg_x1)
        # → x = avg_x1 + (y - avg_y1) / avg_slope
        if abs(avg_slope) < 1e-6:
            return None

        y_bottom = int(cfg.ROI_BOTTOM_Y * self.h)
        y_top    = int(cfg.ROI_TOP_Y    * self.h)

        x_bottom = int(avg_x1 + (y_bottom - avg_y1) / avg_slope)
        x_top    = int(avg_x1 + (y_top    - avg_y1) / avg_slope)

        # Clamp to frame width
        x_bottom = max(0, min(self.w, x_bottom))
        x_top    = max(0, min(self.w, x_top))

        return Lane(x_bottom, y_bottom, x_top, y_top)

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _smooth(self, side: str, lane: Optional[Lane]) -> Optional[Lane]:
        """
        Applies exponential moving average to the lane endpoints.
        If a lane is missing in this frame, the last smoothed value is reused
        (up to a limit — if missing too long it's discarded).
        """
        attr = f"_smooth_{side}"
        prev = getattr(self, attr)
        alpha = cfg.SMOOTHING_ALPHA

        if lane is not None:
            current = np.array([lane.x1, lane.y1, lane.x2, lane.y2], dtype=float)
            if prev is None:
                smoothed = current
            else:
                smoothed = alpha * current + (1 - alpha) * prev
            setattr(self, attr, smoothed)
            s = smoothed.astype(int)
            return Lane(s[0], s[1], s[2], s[3])
        else:
            # Keep last known lane (allows brief occlusion)
            if prev is not None:
                s = prev.astype(int)
                return Lane(s[0], s[1], s[2], s[3])
            return None

    # ------------------------------------------------------------------
    # Result computation
    # ------------------------------------------------------------------

    def _compute_result(
        self,
        left: Optional[Lane],
        right: Optional[Lane],
    ) -> LaneDetectionResult:
        result = LaneDetectionResult(
            left_lane      = left,
            right_lane     = right,
            frame_center_x = self.w // 2,
        )

        # Confidence: 1.0 = both lanes, 0.5 = one lane, 0.0 = none
        n_detected = (left is not None) + (right is not None)
        result.confidence = n_detected / 2.0

        if n_detected == 0:
            result.steering = "UNKNOWN — no lanes detected"
            return result

        # Lane centre = midpoint between top of left and top of right
        if left is not None and right is not None:
            lane_center_x = (left.x2 + right.x2) // 2
        elif left is not None:
            # Only left lane — estimate centre from left lane + expected lane width
            lane_center_x = left.x2 + int(0.25 * self.w)
        else:
            # Only right lane
            lane_center_x = right.x2 - int(0.25 * self.w)

        result.lane_center_x = lane_center_x
        offset = lane_center_x - result.frame_center_x
        result.offset_px = offset

        result.steering = self._steering_decision(offset)
        return result

    def _steering_decision(self, offset: int) -> str:
        """Maps pixel offset to a human-readable steering command."""
        a = abs(offset)
        direction = "RIGHT" if offset > 0 else "LEFT"

        if a < cfg.STEER_THRESHOLD_SLIGHT:
            return "STRAIGHT"
        elif a < cfg.STEER_THRESHOLD_MODERATE:
            return f"SLIGHT {direction}"
        elif a < cfg.STEER_THRESHOLD_HARD:
            return f"MODERATE {direction}"
        else:
            return f"HARD {direction}"

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _annotate(
        self,
        frame: np.ndarray,
        masked_edges: np.ndarray,
        raw_lines: Optional[np.ndarray],
        result: LaneDetectionResult,
    ) -> np.ndarray:
        """Draws lanes, centre line, offset bar, and HUD onto the frame."""

        overlay = frame.copy()

        # ── Debug: ROI outline ──────────────────────────────────────
        if cfg.DEBUG_MODE:
            cv2.polylines(frame, self._roi_polygon, True, cfg.COLOR_ROI, 2)
            if raw_lines is not None:
                for line in raw_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

        # ── Lane fill polygon ────────────────────────────────────────
        if result.left_lane and result.right_lane:
            l, r = result.left_lane, result.right_lane
            pts = np.array([
                [l.x1, l.y1], [l.x2, l.y2],
                [r.x2, r.y2], [r.x1, r.y1],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # ── Lane lines ───────────────────────────────────────────────
        if result.left_lane:
            l = result.left_lane
            cv2.line(frame, (l.x1, l.y1), (l.x2, l.y2), cfg.COLOR_LEFT_LANE, 4)

        if result.right_lane:
            r = result.right_lane
            cv2.line(frame, (r.x1, r.y1), (r.x2, r.y2), cfg.COLOR_RIGHT_LANE, 4)

        # ── Centre reference line ────────────────────────────────────
        cx = self.w // 2
        cv2.line(frame, (cx, self.h), (cx, int(cfg.ROI_TOP_Y * self.h)),
                 (255, 255, 255), 1, cv2.LINE_AA)

        # ── Lane centre marker ───────────────────────────────────────
        if result.lane_center_x is not None:
            lc = result.lane_center_x
            y_marker = int(cfg.ROI_TOP_Y * self.h) + 20
            cv2.circle(frame, (lc, y_marker), 8, cfg.COLOR_CENTER_LINE, -1)
            cv2.line(frame, (cx, y_marker), (lc, y_marker), cfg.COLOR_CENTER_LINE, 2)

        # ── Offset bar ───────────────────────────────────────────────
        self._draw_offset_bar(frame, result.offset_px)

        # ── HUD panel ───────────────────────────────────────────────
        self._draw_hud(frame, result)

        return frame

    def _draw_offset_bar(self, frame: np.ndarray, offset: int) -> None:
        """Draws a horizontal bar at the bottom showing drift direction."""
        bar_y   = self.h - 30
        bar_w   = 300
        bar_h   = 18
        cx      = self.w // 2
        bar_x   = cx - bar_w // 2

        # Background
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)

        # Fill indicating drift
        max_offset = cfg.STEER_THRESHOLD_HARD
        fill = int((offset / max_offset) * (bar_w // 2))
        fill = max(-bar_w // 2, min(bar_w // 2, fill))

        mid = bar_x + bar_w // 2
        if fill >= 0:
            color = cfg.COLOR_WARNING if fill > bar_w // 4 else (0, 200, 255)
            cv2.rectangle(frame, (mid, bar_y), (mid + fill, bar_y + bar_h), color, -1)
        else:
            color = cfg.COLOR_WARNING if fill < -bar_w // 4 else (0, 200, 255)
            cv2.rectangle(frame, (mid + fill, bar_y), (mid, bar_y + bar_h), color, -1)

        # Centre tick
        cv2.line(frame, (mid, bar_y - 4), (mid, bar_y + bar_h + 4), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    def _draw_hud(self, frame: np.ndarray, result: LaneDetectionResult) -> None:
        """Draws the info panel in the top-left corner."""
        panel_w, panel_h = 320, 160
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), cfg.COLOR_HUD_BG, -1)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

        # Steering colour
        s = result.steering
        s_color = cfg.COLOR_OK if "STRAIGHT" in s else cfg.COLOR_WARNING

        lines_text = [
            (f"FPS         : {result.fps:.1f}", (200, 200, 200)),
            (f"Offset      : {result.offset_px:+d} px",
             cfg.COLOR_WARNING if abs(result.offset_px) > cfg.STEER_THRESHOLD_SLIGHT else cfg.COLOR_OK),
            (f"Confidence  : {result.confidence:.0%}", (200, 200, 200)),
            (f"Left lane   : {'YES' if result.left_lane  else 'NO'}",
             cfg.COLOR_OK if result.left_lane  else cfg.COLOR_WARNING),
            (f"Right lane  : {'YES' if result.right_lane else 'NO'}",
             cfg.COLOR_OK if result.right_lane else cfg.COLOR_WARNING),
            (f"Steering    : {s}", s_color),
        ]

        for i, (text, color) in enumerate(lines_text):
            cv2.putText(frame, text, (20, 38 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
