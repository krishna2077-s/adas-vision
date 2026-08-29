"""
hmi_overlay.py — Premium In-Car HMI Dashboard Overlay (Module 15).

Replaces the plain OpenCV text HUD with a styled, color-coded automotive
instrument cluster drawn directly on the frame:

  ┌─ TOP-LEFT ────────────────────────────────────────────────────────┐
  │  [DECISION BADGE]  speed-bar  lane confidence  object count       │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ TOP-RIGHT ───────────────────────────────────────────────────────┐
  │  Nearest hazard  TTC countdown  distance  closing speed           │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ BOTTOM STRIP ────────────────────────────────────────────────────┐
  │  [R-rule badge]  reason text  FPS  frame#  model labels           │
  └───────────────────────────────────────────────────────────────────┘

All drawing is done with OpenCV — no extra dependencies.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

import config as cfg

# ── Palette (BGR) ────────────────────────────────────────────────────────────
_BG_DARK    = (15,  15,  18)
_BG_PANEL   = (22,  24,  30)
_BORDER     = (55,  60,  75)
_TEXT_DIM   = (130, 135, 145)
_TEXT_MID   = (185, 190, 200)
_TEXT_BRIGHT= (230, 235, 245)
_ACCENT_BLUE= (220, 150,  40)   # BGR — warm gold/amber

_LEVEL_BG = {
    "PROCEED":        (20,  60,  20),
    "CAUTION":        (20,  60,  70),
    "SLOW":           (10,  50,  90),
    "BRAKE":          (10,  30, 110),
    "EMERGENCY_STOP": (10,  10, 160),
}
_LEVEL_FG = {
    "PROCEED":        (60, 220,  80),
    "CAUTION":        (60, 210, 255),
    "SLOW":           (40, 165, 255),
    "BRAKE":          (30, 100, 255),
    "EMERGENCY_STOP": (50,  50, 255),
}

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


def _alpha_rect(frame, x0, y0, x1, y1, color, alpha=0.72):
    """Draw a semi-transparent filled rectangle."""
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x1 - x0, y1 - y0), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _border_rect(frame, x0, y0, x1, y1, color, thickness=1):
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)


def _put(frame, text, x, y, scale, color, thickness=1, bold=False):
    font = _FONT_BOLD if bold else _FONT
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _bar(frame, x, y, w, h, value, color, bg=(40, 42, 50)):
    """Horizontal progress bar, value in [0, 1]."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    fill = int(max(0.0, min(1.0, value)) * w)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), _BORDER, 1)


def _arc_gauge(frame, cx, cy, r, value, color, bg=(40, 42, 50), thickness=6):
    """Draws a semicircular arc gauge (bottom-half arc, 0→1 = left→right)."""
    start_angle = 180
    end_angle   = 360
    sweep = int((end_angle - start_angle) * max(0.0, min(1.0, value)))
    cv2.ellipse(frame, (cx, cy), (r, r), 0, start_angle, end_angle, bg, thickness, cv2.LINE_AA)
    if sweep > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), 0, start_angle, start_angle + sweep,
                    color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────

class HMIOverlay:
    """
    Drop-in replacement for the plain text HUD in decision_engine.draw_hud().

    Usage (in main.py, after decision = engine.process(...)):

        annotated = hmi.draw(annotated, decision, lane_result,
                             obj_result, sim_speed_mps)
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self._start = time.time()
        self._blink = 0     # blink counter for emergency

    # ──────────────────────────────────────────────────────────────────────────
    # Public draw entry
    # ──────────────────────────────────────────────────────────────────────────

    def draw(
        self,
        frame,
        decision,
        lane_result=None,
        obj_result=None,
        sim_speed_mps: Optional[float] = None,
    ):
        """
        Draw the full HMI overlay onto *frame* (in-place + returned).

        decision       : DrivingDecision from DecisionEngine
        lane_result    : LaneResult (for confidence, offset_px)
        obj_result     : ObjectResult (for tracked count, sign, light state)
        sim_speed_mps  : simulated ego speed from SimController (may be None)
        """
        self._blink = (self._blink + 1) % 30

        state = decision.longitudinal         # "PROCEED" / "CAUTION" etc.
        level = decision.longitudinal_level   # 0-4 int

        fg    = _LEVEL_FG.get(state,  _TEXT_BRIGHT)
        bg    = _LEVEL_BG.get(state,  _BG_PANEL)

        self._draw_top_bar(frame, decision, lane_result, sim_speed_mps, fg, bg, state, level)
        self._draw_hazard_panel(frame, decision, fg)
        self._draw_bottom_strip(frame, decision, fg, bg)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    # TOP BAR  (full-width, top of frame)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_top_bar(self, frame, d, lane_result, sim_speed, fg, bg, state, level):
        BAR_H = 52
        x0, y0, x1, y1 = 0, 0, self.w, BAR_H

        # Background
        _alpha_rect(frame, x0, y0, x1, y1, _BG_PANEL, alpha=0.82)
        # Bottom border line — colored by decision state
        cv2.line(frame, (0, BAR_H - 1), (self.w, BAR_H - 1), fg, 2)

        # ── Decision badge (left) ──────────────────────────────────────────
        badge_w = 210
        blink_show = not (level == 4 and self._blink < 10)
        badge_text = state if blink_show else "!!!"
        _alpha_rect(frame, 0, 0, badge_w, BAR_H, bg, alpha=0.95)
        cv2.line(frame, (badge_w, 0), (badge_w, BAR_H), fg, 1)

        # rule badge chip
        rule_chip = f" {d.rule_id} "
        _alpha_rect(frame, 8, 8, 8 + 36, BAR_H - 8, (40, 45, 55), alpha=0.9)
        _put(frame, rule_chip, 10, 36, 0.46, _ACCENT_BLUE, 1)

        # main state text
        scale = 0.72 if len(badge_text) <= 7 else 0.58
        _put(frame, badge_text, 52, 34, scale, fg, 2, bold=True)

        # ── Speed gauge (center-left) ──────────────────────────────────────
        spd_x = badge_w + 20
        spd_mps = sim_speed if sim_speed is not None else 0.0
        spd_kmh = spd_mps * 3.6
        spd_str = f"{spd_kmh:04.1f}"

        _put(frame, "SPEED",    spd_x,      18, 0.38, _TEXT_DIM, 1)
        _put(frame, spd_str,    spd_x,      41, 0.72, _TEXT_BRIGHT, 2, bold=True)
        _put(frame, "km/h",     spd_x + 65, 41, 0.36, _TEXT_DIM, 1)

        # Speed bar (0-80 km/h range)
        _bar(frame, spd_x, 44, 110, 6, spd_kmh / 80.0, fg)

        # ── Lane confidence (center) ──────────────────────────────────────
        lane_x = spd_x + 140
        lane_conf = lane_result.confidence if lane_result is not None else 0.0
        lane_mode = _lane_mode_label(lane_result)
        conf_color = (60, 200, 80) if lane_conf >= 0.8 else (60, 180, 255) if lane_conf >= 0.4 else (80, 80, 200)

        _put(frame, "LANE",       lane_x,      18, 0.38, _TEXT_DIM, 1)
        _put(frame, lane_mode,    lane_x,      41, 0.52, conf_color, 1, bold=True)
        _bar(frame, lane_x, 44, 100, 6, lane_conf, conf_color)

        # ── Object count + FPS (right side) ──────────────────────────────
        fps_x = self.w - 200
        _put(frame, "OBJECTS",  fps_x,          18, 0.38, _TEXT_DIM, 1)
        _put(frame, f"{d.tracked_count:02d}",
                                fps_x,          41, 0.72, _TEXT_BRIGHT, 2, bold=True)

        fps_x2 = fps_x + 70
        _put(frame, "FPS",      fps_x2,         18, 0.38, _TEXT_DIM, 1)
        fps_color = (60, 200, 80) if d.fps >= 12 else (60, 180, 255) if d.fps >= 6 else (80, 80, 200)
        _put(frame, f"{d.fps:04.1f}", fps_x2,   41, 0.65, fps_color, 2, bold=True)

        # DEGRADED chip (top-right corner)
        if d.degraded:
            chip_show = (self._blink % 20) < 14
            if chip_show:
                chip_x = self.w - 106
                _alpha_rect(frame, chip_x, 6, chip_x + 100, 28, (10, 10, 120), alpha=0.9)
                _border_rect(frame, chip_x, 6, chip_x + 100, 28, (40, 40, 180), 1)
                _put(frame, "DEGRADED", chip_x + 5, 22, 0.40, (80, 80, 255), 1)

    # ──────────────────────────────────────────────────────────────────────────
    # HAZARD PANEL  (top-right, below top bar)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_hazard_panel(self, frame, d, fg):
        PW, PH = 300, 118
        x0 = self.w - PW - 10
        y0 = 62
        x1 = x0 + PW
        y1 = y0 + PH

        _alpha_rect(frame, x0, y0, x1, y1, _BG_PANEL, alpha=0.82)
        _border_rect(frame, x0, y0, x1, y1, _BORDER, 1)
        # left accent bar (colored by state)
        cv2.rectangle(frame, (x0, y0), (x0 + 4, y1), fg, -1)

        lx = x0 + 14

        if d.hazard_label:
            # Header
            _put(frame, "NEAREST HAZARD", lx, y0 + 18, 0.40, _TEXT_DIM, 1)

            # Class + ID
            label_str = f"#{d.hazard_id}  {d.hazard_label.upper()}"
            _put(frame, label_str, lx, y0 + 40, 0.62, fg, 2, bold=True)

            # Distance
            dist_str = (f"{d.smoothed_distance_m:.1f} m"
                        if d.smoothed_distance_m is not None else "-- m")
            _put(frame, "DIST", lx,      y0 + 62, 0.38, _TEXT_DIM, 1)
            _put(frame, dist_str, lx + 42, y0 + 62, 0.55, _TEXT_BRIGHT, 1)

            # TTC countdown — color urgency
            if d.ttc_s is not None:
                ttc_color = ((50, 50, 255) if d.ttc_s < 2.0 else
                             (30, 130, 255) if d.ttc_s < 4.0 else
                             (60, 210, 255))
                ttc_str = f"{d.ttc_s:.1f} s"
                _put(frame, "TTC ",  lx + 120, y0 + 62, 0.38, _TEXT_DIM, 1)
                _put(frame, ttc_str, lx + 152, y0 + 62, 0.55, ttc_color, 1, bold=True)

                # TTC bar (5s max)
                urgency = 1.0 - min(1.0, d.ttc_s / 5.0)
                _bar(frame, lx, y0 + 70, PW - 24, 7, urgency, ttc_color)
            else:
                _bar(frame, lx, y0 + 70, PW - 24, 7, 0.0, _TEXT_DIM)

            # Closing speed + side
            v = d.closing_speed_mps or 0.0
            v_str = f"{'▲' if v > 0.3 else '▼'} {abs(v):.1f} m/s closing"
            side_str = f"  [{d.hazard_side}]" if d.hazard_side else ""
            _put(frame, v_str + side_str, lx, y0 + 100, 0.42, _TEXT_MID, 1)

        else:
            # Clear — show a subtle "all clear" state
            _put(frame, "NEAREST HAZARD", lx, y0 + 18, 0.40, _TEXT_DIM, 1)
            _put(frame, "ALL CLEAR",      lx, y0 + 50, 0.72, (60, 200, 80), 2, bold=True)
            _put(frame, "No obstacles detected in path",
                        lx, y0 + 80, 0.38, _TEXT_DIM, 1)
            _bar(frame, lx, y0 + 94, PW - 24, 6, 0.0, (60, 200, 80))

    # ──────────────────────────────────────────────────────────────────────────
    # BOTTOM STRIP  (full-width, bottom of frame)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_bottom_strip(self, frame, d, fg, bg):
        STRIP_H = 42
        y0 = self.h - STRIP_H
        y1 = self.h

        _alpha_rect(frame, 0, y0, self.w, y1, _BG_PANEL, alpha=0.85)
        # Top border line
        cv2.line(frame, (0, y0), (self.w, y0), fg, 2)

        # ── Left: lateral action ──────────────────────────────────────────
        lat_arrow = {"CORRECT_LEFT": "◀", "CORRECT_RIGHT": "▶"}.get(d.lateral, "▮")
        lat_color = _TEXT_DIM if d.lateral == "HOLD" else _TEXT_BRIGHT
        lat_text  = f"{lat_arrow}  {d.lateral}  {d.lateral_magnitude}"
        _put(frame, "STEER",   12, y0 + 14, 0.38, _TEXT_DIM, 1)
        _put(frame, lat_text,  12, y0 + 32, 0.48, lat_color, 1)

        # ── Center: reason text ───────────────────────────────────────────
        reason_x = 200
        reason_w = self.w - 400
        reason = _fit(d.reason, reason_w, 0.45)
        # colored left tab
        cv2.rectangle(frame, (reason_x - 4, y0 + 8),
                      (reason_x, y0 + STRIP_H - 6), fg, -1)
        _put(frame, reason, reason_x + 6, y0 + 14, 0.38, _TEXT_DIM, 1)
        # rule highlight inside reason
        if d.rule_id:
            _put(frame, d.rule_id, reason_x + 6, y0 + 30, 0.44, fg, 1, bold=True)

        # ── Right: frame# + elapsed time ─────────────────────────────────
        elapsed = time.time() - self._start
        elapsed_str = f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}"
        right_x = self.w - 180
        _put(frame, f"FRAME  {d.frame_index:06d}", right_x, y0 + 14, 0.38, _TEXT_DIM, 1)
        _put(frame, f"TIME   {elapsed_str}",       right_x, y0 + 32, 0.38, _TEXT_DIM, 1)

        # ── Brake intensity bar (bottom-left edge) ────────────────────────
        brake_color = _LEVEL_FG.get(d.longitudinal, fg)
        _bar(frame, 0, y1 - 4, int(self.w * d.brake), 4, 1.0, brake_color, bg=(0, 0, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lane_mode_label(lane_result) -> str:
    if lane_result is None:
        return "OFF"
    c = lane_result.confidence
    # Check source via attribute duck-typing (LaneResult vs RoadResult)
    source = getattr(lane_result, "source", None)
    if c == 0.0:
        return "NO LANE"
    if source == "learned":
        return f"IDD CNN  {c:.0%}"
    if source == "classical":
        return f"ROAD     {c:.0%}"
    return f"PAINT    {c:.0%}"


def _fit(text: str, max_w: int, scale: float) -> str:
    """Truncate text with ellipsis if it exceeds max_w pixels."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, scale, 1)[0][0] <= max_w:
        return text
    while len(text) > 4 and cv2.getTextSize(text + "…", font, scale, 1)[0][0] > max_w:
        text = text[:-1]
    return text + "…"
