"""
perception_bev.py — Module 5 (Layer 3, "3D-ish" perception).

Turns the 2D image detections + monocular distance into a top-down, ego-frame
(bird's-eye) view. This is the shared spatial representation the fusion,
prediction, planning, and simulated-control layers all reason in.

How the projection works (flat-ground / pinhole model):
    Z (forward, m)  = the tracker's smoothed monocular distance for the object
    X (lateral, m)  = (bottom_centre_x - image_centre_x) * Z / focal_px
    focal_px        = CALIB_PIXEL_HEIGHT * CALIB_DISTANCE_M / CALIB_REAL_HEIGHT_M

Lateral velocity is estimated by differencing X per track ID across frames;
forward velocity comes from the tracker's per-object closing speed.

This is an approximation on a single camera — honest enough to *reason* about
layout, not survey-grade. Advisory / research only.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class EgoObject:
    """One object placed in the ego (bird's-eye) frame. Metres, ego at origin."""
    id:      int
    label:   str
    x_m:     float                 # lateral, + = right of the car
    z_m:     float                 # forward distance (camera estimate)
    vx_mps:  float = 0.0           # lateral velocity (estimated)
    vz_mps:  float = 0.0           # forward velocity, + = moving away
    risk:    str   = "LOW"
    in_path: bool  = False
    img_bottom: Tuple[int, int] = (0, 0)
    # Filled in by the fusion layer (Module 6), if enabled:
    radar_z_m:   Optional[float] = None
    fused_z_m:   Optional[float] = None
    fused_vz_mps: Optional[float] = None


class BEVProjector:
    """Projects confirmed tracks into the ego frame and draws the BEV panel."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self.cx = frame_width / 2.0
        self.focal_px = (cfg.CALIB_PIXEL_HEIGHT * cfg.CALIB_DISTANCE_M) / cfg.CALIB_REAL_HEIGHT_M
        self._last_x: dict = {}          # id -> (x_m, tick) for lateral-velocity estimate
        self._prev_tick = cv2.getTickCount()

    def reset(self) -> None:
        self._last_x.clear()

    def project(self, tracks: List) -> List[EgoObject]:
        """Confirmed tracks with a known distance -> ego-frame objects."""
        tick = cv2.getTickCount()
        dt = (tick - self._prev_tick) / cv2.getTickFrequency()
        self._prev_tick = tick
        dt = max(0.02, min(0.2, dt))

        ego: List[EgoObject] = []
        seen = set()
        for t in tracks:
            if not getattr(t, "confirmed", False) or t.smoothed_distance_m is None:
                continue
            z = float(t.smoothed_distance_m)
            if z <= 0:
                continue
            bx, by = t.bottom_center
            x = (bx - self.cx) * z / self.focal_px

            # lateral velocity from per-ID history
            vx = 0.0
            if t.id in self._last_x:
                px, ptick = self._last_x[t.id]
                vx = (x - px) / dt
                vx = max(-8.0, min(8.0, vx))
            self._last_x[t.id] = (x, tick)
            seen.add(t.id)

            ego.append(EgoObject(
                id=t.id, label=t.label, x_m=x, z_m=z,
                vx_mps=vx, vz_mps=-float(t.closing_speed_mps),   # closing + = approaching = Z shrinking
                risk=t.risk, in_path=t.in_path, img_bottom=(int(bx), int(by)),
            ))

        # forget stale IDs
        for k in list(self._last_x):
            if k not in seen:
                self._last_x.pop(k, None)
        return ego

    # ------------------------------------------------------------------
    # Bird's-eye panel (bottom-right corner)
    # ------------------------------------------------------------------

    def _to_panel(self, x_m: float, z_m: float) -> Tuple[int, int]:
        pw, ph = cfg.BEV_PANEL_W, cfg.BEV_PANEL_H
        px = int((x_m + cfg.BEV_HALF_WIDTH_M) / (2 * cfg.BEV_HALF_WIDTH_M) * pw)
        py = int(ph - (z_m / cfg.BEV_RANGE_M) * ph)
        return px, py

    def draw_panel(self, frame, ego_objs: List[EgoObject], predictions: Optional[dict] = None):
        """Draws the bird's-eye map with camera / radar / fused markers."""
        pw, ph = cfg.BEV_PANEL_W, cfg.BEV_PANEL_H
        panel = np.full((ph, pw, 3), cfg.COLOR_BEV_BG, np.uint8)

        # range rings + lateral centre line
        for zm in range(10, int(cfg.BEV_RANGE_M) + 1, 10):
            y = int(ph - (zm / cfg.BEV_RANGE_M) * ph)
            cv2.line(panel, (0, y), (pw, y), (55, 55, 55), 1)
            cv2.putText(panel, f"{zm}", (3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
        cv2.line(panel, (pw // 2, 0), (pw // 2, ph), (55, 55, 55), 1)

        # ego (triangle at bottom centre)
        ex, ey = pw // 2, ph - 6
        cv2.drawContours(panel, [np.array([[ex, ey - 12], [ex - 7, ey], [ex + 7, ey]], np.int32)],
                         0, cfg.COLOR_BEV_EGO, -1)

        for o in ego_objs:
            # predicted trail (ego frame), drawn faint first
            if predictions and o.id in predictions:
                for (fx, fz) in predictions[o.id]:
                    if 0 < fz < cfg.BEV_RANGE_M:
                        cv2.circle(panel, self._to_panel(fx, fz), 1, cfg.COLOR_PRED_ARROW, -1)

            cam = self._to_panel(o.x_m, o.z_m)
            if 0 <= cam[0] < pw and 0 <= cam[1] < ph:
                cv2.circle(panel, cam, 3, cfg.COLOR_BEV_CAM, -1)               # camera
            if o.radar_z_m is not None:
                rp = self._to_panel(o.x_m, o.radar_z_m)
                cv2.circle(panel, rp, 4, cfg.COLOR_BEV_RADAR, 1)              # radar ring
            zf = o.fused_z_m if o.fused_z_m is not None else o.z_m
            fp = self._to_panel(o.x_m, zf)
            cv2.circle(panel, fp, 2, cfg.COLOR_BEV_FUSED, -1)                 # fused
            cv2.putText(panel, f"{o.id}", (fp[0] + 4, fp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, cfg.COLOR_BEV_FUSED, 1)

        # title + legend
        cv2.putText(panel, "BIRD'S-EYE (sim)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # blit into the frame (bottom-right), with a border
        H, W = frame.shape[:2]
        x0, y0 = W - pw - 12, H - ph - 12
        if x0 > 0 and y0 > 0:
            frame[y0:y0 + ph, x0:x0 + pw] = panel
            cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), (90, 90, 90), 1)
        return frame
