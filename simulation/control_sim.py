"""
control_sim.py — Module 8 (Layer 5): SIMULATED controller.

Turns the advisory plan into steering + throttle/brake for a SIMULATED ego
vehicle (kinematic model + a first-order speed response) and draws a
steering/pedal readout.

⚠️ There is NO hardware interface anywhere in this file, by design. It never
connects to a real vehicle's steering, throttle, or brakes — it drives a number
on the screen. This exists to show the shape of the control layer, not to
control anything. Advisory / research only.

Lateral      : pure-pursuit from the lane offset.
Longitudinal : PID tracking the plan's target speed.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2

import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class ControlCommand:
    steering_deg: float = 0.0
    throttle:     float = 0.0     # 0..1  (simulated)
    brake:        float = 0.0     # 0..1  (simulated)
    sim_speed_mps: float = 0.0
    reason:       str = ""


class SimController:
    """Pure-pursuit + PID over a SIMULATED ego. Never wired to a vehicle."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self.focal_px = (cfg.CALIB_PIXEL_HEIGHT * cfg.CALIB_DISTANCE_M) / cfg.CALIB_REAL_HEIGHT_M
        self.sim_speed = cfg.CTRL_SIM_INIT_MPS
        self._integral = 0.0
        self._prev_tick = cv2.getTickCount()

    def reset(self) -> None:
        self.sim_speed = cfg.CTRL_SIM_INIT_MPS
        self._integral = 0.0

    def _dt(self) -> float:
        tick = cv2.getTickCount()
        dt = (tick - self._prev_tick) / cv2.getTickFrequency()
        self._prev_tick = tick
        return max(0.02, min(0.2, dt))

    def process(self, plan, lane_result) -> ControlCommand:
        dt = self._dt()

        # ── lateral: pure pursuit from the lane offset ────────────────
        offset_px = 0.0
        if lane_result is not None and getattr(lane_result, "confidence", 0.0) > 0.0:
            offset_px = float(getattr(lane_result, "offset_px", 0.0) or 0.0)
        Ld = cfg.CTRL_LOOKAHEAD_M
        e_m = (offset_px / self.focal_px) * Ld          # lateral error at look-ahead
        alpha = math.atan2(e_m, Ld)
        steer_rad = math.atan2(2.0 * cfg.CTRL_WHEELBASE_M * math.sin(alpha), Ld)
        steer_deg = max(-cfg.CTRL_MAX_STEER_DEG,
                        min(cfg.CTRL_MAX_STEER_DEG, math.degrees(steer_rad)))

        # ── longitudinal: PID to the plan's target speed ──────────────
        target = plan.target_speed_mps if plan is not None else cfg.PLAN_CRUISE_MPS
        err = target - self.sim_speed
        self._integral = max(-5.0, min(5.0, self._integral + err * dt))
        accel = cfg.CTRL_KP * err + cfg.CTRL_KI * self._integral
        accel = max(-cfg.CTRL_MAX_BRAKE_MPS2, min(cfg.CTRL_MAX_ACCEL_MPS2, accel))
        self.sim_speed = max(0.0, self.sim_speed + accel * dt)

        throttle = max(0.0, accel) / cfg.CTRL_MAX_ACCEL_MPS2
        brake = max(0.0, -accel) / cfg.CTRL_MAX_BRAKE_MPS2
        return ControlCommand(steer_deg, throttle, brake, self.sim_speed,
                              f"target {target*3.6:.0f} km/h")

    # ------------------------------------------------------------------
    def draw(self, frame, cmd: ControlCommand):
        """Steering wheel + pedal bars readout (bottom-left)."""
        H = frame.shape[0]
        ox, oy, R = 60, H - 70, 34

        # steering wheel
        cv2.circle(frame, (ox, oy), R, (200, 200, 200), 2, cv2.LINE_AA)
        ang = math.radians(cmd.steering_deg * 2.2)         # exaggerate for visibility
        hx = int(ox + R * math.sin(ang))
        hy = int(oy - R * math.cos(ang))
        cv2.line(frame, (ox, oy), (hx, hy), (0, 220, 220), 3, cv2.LINE_AA)
        cv2.circle(frame, (ox, oy), 4, (0, 220, 220), -1)

        # throttle / brake bars
        bx = ox + R + 18
        cv2.rectangle(frame, (bx, oy - 30), (bx + 12, oy + 30), (70, 70, 70), 1)
        th = int(30 * cmd.throttle)
        cv2.rectangle(frame, (bx, oy - th), (bx + 12, oy), (0, 200, 0), -1)   # up = throttle
        br = int(30 * cmd.brake)
        cv2.rectangle(frame, (bx, oy), (bx + 12, oy + br), (0, 0, 220), -1)   # down = brake

        cv2.putText(frame, f"SIM {cmd.sim_speed_mps*3.6:4.0f} km/h  str {cmd.steering_deg:+.0f}",
                    (ox - 44, oy + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "SIMULATED CONTROL - NOT WIRED TO ANY VEHICLE",
                    (ox - 44, oy - R - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)
        return frame
