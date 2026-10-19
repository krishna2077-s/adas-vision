"""
hackathon_cockpit.py — Hackathon-Ready ADAS Mission Control HUD & Vehicle Dynamics Telemetry.

Features:
  1. Full Real-World Vehicle Physics Integration:
     - Average C-segment sedan mass (1550 kg), tire friction (mu = 0.88), braking load transfer.
     - Dynamic normal loads: Front axle nose-dive vs rear unloading.
     - Stopping distance equation vs actual obstacle distance with safety margin bar.
     - G-Force accelerometer (Gx longitudinal braking/acceleration, Gy lateral cornering).
     - Tire adhesion friction utilization percentage.
  2. Full ADAS Decision & Control Integration:
     - Finite State Machine: ACC Cruise -> FCW Alert -> Autonomous Lane Change -> LKA Centering -> AEB Emergency.
     - Quintic polynomial planned trajectory spline drawn on BEV Radar Scope.
     - Rotating Steering Handwheel dial (-540 deg to +540 deg) with steering torque feedback.
     - Throttle & Brake pedal force bars with pulsing ABS ACTIVE indicator under hard braking.
     - Live Threat Assessment Matrix with Doppler closing speeds and graded risk levels.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.adas_controller import ADASCommand, ADASDecisionSystem, ADASState
from core.vehicle_physics import VehiclePhysicsEngine, VehicleSpecs, VEHICLE_PROFILES


class ADASCockpitHUD:
    """Master HUD rendering the unified vehicle physics, ADAS decisions, steering, and BEV radar."""

    def __init__(self, width: int = 560, height: int = 720):
        self.w = width
        self.h = height
        self.start_time = time.time()
        self.last_tick = time.time()
        self.adas = ADASDecisionSystem(cruise_speed_kmh=72.0)

    def draw_steering_wheel(self, canvas: np.ndarray, cx: int, cy: int, r: int,
                            handwheel_deg: float, torque_nm: float):
        """Draws a sleek rotating 3-spoke sports steering wheel with leather texture and center stripe."""
        # Steering pod bounding box
        cv2.rectangle(canvas, (cx - r - 12, cy - r - 12), (cx + r + 12, cy + r + 24), (16, 22, 30), -1)
        cv2.rectangle(canvas, (cx - r - 12, cy - r - 12), (cx + r + 12, cy + r + 24), (35, 48, 65), 1)

        # Outer rim
        cv2.circle(canvas, (cx, cy), r, (40, 50, 65), 5, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), r - 5, (22, 28, 38), -1)

        # Angle in radians (inverted for standard automotive steering visual)
        ang_rad = math.radians(handwheel_deg)

        # 3-Spoke design: Left (-150 deg), Right (-30 deg), Bottom (+90 deg) rotated by ang_rad
        spoke_angles = [-math.pi * 5 / 6, -math.pi / 6, math.pi / 2]
        for base_ang in spoke_angles:
            cur_a = base_ang + ang_rad
            px = int(cx + (r - 8) * math.cos(cur_a))
            py = int(cy + (r - 8) * math.sin(cur_a))
            cv2.line(canvas, (cx, cy), (px, py), (80, 95, 115), 3, cv2.LINE_AA)

        # Center Hub & ADAS Badge
        cv2.circle(canvas, (cx, cy), 16, (32, 40, 52), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 16, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 4, (0, 230, 255), -1, cv2.LINE_AA)

        # 12 o'clock center alignment stripe on rim (rotates with steering)
        top_ang = -math.pi / 2 + ang_rad
        tx1 = int(cx + (r - 7) * math.cos(top_ang))
        ty1 = int(cy + (r - 7) * math.sin(top_ang))
        tx2 = int(cx + (r + 1) * math.cos(top_ang))
        ty2 = int(cy + (r + 1) * math.sin(top_ang))
        cv2.line(canvas, (tx1, ty1), (tx2, ty2), (0, 100, 255), 4, cv2.LINE_AA)

        # Steering angle & torque text readout
        steer_dir = "LEFT" if handwheel_deg < -1.0 else ("RIGHT" if handwheel_deg > 1.0 else "CTR")
        steer_col = (0, 214, 255) if abs(handwheel_deg) > 5.0 else (0, 230, 118)
        cv2.putText(canvas, f"STEER: {abs(handwheel_deg):4.1f}° {steer_dir}",
                    (cx - r - 6, cy + r + 8), cv2.FONT_HERSHEY_DUPLEX, 0.34, steer_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"TORQUE: {abs(torque_nm):3.1f} Nm",
                    (cx - r - 6, cy + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (160, 175, 195), 1, cv2.LINE_AA)

    def draw_speedometer(self, canvas: np.ndarray, cx: int, cy: int, r: int, speed: float, target_spd: float):
        """Draws the circular dial analog + digital speedometer."""
        cv2.rectangle(canvas, (cx - r - 12, cy - r - 12), (cx + r + 12, cy + r + 24), (16, 22, 30), -1)
        cv2.rectangle(canvas, (cx - r - 12, cy - r - 12), (cx + r + 12, cy + r + 24), (35, 48, 65), 1)

        cv2.circle(canvas, (cx, cy), r, (38, 48, 60), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), r - 6, (20, 26, 36), -1)

        start_angle = 140.0
        sweep_angle = 260.0

        for kmh in range(0, 141, 10):
            frac = kmh / 140.0
            ang = math.radians(start_angle + frac * sweep_angle)
            p1 = (int(cx + (r - 14) * math.cos(ang)), int(cy + (r - 14) * math.sin(ang)))
            p2 = (int(cx + (r - 6) * math.cos(ang)), int(cy + (r - 6) * math.sin(ang)))
            col = (0, 230, 118) if kmh <= 60 else ((0, 214, 255) if kmh <= 90 else (50, 70, 255))
            cv2.line(canvas, p1, p2, col, 2 if kmh % 20 == 0 else 1, cv2.LINE_AA)

            if kmh % 20 == 0:
                pt = (int(cx + (r - 24) * math.cos(ang) - 7), int(cy + (r - 24) * math.sin(ang) + 4))
                cv2.putText(canvas, str(kmh), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 175, 190), 1, cv2.LINE_AA)

        # Speed needle
        cur_frac = max(0.0, min(1.0, speed / 140.0))
        needle_ang = math.radians(start_angle + cur_frac * sweep_angle)
        nx = int(cx + (r - 14) * math.cos(needle_ang))
        ny = int(cy + (r - 14) * math.sin(needle_ang))
        cv2.line(canvas, (cx, cy), (nx, ny), (0, 80, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, (0, 160, 255), -1, cv2.LINE_AA)

        # Digital Readout
        cv2.putText(canvas, f"{int(speed)}", (cx - 20, cy + 24), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "KM/H", (cx - 15, cy + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 230, 255), 1, cv2.LINE_AA)

        # Cruise and Gear
        cv2.putText(canvas, f"SET: {int(target_spd)}", (cx - r - 4, cy + r + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 170, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, "GEAR: D4", (cx + 8, cy + r + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 118), 1, cv2.LINE_AA)

    def draw_pedals_and_gmeter(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                               throttle_pct: float, brake_pct: float, ax_g: float, ay_g: float):
        """Draws Throttle/Brake vertical pedal bars and 2D G-Force accelerometer ball meter."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (16, 22, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 48, 65), 1)

        # Title
        cv2.putText(canvas, "PEDALS & G-METER", (x + 8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 230, 255), 1, cv2.LINE_AA)

        # Throttle Bar
        bar_w = 12
        bar_h = 75
        tx = x + 16
        by = y + 25
        cv2.rectangle(canvas, (tx, by), (tx + bar_w, by + bar_h), (25, 32, 42), -1)
        th_fill = int((throttle_pct / 100.0) * bar_h)
        if th_fill > 0:
            cv2.rectangle(canvas, (tx, by + bar_h - th_fill), (tx + bar_w, by + bar_h), (0, 230, 118), -1)
        cv2.rectangle(canvas, (tx, by), (tx + bar_w, by + bar_h), (60, 80, 100), 1)
        cv2.putText(canvas, "GAS", (tx - 3, by + bar_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 230, 118), 1)

        # Brake Bar
        bx = x + 40
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (25, 32, 42), -1)
        brk_fill = int((brake_pct / 100.0) * bar_h)
        if brk_fill > 0:
            brk_col = (0, 0, 255) if brake_pct > 70.0 else (0, 140, 255)
            cv2.rectangle(canvas, (bx, by + bar_h - brk_fill), (bx + bar_w, by + bar_h), brk_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (60, 80, 100), 1)
        cv2.putText(canvas, "BRK", (bx - 3, by + bar_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 80, 255), 1)

        # ABS Active blinking alert
        if brake_pct > 65.0:
            abs_col = (0, 0, 255) if int(time.time() * 8) % 2 == 0 else (0, 200, 255)
            cv2.putText(canvas, "ABS ACTIVE", (x + 8, by + bar_h + 24), cv2.FONT_HERSHEY_DUPLEX, 0.32, abs_col, 1, cv2.LINE_AA)

        # 2D G-Force Accelerometer (Right side of box)
        gx_cx = x + 115
        gx_cy = y + 62
        g_rad = 34
        cv2.circle(canvas, (gx_cx, gx_cy), g_rad, (30, 38, 50), -1)
        cv2.circle(canvas, (gx_cx, gx_cy), int(g_rad * 0.4), (45, 58, 75), 1)  # 0.35g comfort ring
        cv2.circle(canvas, (gx_cx, gx_cy), g_rad, (60, 78, 100), 1)            # 0.88g tire friction ring
        cv2.line(canvas, (gx_cx - g_rad, gx_cy), (gx_cx + g_rad, gx_cy), (40, 52, 68), 1)
        cv2.line(canvas, (gx_cx, gx_cy - g_rad), (gx_cx, gx_cy + g_rad), (40, 52, 68), 1)

        # G-ball position: ax is vertical (-ax is braking = forward ball pitch), ay is lateral
        # 1.0g = g_rad pixels
        ball_x = int(gx_cx + max(-g_rad, min(g_rad, ay_g * g_rad)))
        ball_y = int(gx_cy - max(-g_rad, min(g_rad, ax_g * g_rad)))
        ball_col = (0, 0, 255) if abs(ax_g) > 0.6 or abs(ay_g) > 0.5 else (0, 230, 118)
        cv2.circle(canvas, (ball_x, ball_y), 5, ball_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (ball_x, ball_y), 6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Gx: {ax_g:+4.2f}g", (gx_cx - 28, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 195, 215), 1)
        cv2.putText(canvas, f"Gy: {ay_g:+4.2f}g", (gx_cx + 12, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 195, 215), 1)

    def draw_bev_radar(self, canvas: np.ndarray, rx: int, ry: int, rw: int, rh: int,
                       cmd: ADASCommand, phys: dict, scene: dict):
        """Draws BEV Radar Scope with lane corridors, surrounding traffic blips, and planned trajectory spline."""
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (14, 20, 28), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (40, 55, 75), 1)

        cv2.putText(canvas, "BIRD'S-EYE VIEW (BEV) RADAR SCOPE (60m)", (rx + 12, ry + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (rx, ry + 22), (rx + rw, ry + 22), (35, 48, 65), 1)

        # Ego anchor at bottom center
        ego_x = rx + rw // 2
        ego_y = ry + rh - 24

        lane_w_px = 38
        scale_px_per_m = (rh - 58) / 60.0  # 60m forward

        # Draw 5 lanes (-1, -2, -3, -4, -5)
        # Ego lateral offset shifts road visually so ego stays centered or moves across lanes
        lat_shift_px = int((cmd.current_lane_offset_m / 3.5) * lane_w_px)

        for lane_offset in [-2, -1, 0, 1, 2]:
            lx = ego_x + int(lane_offset * lane_w_px) - lat_shift_px
            if rx + 6 < lx < rx + rw - 6:
                style_color = (65, 80, 100) if lane_offset in (-2, 2) else (32, 44, 58)
                cv2.line(canvas, (lx, ry + 24), (lx, ry + rh - 4), style_color, 1)

        # Range rings: 20m, 40m, 60m
        for m in [20, 40, 60]:
            dist_px = int(m * scale_px_per_m)
            ring_y = ego_y - dist_px
            cv2.line(canvas, (rx + 10, ring_y), (rx + rw - 10, ring_y), (28, 38, 50), 1)
            cv2.putText(canvas, f"{m}m", (rx + 12, ring_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 110, 130), 1)

        # Radar Sweep Cone
        sweep_ang = (scene["sim_t"] * 3.5) % (math.pi / 3.0) - (math.pi / 6.0)
        end_x = int(ego_x + math.sin(sweep_ang) * (rh - 35))
        end_y = int(ego_y - math.cos(sweep_ang) * (rh - 35))
        cv2.line(canvas, (ego_x, ego_y), (end_x, end_y), (0, 180, 220), 1, cv2.LINE_AA)

        # Draw Planned Trajectory Spline (Quintic polynomial path during lane change)
        if cmd.state == ADASState.AUTONOMOUS_LANE_CHANGE and len(cmd.planned_path) > 1:
            spline_pts = []
            for dx_m, dy_m in cmd.planned_path:
                px = int(ego_x + ((dy_m - cmd.current_lane_offset_m) / 3.5) * lane_w_px)
                py = int(ego_y - dx_m * scale_px_per_m)
                if rx + 10 < px < rx + rw - 10 and ry + 25 < py < ry + rh:
                    spline_pts.append((px, py))
            if len(spline_pts) > 1:
                for i in range(len(spline_pts) - 1):
                    cv2.line(canvas, spline_pts[i], spline_pts[i + 1], (0, 230, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "PLANNED TRAJECTORY SPLINE", (spline_pts[-1][0] - 50, spline_pts[-1][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 230, 255), 1, cv2.LINE_AA)

        # Draw Ego Vehicle (Green Rectangle)
        cv2.rectangle(canvas, (ego_x - 8, ego_y - 15), (ego_x + 8, ego_y + 4), (0, 230, 118), -1)
        cv2.putText(canvas, "EGO", (ego_x - 12, ego_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 230, 118), 1)

        # Plot surrounding obstacle blips
        def plot_blip(name: str, lat_m: float, dist_m: float, color: tuple, is_hazard: bool = False):
            if 0.0 < dist_m < 65.0:
                px = int(ego_x + ((lat_m - cmd.current_lane_offset_m) / 3.5) * lane_w_px)
                py = int(ego_y - dist_m * scale_px_per_m)
                if rx + 8 < px < rx + rw - 8 and ry + 25 < py < ry + rh:
                    cv2.rectangle(canvas, (px - 6, py - 9), (px + 6, py + 9), color, -1)
                    if is_hazard:
                        cv2.circle(canvas, (px, py), 13, (0, 80, 255), 1, cv2.LINE_AA)
                    cv2.putText(canvas, name, (px + 9, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1, cv2.LINE_AA)

        # Lead Car
        plot_blip(f"LeadCar ({scene['lead_dist']:.0f}m)", 0.0, scene["lead_dist"], (255, 140, 0))
        # Overtaker in Lane -2 (-3.5m)
        if scene["overtaker_gap"] > 0:
            plot_blip("Overtaker", -3.5, scene["overtaker_gap"], (0, 80, 255))
        # Heavy Truck in Lane -4 (+3.5m)
        plot_blip("Truck", 3.5, max(15.0, 52.0 - scene["sim_t"] * 1.0), (0, 220, 255))
        # Stray Cattle Crossing
        if scene["cattle_dist"] < 65.0:
            plot_blip(f"CATTLE ({scene['cattle_dist']:.0f}m)", scene["cattle_lat_m"], scene["cattle_dist"], (0, 0, 255), is_hazard=True)

    def draw_physics_telemetry(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                               cmd: ADASCommand, phys: dict, scene: dict):
        """Draws the Real-World Physics panel: weather surface friction, vehicle mass, stopping distance, load transfer."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (16, 22, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 55, 75), 1)

        cv2.putText(canvas, "VEHICLE DYNAMICS & FRICTION", (x + 8, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (x, y + 20), (x + w, y + 20), (40, 55, 75), 1)

        # 1. Active Weather & Vehicle Profile Badges
        w_type = scene.get("weather_type", "dry").upper().replace("_", " ")
        v_type = scene.get("vehicle_type", "hector_ev").upper().replace("_", " ")
        mu = scene.get("mu_peak", 0.88)
        mass = scene.get("vehicle_mass_kg", 1940.0)
        fcw_sens = scene.get("fcw_sensitivity", "comfort").upper()
        tja_on = "ON" if scene.get("tja_enabled", True) else "OFF"
        lka_mode = "GENTLE" if scene.get("lka_intervention", "gentle_nudge") == "gentle_nudge" else "FIRM"

        w_col = (0, 230, 118) if mu >= 0.75 else ((0, 214, 255) if mu >= 0.50 else (0, 90, 255))
        cv2.putText(canvas, f"ENV: {w_type} (mu={mu:.2f})", (x + 8, y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.28, w_col, 1)

        # Vehicle label
        if "HECTOR" in v_type:
            v_label = f"MG HECTOR TOMAHAWK ({int(mass)}kg)"
        else:
            v_label = f"{v_type} ({int(mass)}kg)"
        cv2.putText(canvas, f"CAR: {v_label}", (x + 8, y + 43), cv2.FONT_HERSHEY_DUPLEX, 0.26, (220, 230, 245), 1)
        ihc_txt = "HIGH" if "HIGH" in cmd.ihc_mode else "LOW"
        cv2.putText(canvas, f"ADAS: FCW:{fcw_sens} | TJA:{tja_on} | LKA:{lka_mode} | IHC:{ihc_txt}", (x + 8, y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.23, (0, 215, 255), 1)

        # 2. Stopping Distance vs Threat Distance Bar
        stop_d = phys["stopping_dist_m"]
        actual_d = scene["cattle_dist"] if scene["cattle_dist"] > 0 else scene["lead_dist"]
        safety_margin = actual_d - stop_d

        cv2.putText(canvas, f"StopDist: {stop_d:4.1f}m | Threat: {actual_d:4.1f}m",
                    (x + 8, y + 69), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (190, 205, 225), 1)

        # Margin Bar
        margin_bar_w = w - 16
        cv2.rectangle(canvas, (x + 8, y + 74), (x + 8 + margin_bar_w, y + 81), (28, 36, 48), -1)
        margin_frac = max(0.0, min(1.0, stop_d / max(1.0, actual_d)))
        fill_w = int(margin_frac * margin_bar_w)
        bar_col = (0, 0, 255) if margin_frac > 0.8 else ((0, 140, 255) if margin_frac > 0.5 else (0, 230, 118))
        cv2.rectangle(canvas, (x + 8, y + 74), (x + 8 + fill_w, y + 81), bar_col, -1)
        cv2.rectangle(canvas, (x + 8, y + 74), (x + 8 + margin_bar_w, y + 81), (50, 70, 90), 1)

        margin_txt = f"SAFETY MARGIN: {safety_margin:+4.1f} m"
        cv2.putText(canvas, margin_txt, (x + 8, y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.29, bar_col, 1)

        # 3. Dynamic Normal Axle Load Transfer & Nose Dive
        fz_f_kg = phys["fz_front_n"] / 9.81
        fz_r_kg = phys["fz_rear_n"] / 9.81
        cv2.putText(canvas, f"Front Load: {fz_f_kg:4.0f} kg ({phys['pitch_deg']:+3.1f}° dive)",
                    (x + 8, y + 109), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (180, 200, 220), 1)
        cv2.putText(canvas, f"Rear Load : {fz_r_kg:4.0f} kg",
                    (x + 8, y + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (180, 200, 220), 1)

        # 4. Tire Friction Utilization
        fric_pct = phys["friction_util_pct"]
        fric_col = (0, 0, 255) if fric_pct > 80.0 else ((0, 140, 255) if fric_pct > 50.0 else (0, 230, 118))
        cv2.putText(canvas, f"Tire Grip Util: {fric_pct:4.1f}%",
                    (x + 8, y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.27, fric_col, 1)

        # 5. Hotkey prompt
        cv2.putText(canvas, "[W]Weather [V]Car [A]FCW [T]TJA [L]LKA", (x + 8, y + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (140, 155, 175), 1)

    def draw_threat_matrix(self, canvas: np.ndarray, tx: int, ty: int, tw: int, th: int, scene: dict):
        """Draws the obstacle threat assessment table with dynamic Doppler closing velocities."""
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (16, 22, 30), -1)
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (40, 55, 75), 1)

        cv2.putText(canvas, "LIVE THREAT ASSESSMENT MATRIX", (tx + 8, ty + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (tx, ty + 22), (tx + tw, ty + 22), (50, 68, 90), 1)

        headers = ["ID", "OBJECT", "DISTANCE", "REL SPD", "RISK"]
        col_x = [tx + 8, tx + 32, tx + 130, tx + 195, tx + 248]
        for h_text, x_pos in zip(headers, col_x):
            cv2.putText(canvas, h_text, (x_pos, ty + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 170, 190), 1)
        cv2.line(canvas, (tx, ty + 38), (tx + tw, ty + 38), (40, 55, 75), 1)

        c_dist = scene["cattle_dist"]
        c_risk = "CRITICAL" if c_dist < 35.0 else ("WARNING" if c_dist < 70.0 else "TRACKING")
        c_col = (0, 0, 255) if c_risk == "CRITICAL" else ((0, 140, 255) if c_risk == "WARNING" else (140, 160, 180))

        lead_risk = "WARNING" if scene["lead_dist"] < 60.0 else "SAFE"
        lead_col = (0, 140, 255) if lead_risk == "WARNING" else (0, 230, 118)

        rows = [
            ("01", "Stray Cattle", f"{c_dist:4.1f}m", "-0.8m/s", c_risk, c_col),
            ("02", "Lead Vehicle", f"{scene['lead_dist']:4.1f}m", f"{scene['lead_rel_speed']:+4.1f}m/s", lead_risk, lead_col),
            ("03", "Red Overtaker", f"{abs(scene['overtaker_gap']):4.1f}m", "+4.0m/s", "LOW", (0, 230, 118)),
            ("04", "Yellow Truck", "48.2m", "-3.5m/s", "LOW", (0, 230, 118)),
            ("05", "Police Nakabandi", "180.0m", "0.0m/s", "ADVISORY", (170, 190, 210)),
        ]

        row_y = ty + 52
        for r_id, r_cls, r_dst, r_spd, r_rsk, r_col in rows:
            cv2.putText(canvas, r_id, (col_x[0], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (180, 200, 220), 1)
            cv2.putText(canvas, r_cls, (col_x[1], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (230, 240, 255), 1)
            cv2.putText(canvas, r_dst, (col_x[2], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255, 255, 255), 1)
            cv2.putText(canvas, r_spd, (col_x[3], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (170, 190, 210), 1)
            cv2.putText(canvas, r_rsk, (col_x[4], row_y), cv2.FONT_HERSHEY_DUPLEX, 0.27, r_col, 1)
            row_y += 14

    def render_frame(self, sim_t: Optional[float] = None) -> np.ndarray:
        now = time.time()
        if sim_t is None:
            sim_t = now - self.start_time
        dt = max(0.005, min(0.1, now - self.last_tick))
        self.last_tick = now

        cmd, phys = self.adas.update(sim_t=sim_t, dt=dt)
        scene = self.adas.evaluate_scene(sim_t)

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = (12, 16, 22)  # Graphite dark theme

        # 1. Top Header Banner
        cv2.rectangle(canvas, (0, 0), (self.w, 32), (18, 24, 34), -1)
        cv2.line(canvas, (0, 32), (self.w, 32), (0, 180, 230), 2)
        cv2.putText(canvas, "ADAS MISSION CONTROL & DYNAMICS HUD", (12, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"T: {sim_t:4.1f}s", (self.w - 74, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)

        # 2. Cluster Top Row: Steering Wheel (Left) | Speedometer (Center) | Pedals & GMeter (Right)
        # Steering Wheel at (95, 105), r=55
        self.draw_steering_wheel(canvas, cx=95, cy=105, r=54,
                                 handwheel_deg=cmd.steering_wheel_deg,
                                 torque_nm=phys["steering_torque_nm"])

        # Speedometer at (280, 105), r=60
        self.draw_speedometer(canvas, cx=280, cy=105, r=60,
                              speed=phys["speed_kmh"], target_spd=cmd.target_speed_kmh)

        # Pedals & G-Meter at (380, 42), w=166, h=136
        self.draw_pedals_and_gmeter(canvas, x=380, y=42, w=166, h=136,
                                    throttle_pct=cmd.throttle_cmd * 100.0,
                                    brake_pct=cmd.brake_cmd * 100.0,
                                    ax_g=cmd.longitudinal_g, ay_g=cmd.lateral_g)

        # 3. Decision Engine Status Banner
        cv2.rectangle(canvas, (14, 186), (self.w - 14, 226), (20, 28, 38), -1)
        cv2.rectangle(canvas, (14, 186), (self.w - 14, 226), cmd.status_color, 2)
        cv2.putText(canvas, cmd.state_desc, (22, 206),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, cmd.status_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"TTC: {cmd.ttc_s:3.1f}s", (self.w - 95, 206),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, cmd.decision_reason, (22, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (190, 205, 225), 1, cv2.LINE_AA)

        # 4. Bird's-Eye View (BEV) Radar Scope
        self.draw_bev_radar(canvas, rx=14, ry=234, rw=self.w - 28, rh=285,
                            cmd=cmd, phys=phys, scene=scene)

        # 5. Bottom Split: Physics Telemetry (Left) | Threat Matrix (Right)
        split_w = (self.w - 34) // 2
        self.draw_physics_telemetry(canvas, x=14, y=528, w=split_w, h=182,
                                    cmd=cmd, phys=phys, scene=scene)

        self.draw_threat_matrix(canvas, tx=18 + split_w, ty=528, tw=split_w, th=182,
                                scene=scene)

        return canvas


def run_cockpit(weather: str = "dry", vehicle: str = "hector_ev",
                weight: Optional[float] = None, friction: Optional[float] = None,
                fcw_sens: str = "comfort", tja: bool = True):
    """
    Runs the ADAS Cockpit HUD standalone or side-by-side with esmini 3D simulator.
    Interactive Keys:
        [W] / [w] : Cycle Weather Profile (Dry -> Monsoon -> Heavy Downpour -> Gravel)
        [V] / [v] : Cycle Vehicle Profile (MG Hector EV -> Hector PHEV -> Sedan -> SUV)
        [A] / [a] : Cycle FCW Warning Sensitivity (Comfort/Dialed-Back -> Normal -> Aggressive)
        [T] / [t] : Toggle Traffic Jam Assist (TJA) with Curve/Bend Control
        [L] / [l] : Toggle LKA Intervention Level (Gentle Tactile Nudge <-> Firm)
        [R] / [r] : Reset Simulation timeline to t=0.0s
        [Q] / [ESC] : Exit cleanly
    """
    cockpit = ADASCockpitHUD(width=560, height=720)
    if vehicle:
        cockpit.adas.set_vehicle(vehicle, weight)
    if weather:
        cockpit.adas.set_weather(weather, friction)
    cockpit.adas.fcw_sensitivity = fcw_sens
    cockpit.adas.tja_enabled = tja

    win_name = "ADAS Mission Control & Vehicle Dynamics Telemetry"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_name, 980, 30)

    car_name = VEHICLE_PROFILES.get(cockpit.adas.physics.specs.vehicle_type, {}).get("name", vehicle.upper())
    print(f"\n{'=' * 68}")
    print(f"  ADAS Cockpit HUD Active — Level-2 MG Pilot Suite")
    print(f"  Vehicle : {car_name} ({int(cockpit.adas.physics.specs.total_mass_kg)} kg)")
    print(f"  Weather : {cockpit.adas.physics.specs.weather_type.upper()} (mu = {cockpit.adas.physics.specs.mu_peak:.2f})")
    print(f"  ADAS    : FCW: {cockpit.adas.fcw_sensitivity.upper()} | TJA: {'ON' if cockpit.adas.tja_enabled else 'OFF'} | LKA: {cockpit.adas.lka_intervention.upper()}")
    print(f"  Hotkeys : [W] Weather | [V] Car | [A] FCW Sens | [T] TJA | [L] LKA | [R] Reset | [Q] Quit")
    print(f"{'=' * 68}\n")

    while True:
        frame = cockpit.render_frame()
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(25) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('w'), ord('W')):
            new_w = cockpit.adas.physics.cycle_weather_profile()
            print(f"[*] Weather changed -> {new_w.upper()} (mu={cockpit.adas.physics.specs.mu_peak:.2f}, Crr={cockpit.adas.physics.specs.Crr:.3f})")
        elif key in (ord('v'), ord('V')):
            new_v = cockpit.adas.physics.cycle_vehicle_profile()
            v_name = VEHICLE_PROFILES.get(new_v, {}).get("name", new_v.upper())
            print(f"[*] Vehicle changed -> {v_name} ({int(cockpit.adas.physics.specs.total_mass_kg)} kg, L={cockpit.adas.physics.specs.wheelbase_m:.2f}m)")
        elif key in (ord('a'), ord('A')):
            new_sens = cockpit.adas.cycle_fcw_sensitivity()
            print(f"[*] FCW Warning Sensitivity -> {new_sens.upper()} (TTC thresh adapted)")
        elif key in (ord('t'), ord('T')):
            tja_st = cockpit.adas.toggle_tja()
            print(f"[*] Traffic Jam Assist (Curve/Bend Control) -> {'ENABLED' if tja_st else 'DISABLED'}")
        elif key in (ord('l'), ord('L')):
            lka_st = cockpit.adas.cycle_lka_intervention()
            print(f"[*] Lane Keep Intervention Level -> {lka_st.upper()}")
        elif key in (ord('r'), ord('R')):
            cur_veh = cockpit.adas.physics.specs.vehicle_type
            cur_wth = cockpit.adas.physics.specs.weather_type
            cur_mass = cockpit.adas.physics.specs.total_mass_kg
            cur_mu = cockpit.adas.physics.specs.mu_peak
            cur_sens = cockpit.adas.fcw_sensitivity
            cur_tja = cockpit.adas.tja_enabled
            cur_lka = cockpit.adas.lka_intervention
            cockpit.adas = ADASDecisionSystem(
                cruise_speed_kmh=72.0,
                fcw_sensitivity=cur_sens,
                tja_enabled=cur_tja,
                lka_intervention=cur_lka
            )
            cockpit.adas.set_vehicle(cur_veh, cur_mass)
            cockpit.adas.set_weather(cur_wth, cur_mu)
            cockpit.start_time = time.time()
            cockpit.last_tick = time.time()
            print("[*] Simulation reset to t=0.0s")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS Mission Control Cockpit HUD")
    parser.add_argument("--weather", default="dry",
                        choices=["dry", "wet", "heavy_rain", "gravel"],
                        help="Weather & road friction profile")
    parser.add_argument("--vehicle", default="hector_ev",
                        choices=["hector_ev", "hector_phev", "sedan", "hatchback", "compact_suv", "heavy_suv"],
                        help="Vehicle class preset (Default: hector_ev)")
    parser.add_argument("--weight", type=float, default=None,
                        help="Custom total vehicle mass (kg)")
    parser.add_argument("--friction", type=float, default=None,
                        help="Custom road surface peak friction coefficient mu (e.g. 0.88, 0.55)")
    parser.add_argument("--fcw", default="comfort",
                        choices=["comfort", "normal", "aggressive"],
                        help="Driver-adjustable FCW sensitivity mode (Default: comfort)")
    parser.add_argument("--no-tja", action="store_true",
                        help="Disable Traffic Jam Assist (TJA)")
    args = parser.parse_args()

    run_cockpit(weather=args.weather, vehicle=args.vehicle,
                weight=args.weight, friction=args.friction,
                fcw_sens=args.fcw, tja=not args.no_tja)
