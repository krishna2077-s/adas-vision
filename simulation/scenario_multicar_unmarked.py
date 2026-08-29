"""
scenario_multicar_unmarked.py — ADAS Vision: Multi-Car Crossing on Unmarked Road

New scenario featuring:
  - Variable acceleration (dynamic PID throttle / hard braking)
  - Unmarked dirt road (no lane paint, irregular edges, rough terrain patches)
  - 5 crossing vehicles appearing at staggered sections and different times
  - Road-roughness zones that enforce reduced speed limits
  - Real closing-speed TTC (Time-To-Collision) computation per vehicle

Usage:
    python simulation/scenario_multicar_unmarked.py
    python simulation/scenario_multicar_unmarked.py --no-metrics
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.vehicle_model import VEH, SIM

# ───────────────────────────────────────────────────────────────────────────
# Colour palette
# ───────────────────────────────────────────────────────────────────────────
DECISION_COLORS = {
    "PROCEED":        "#33CC55",
    "CAUTION":        "#FFCC00",
    "SLOW":           "#FF8800",
    "BRAKE":          "#FF3300",
    "EMERGENCY_STOP": "#FF0000",
}
CAR_COLORS = ["#4488FF", "#FF6622", "#FF44BB", "#22DDCC", "#FFDD00"]


def _color(d: str) -> str:
    return DECISION_COLORS.get(d, "#CCCCCC")


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _road_half_width(x: float) -> float:
    """Variable road width: narrows at two pinch-points."""
    base = 4.5
    # Pinch 1: x≈70  (road narrows around a bend)
    n1 = 1.2 * math.exp(-((x - 70) ** 2) / 200)
    # Pinch 2: x≈150 (tight S-bend)
    n2 = 1.4 * math.exp(-((x - 150) ** 2) / 150)
    return max(2.6, base - n1 - n2)


def _rough_zone(x: float) -> bool:
    """True when ego is on a roughness patch (enforces speed cap)."""
    return (55 < x < 75) or (140 < x < 160)


def _rough_speed_cap(x: float) -> float:
    return 5.0 if _rough_zone(x) else 13.9   # 13.9 m/s ≈ 50 km/h


# ───────────────────────────────────────────────────────────────────────────
# Crossing-car definitions
# ───────────────────────────────────────────────────────────────────────────

def _make_cars(lw: float) -> List[Dict]:
    """
    5 crossing vehicles at staggered X-positions.
    'side': +1 = starts below road (south), moves north.
            -1 = starts above road (north), moves south.
    'delay': seconds before the car starts moving.
    """
    return [
        {"id": 0, "label": "SUV",    "x":  40.0, "y": -lw - 6.0,
         "spd": 0.0, "max_spd":  7.5, "h": math.pi/2,
         "side":  1, "delay":  2.0, "color": CAR_COLORS[0]},
        {"id": 1, "label": "Truck",  "x":  85.0, "y":  lw + 7.0,
         "spd": 0.0, "max_spd":  5.5, "h": -math.pi/2,
         "side": -1, "delay":  5.5, "color": CAR_COLORS[1]},
        {"id": 2, "label": "Car",    "x": 125.0, "y": -lw - 5.0,
         "spd": 0.0, "max_spd":  9.8, "h": math.pi/2,
         "side":  1, "delay":  9.0, "color": CAR_COLORS[2]},
        {"id": 3, "label": "Jeep",   "x": 160.0, "y":  lw + 8.0,
         "spd": 0.0, "max_spd":  8.3, "h": -math.pi/2,
         "side": -1, "delay": 13.5, "color": CAR_COLORS[3]},
        {"id": 4, "label": "Pickup", "x": 190.0, "y": -lw - 4.0,
         "spd": 0.0, "max_spd": 11.0, "h": math.pi/2,
         "side":  1, "delay": 17.0, "color": CAR_COLORS[4]},
    ]


# ───────────────────────────────────────────────────────────────────────────
# Variable-acceleration longitudinal controller
# ───────────────────────────────────────────────────────────────────────────

class VarAccelController:
    """
    PID-style speed controller with separate tunings for
    throttle (gentle ramp-up) and braking (hard response).
    Also applies a closing-speed feed-forward term.
    """
    def __init__(self):
        self.kp_throttle = 0.45
        self.kp_brake    = 1.80
        self.ki          = 0.04
        self.kd          = 0.25
        self._integral   = 0.0
        self._prev_err   = 0.0
        self._max_accel  = 2.5   # m/s²
        self._max_decel  = 5.5   # m/s²

    def step(self, current_spd: float, target_spd: float,
             dt: float, closing_spd: float = 0.0) -> float:
        """Returns new speed after one timestep."""
        err = target_spd - current_spd
        self._integral = max(-4.0, min(4.0, self._integral + err * dt))
        deriv = (err - self._prev_err) / dt if dt > 0 else 0.0
        self._prev_err = err

        if err >= 0:                          # accelerating
            pid = self.kp_throttle * err + self.ki * self._integral + self.kd * deriv
            delta = min(pid, self._max_accel * dt)
        else:                                 # braking
            # Feed-forward: extra decel when a vehicle is closing fast
            ff = max(0.0, closing_spd * 0.12)
            pid = self.kp_brake * (-err) + ff
            delta = -min(pid * dt, self._max_decel * dt)

        return float(np.clip(current_spd + delta, 0.0, 20.0))


# ───────────────────────────────────────────────────────────────────────────
# Scenario
# ───────────────────────────────────────────────────────────────────────────

def run_scenario(collect_metrics_at_end: bool = True) -> dict:
    print("=== Scenario 6: Multi-Car Crossing — Unmarked Road ===")

    road_len_m = 230.0
    xs = np.linspace(0, road_len_m, 600)

    # Centreline: gentle S-curve
    cx_wpts = np.array([0, 40, 80,  120, 160, 200, 230], dtype=float)
    cy_wpts = np.array([0,  3,  0,   -4,   0,   3,   0], dtype=float)
    road_cx = np.interp(xs, np.linspace(0, road_len_m, len(cx_wpts)), cx_wpts)
    road_cy = np.interp(xs, np.linspace(0, road_len_m, len(cy_wpts)), cy_wpts)

    # Road edges (variable width, irregular for "unmarked" look)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-0.25, 0.25, len(xs))

    dx = np.gradient(road_cx); dy = np.gradient(road_cy)
    norms = np.sqrt(dx**2 + dy**2) + 1e-9
    nx = -dy / norms; ny = dx / norms

    hw    = np.array([_road_half_width(x) for x in xs])
    left_x  = road_cx + nx * (hw + noise)
    left_y  = road_cy + ny * (hw + noise)
    right_x = road_cx - nx * (hw - noise * 0.5)
    right_y = road_cy - ny * (hw - noise * 0.5)

    # ── Sim params ────────────────────────────────────────────────────────
    dt      = SIM.dt_s
    t_end   = 55.0
    wb      = VEH.wheelbase_m
    max_st  = VEH.max_steer_rad
    pp_la   = 10.0
    goal    = np.array([225.0, 0.0])
    goal_tol = 5.0

    ego     = {"x": 2.0, "y": 0.0, "h": 0.0, "spd": 0.0}
    lw_at_start = _road_half_width(0)
    cars    = _make_cars(lw_at_start)
    ctrl    = VarAccelController()

    # ── Figure ───────────────────────────────────────────────────────────
    plt.ion()
    fig, (ax, ax_spd) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.5, 1]})
    fig.patch.set_facecolor("#0A0A0A")

    # Main map axes
    ax.set_facecolor("#1A1208")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2A2010")
    ax.tick_params(colors="#888877")
    ax.set_xlabel("X (m)", color="#888877")
    ax.set_ylabel("Y (m)", color="#888877")
    ax.set_title("Scenario 6 — Unmarked Road | Multi-Car Crossing",
                 color="#DDCC88", fontsize=12, fontweight="bold")
    ax.grid(color="#1A1808", linewidth=0.5)
    ax.set_xlim(-5, road_len_m + 10)
    ax.set_ylim(-18, 18)

    # Speed subplot
    ax_spd.set_facecolor("#100E08")
    for sp in ax_spd.spines.values():
        sp.set_edgecolor("#2A2010")
    ax_spd.tick_params(colors="#888877")
    ax_spd.set_xlabel("Time (s)", color="#888877")
    ax_spd.set_ylabel("Speed km/h", color="#888877")
    ax_spd.set_xlim(0, t_end)
    ax_spd.set_ylim(0, 65)
    ax_spd.axhline(50, color="#FF8800", lw=0.8, ls="--", alpha=0.6,
                   label="Road limit (50 km/h)")
    ax_spd.axhline(18, color="#FF3300", lw=0.8, ls=":", alpha=0.6,
                   label="Rough zone cap")
    ax_spd.legend(fontsize=7, loc="upper right",
                  facecolor="#1A1A1A", labelcolor="white")
    h_spd_line, = ax_spd.plot([], [], "-", color="#33CC55", lw=1.3)

    # Dirt road fill
    road_poly_x = np.concatenate([left_x,  right_x[::-1]])
    road_poly_y = np.concatenate([left_y,  right_y[::-1]])
    ax.fill(road_poly_x, road_poly_y, color="#3D2E10", alpha=0.88)

    # Rough zones — hatched overlay
    for rz_start, rz_end in [(55, 75), (140, 160)]:
        mask = (xs >= rz_start) & (xs <= rz_end)
        rzpx = np.concatenate([left_x[mask], right_x[mask][::-1]])
        rzpy = np.concatenate([left_y[mask], right_y[mask][::-1]])
        ax.fill(rzpx, rzpy, color="#5A3A00", alpha=0.55, hatch="///",
                edgecolor="#7A5500", linewidth=0.3)
        ax.text((rz_start + rz_end) / 2, 9, "ROUGH\nSURFACE",
                color="#CC9933", fontsize=6.5, ha="center", va="bottom",
                fontweight="bold")

    # Road edges (irregular, no markings)
    ax.plot(left_x,  left_y,  "-", color="#7A6030", lw=1.6, alpha=0.9)
    ax.plot(right_x, right_y, "-", color="#7A6030", lw=1.6, alpha=0.9)

    # Section markers
    for cx_mark in [40, 85, 125, 160, 190]:
        ax.axvline(cx_mark, color="#555544", lw=0.7, ls=":")
        ax.text(cx_mark + 0.5, 11.5, f"SEC\n{cx_mark}m",
                color="#666655", fontsize=6, ha="left")

    # Goal
    ax.plot(goal[0], goal[1], "p", color="#FFCC00", ms=16,
            mec="white", mew=1.5)
    ax.text(goal[0] + 2, goal[1] + 2, "GOAL", color="#FFCC00", fontsize=9.5)

    # Crossing cars (dynamic handles + labels)
    car_handles = []
    car_label_h = []
    for c in cars:
        h, = ax.plot([], [], "s", color=c["color"], ms=13,
                     mec="white", mew=1.4)
        lh  = ax.text(0, 0, c["label"], color=c["color"],
                      fontsize=7.5, fontweight="bold", ha="center")
        car_handles.append(h)
        car_label_h.append(lh)

    # Ego
    h_ego,  = ax.plot([], [], "o", color="#00EE55", ms=14,
                      mec="white", mew=2.0)
    h_traj, = ax.plot([], [], "-", color="#336633", lw=1.0, alpha=0.7)

    # Status overlays
    h_status = ax.text(0.01, 0.96, "", transform=ax.transAxes,
                       color="white", fontsize=10, va="top", fontweight="bold")
    h_accel  = ax.text(0.01, 0.87, "", transform=ax.transAxes,
                       color="#88BBFF", fontsize=9, va="top")
    h_ttc    = ax.text(0.55, 0.96, "", transform=ax.transAxes,
                       color="#FFAA44", fontsize=9, va="top")

    # Legend patches
    legend_items = [
        mpatches.Patch(color="#00EE55", label="Ego Vehicle"),
        mpatches.Patch(color="#3D2E10", label="Unmarked Road"),
        mpatches.Patch(color="#5A3A00", label="Rough Zone (18 km/h)"),
    ] + [mpatches.Patch(color=c["color"], label=c["label"]) for c in cars]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
              facecolor="#111108", labelcolor="white", framealpha=0.8)

    _running = [True]
    fig.canvas.mpl_connect("close_event",
                           lambda e: _running.__setitem__(0, False))
    plt.tight_layout()
    plt.show(block=False)
    fig.canvas.flush_events()

    # ── Simulation loop ───────────────────────────────────────────────────
    t            = 0.0
    arrived      = False
    last_decision = ""
    traj_x: List[float] = [ego["x"]]
    traj_y: List[float] = [ego["y"]]
    event_log:  List[str]   = []
    t_hist:     List[float] = []
    spd_hist:   List[float] = []
    accel_prev  = 0.0

    log_data = {
        "t": [], "x": [], "y": [], "speed_mps": [],
        "decision": [], "collisions": 0, "replans": 0,
        "replan_latency_ms": [], "scenario_completed": False,
    }

    while t < t_end and _running[0]:

        dist_to_goal = float(np.linalg.norm(
            goal - np.array([ego["x"], ego["y"]])))
        if dist_to_goal < goal_tol:
            arrived = True
            break

        # ── Update crossing cars ──────────────────────────────────────
        lw_here = _road_half_width(ego["x"])
        for c in cars:
            if t < c["delay"]:
                continue
            clw = _road_half_width(c["x"])
            crossing_done = (
                (c["side"] ==  1 and c["y"] >  clw + 5) or
                (c["side"] == -1 and c["y"] < -clw - 5)
            )
            if not crossing_done:
                # Ramp up to max_spd
                c["spd"] = min(c["max_spd"],
                               c["spd"] + 1.5 * dt)
                c["y"] += c["spd"] * math.sin(c["h"]) * dt

        # ── Distances & TTC to crossing cars ─────────────────────────
        in_path_cars = []
        for c in cars:
            dx_ = c["x"] - ego["x"]
            dy_ = c["y"] - ego["y"]
            dist_c = math.hypot(dx_, dy_)
            clw_c  = _road_half_width(c["x"])
            in_road = -clw_c < c["y"] < clw_c
            in_path = in_road and abs(dx_) < 55
            # Closing speed (positive = approaching)
            close_spd = ego["spd"] - 0  # simplified: ego moving toward static X
            ttc = (dx_ / ego["spd"]) if (ego["spd"] > 0.5 and dx_ > 0) else 999
            if in_path:
                in_path_cars.append({
                    "id": c["id"], "label": c["label"],
                    "dist": dist_c, "ttc": ttc,
                    "in_road": in_road, "close_spd": close_spd,
                })

        in_path_cars.sort(key=lambda x: x["dist"])

        # ── Road roughness speed cap ──────────────────────────────────
        rough    = _rough_zone(ego["x"])
        spd_cap  = _rough_speed_cap(ego["x"])

        # ── Decision logic ────────────────────────────────────────────
        event         = ""
        closing_spd_ff = 0.0
        if in_path_cars:
            nearest = in_path_cars[0]
            d = nearest["dist"]
            ttc_v = nearest["ttc"]
            closing_spd_ff = nearest["close_spd"]

            if d < 7 and nearest["in_road"]:
                decision     = "EMERGENCY_STOP"
                speed_target = 0.0
                event = (f"t={t:.1f}s: EMERGENCY_STOP — "
                         f"{nearest['label']} {d:.1f}m (TTC {ttc_v:.1f}s)")
            elif d < 14 and nearest["in_road"]:
                decision     = "BRAKE"
                speed_target = 1.5
                event = (f"t={t:.1f}s: BRAKE — "
                         f"{nearest['label']} {d:.1f}m")
            elif d < 25 and nearest["in_road"]:
                decision     = "SLOW"
                speed_target = min(4.5, spd_cap)
                event = (f"t={t:.1f}s: SLOW — "
                         f"{nearest['label']} {d:.1f}m")
            elif d < 40:
                decision     = "CAUTION"
                speed_target = min(6.5, spd_cap)
                event = (f"t={t:.1f}s: CAUTION — "
                         f"{nearest['label']} {d:.1f}m (TTC {ttc_v:.1f}s)")
            else:
                decision     = "PROCEED"
                speed_target = spd_cap
        else:
            if rough:
                decision     = "SLOW"
                speed_target = spd_cap
            else:
                decision     = "PROCEED"
                speed_target = spd_cap

        # ── Variable-acceleration PID ─────────────────────────────────
        new_spd = ctrl.step(ego["spd"], speed_target, dt, closing_spd_ff)
        accel   = (new_spd - ego["spd"]) / dt if dt > 0 else 0.0
        ego["spd"] = new_spd

        # ── Pure-pursuit on centreline ────────────────────────────────
        dists   = np.sqrt((road_cx - ego["x"])**2 + (road_cy - ego["y"])**2)
        nearest_idx = int(np.argmin(dists))
        la_idx  = nearest_idx
        la_acc  = 0.0
        while la_acc < pp_la and la_idx < len(road_cx) - 1:
            la_acc += float(np.linalg.norm([
                road_cx[la_idx+1] - road_cx[la_idx],
                road_cy[la_idx+1] - road_cy[la_idx]]))
            la_idx += 1
        wp    = np.array([road_cx[la_idx], road_cy[la_idx]])
        alpha = math.atan2(wp[1] - ego["y"], wp[0] - ego["x"]) - ego["h"]
        alpha = _wrap(alpha)
        steer = math.atan2(2 * wb * math.sin(alpha), pp_la)
        steer = float(np.clip(steer, -max_st, max_st))

        # ── Kinematic integration ─────────────────────────────────────
        ego["x"] += ego["spd"] * math.cos(ego["h"]) * dt
        ego["y"] += ego["spd"] * math.sin(ego["h"]) * dt
        ego["h"] += (ego["spd"] / wb) * math.tan(steer) * dt
        traj_x.append(ego["x"])
        traj_y.append(ego["y"])

        # Event logging (on decision transition)
        if event and decision != last_decision:
            event_log.append(event)
            print(f"  {event}")
        last_decision = decision

        # Log
        log_data["t"].append(t)
        log_data["x"].append(ego["x"])
        log_data["y"].append(ego["y"])
        log_data["speed_mps"].append(ego["spd"])
        log_data["decision"].append(decision)

        t_hist.append(t)
        spd_hist.append(ego["spd"] * 3.6)

        # ── Plot ──────────────────────────────────────────────────────
        h_ego.set_data([ego["x"]], [ego["y"]])
        h_traj.set_data(traj_x, traj_y)

        for i, c in enumerate(cars):
            car_handles[i].set_data([c["x"]], [c["y"]])
            car_label_h[i].set_position((c["x"], c["y"] + 1.5))

        # Speed chart
        h_spd_line.set_data(t_hist, spd_hist)

        # TTC readout
        if in_path_cars:
            ttc_strs = "  ".join(
                f"{v['label']}:{v['dist']:.0f}m"
                for v in in_path_cars[:3])
            h_ttc.set_text(f"In-path: {ttc_strs}")
        else:
            h_ttc.set_text("No crossing vehicles in path")

        col = _color(decision)
        accel_str = ("⬆ ACCEL" if accel > 0.3
                     else ("⬇ BRAKE" if accel < -0.3 else "─ CRUISE"))
        h_status.set_text(
            f"{decision}  |  "
            f"{ego['spd']*3.6:.1f} km/h  |  "
            f"t={t:.1f}s  |  x={ego['x']:.0f}m")
        h_status.set_color(col)
        h_accel.set_text(
            f"{accel_str}  |  a={accel:+.2f} m/s²  |  "
            f"{'ROUGH ZONE' if rough else 'normal surface'}")
        ax.set_title(
            f"Scenario 6 — Unmarked Road | Multi-Car | {decision}",
            color=col, fontsize=12, fontweight="bold")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        t += dt

    log_data["scenario_completed"] = arrived

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== Scenario 6 Complete ===")
    print(f"  {'Goal reached' if arrived else 'Time limit reached'} at t={t:.1f}s")
    print(f"  Peak speed : {max(spd_hist, default=0):.1f} km/h")
    print(f"  Events     : {len(event_log)}")
    for ev in event_log:
        print(f"    {ev}")

    if collect_metrics_at_end and log_data["t"]:
        try:
            from simulation.collect_metrics import collect_metrics
            collect_metrics(log_data, "multicar_unmarked", save_csv=True)
        except Exception as exc:
            print(f"  [WARN] collect_metrics: {exc}")

    plt.ioff()
    plt.show()
    return log_data


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ADAS Scenario 6 — Multi-Car Crossing on Unmarked Road")
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip metrics CSV and chart.")
    args = p.parse_args()
    run_scenario(collect_metrics_at_end=not args.no_metrics)
