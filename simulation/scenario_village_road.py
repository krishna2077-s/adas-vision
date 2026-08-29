"""
scenario_village_road.py — ADAS Vision Driving Scenario 1: Narrow Village Road

Python equivalent of scenario_village_road.m.

Standalone simulation (no UDP bridge required). Runs a narrow winding
village road with:
  - A pedestrian crossing mid-road ahead
  - An oncoming motorcycle
  - A parked pushcart on the left shoulder

Usage:
    python simulation/scenario_village_road.py
    python simulation/scenario_village_road.py --no-metrics
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.vehicle_model import VEH, SIM


# ===========================================================================
# Helpers
# ===========================================================================

DECISION_COLORS = {
    "PROCEED":        "#33CC55",
    "CAUTION":        "#FFCC00",
    "SLOW":           "#FF8800",
    "BRAKE":          "#FF3300",
    "EMERGENCY_STOP": "#FF0000",
}


def _color(decision: str) -> str:
    return DECISION_COLORS.get(decision, "#CCCCCC")


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ===========================================================================
# Scenario
# ===========================================================================

def run_scenario(collect_metrics_at_end: bool = True) -> dict:
    print("=== Scenario 1: Village Road ===")

    # ── Road geometry — narrow winding road (7 m wide, two 3.5 m lanes) ──
    road_width_m = 7.0
    road_len_m   = 200.0

    # Centreline waypoints (winding path)
    cx_wpts = np.array([0, 20, 45, 70, 100, 130, 160, 190, 200], dtype=float)
    cy_wpts = np.array([0,  4,  2,  -3,   0,   5,   2,  -1,   0], dtype=float)

    t_road   = np.linspace(0, 1, 500)
    road_cx  = np.interp(t_road,
                         np.linspace(0, 1, len(cx_wpts)), cx_wpts)
    road_cy  = np.interp(t_road,
                         np.linspace(0, 1, len(cy_wpts)), cy_wpts)

    # Compute road edges via normal vectors
    dx = np.gradient(road_cx)
    dy = np.gradient(road_cy)
    norms = np.sqrt(dx**2 + dy**2) + 1e-9
    nx = -dy / norms
    ny =  dx / norms
    half = road_width_m / 2
    left_x  = road_cx + nx * half
    left_y  = road_cy + ny * half
    right_x = road_cx - nx * half
    right_y = road_cy - ny * half

    # ── Actors ────────────────────────────────────────────────────────────
    dt     = SIM.dt_s
    t_end  = 25.0
    wb     = VEH.wheelbase_m
    max_st = VEH.max_steer_rad
    pp_la  = 8.0
    goal   = np.array([200.0, 0.0])
    tol    = 4.0

    # Ego
    ego  = {"x": 0.0, "y": 0.0, "h": 0.0, "spd": 8.33}

    # Actor 1: pedestrian crossing at x=80
    ped  = {"x": 80.0, "y": 0.0, "spd": 1.2, "h": math.pi / 2}

    # Actor 2: oncoming motorcycle at x=160, heading west
    moto = {"x": 160.0, "y": 1.5, "spd": 11.1, "h": math.pi}

    # Actor 3: parked pushcart (static)
    cart = {"x": 55.0, "y": 3.0}

    # ── Figure ────────────────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#121A10")
    ax.set_facecolor("#1E2A18")
    for sp in ax.spines.values():
        sp.set_edgecolor("#304028")
    ax.tick_params(colors="white")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title("Scenario 1 — Village Road  |  Initialising...",
                 color="#AAFFAA", fontsize=12)
    ax.grid(color="#263020", linewidth=0.5)

    # Road fill
    road_poly_x = np.concatenate([left_x,  right_x[::-1]])
    road_poly_y = np.concatenate([left_y,  right_y[::-1]])
    ax.fill(road_poly_x, road_poly_y, color="#3A4430", alpha=0.75)

    # Road edges
    ax.plot(left_x,  left_y,  "-", color="white", lw=1.4)
    ax.plot(right_x, right_y, "-", color="white", lw=1.4)
    ax.plot(road_cx, road_cy, "--", color="#CCBB00", lw=0.8, alpha=0.7)

    # Pedestrian crossing zone
    pcx = [ped["x"] - 1, ped["x"] + 1, ped["x"] + 1, ped["x"] - 1]
    pcy = [-half, -half, half, half]
    ax.fill(pcx, pcy, color="#EEEEEE", alpha=0.22)
    ax.text(ped["x"] - 1, half + 0.8, "PEDESTRIAN CROSSING",
            color="#CCCCAA", fontsize=7.5)

    # Static pushcart
    ax.plot(cart["x"], cart["y"], "s", color="#997733",
            ms=12, mec="white", mew=1.2)
    ax.text(cart["x"] + 1, cart["y"] + 1.5, "Pushcart",
            color="#CCAA55", fontsize=8.5)

    # Goal
    ax.plot(goal[0], goal[1], "p", color="#FFCC00", ms=16, mec="white", mew=1.5)
    ax.text(goal[0] + 2, goal[1] + 1.5, "GOAL", color="#FFCC00", fontsize=9.5)

    # Dynamic handles
    h_ego,  = ax.plot([], [], "o", color="#00DD44", ms=13,
                      mec="white", mew=1.8)
    h_ped,  = ax.plot([], [], "^", color="#CC44CC", ms=11,
                      mec="white", mew=1.5)
    h_moto, = ax.plot([], [], "d", color="#FF8833", ms=11,
                      mec="white", mew=1.5)
    h_traj, = ax.plot([], [], "--", color="#55AA55", lw=0.9)

    ax.set_xlim(-5, road_len_m + 10)
    ax.set_ylim(-half - 5, half + 5)
    fig.tight_layout()

    _running = [True]
    fig.canvas.mpl_connect("close_event", lambda e: _running.__setitem__(0, False))

    plt.show(block=False)
    fig.canvas.flush_events()

    h_status = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                       color="white", fontsize=10, va="top", fontweight="bold")

    # ── Simulation ────────────────────────────────────────────────────────
    t         = 0.0
    arrived   = False
    last_decision = ""
    traj_x: List[float] = [ego["x"]]
    traj_y: List[float] = [ego["y"]]
    event_log: List[str] = []

    log_data = {
        "t": [], "x": [], "y": [], "speed_mps": [],
        "decision": [], "collisions": 0, "replans": 0,
        "replan_latency_ms": [], "scenario_completed": False,
    }

    while t < t_end and _running[0]:

        dist_to_goal = float(np.linalg.norm(goal - np.array([ego["x"], ego["y"]])))
        if dist_to_goal < tol:
            arrived = True
            break

        # ── Pure-pursuit on road centreline ───────────────────────────
        ego_pos = np.array([ego["x"], ego["y"]])
        dists   = np.sqrt((road_cx - ego["x"])**2 + (road_cy - ego["y"])**2)
        nearest = int(np.argmin(dists))
        la_idx  = nearest
        la_acc  = 0.0
        while la_acc < pp_la and la_idx < len(road_cx) - 1:
            la_acc += float(np.linalg.norm([road_cx[la_idx+1] - road_cx[la_idx],
                                            road_cy[la_idx+1] - road_cy[la_idx]]))
            la_idx += 1
        wp    = np.array([road_cx[la_idx], road_cy[la_idx]])
        alpha = math.atan2(wp[1] - ego["y"], wp[0] - ego["x"]) - ego["h"]
        alpha = _wrap_angle(alpha)
        steer = math.atan2(2 * wb * math.sin(alpha), pp_la)
        steer = float(np.clip(steer, -max_st, max_st))

        # ── Decision ───────────────────────────────────────────────────
        dist_ped  = float(np.linalg.norm([ped["x"]  - ego["x"], ped["y"]  - ego["y"]]))
        dist_moto = float(np.linalg.norm([moto["x"] - ego["x"], moto["y"] - ego["y"]]))
        dist_cart = float(np.linalg.norm([cart["x"] - ego["x"], cart["y"] - ego["y"]]))

        event = ""
        if dist_ped < 15 and abs(ped["y"] - ego["y"]) < 3.5:
            decision      = "SLOW"
            speed_target  = 2.5
            event = f"t={t:.1f}s: SLOW — pedestrian {dist_ped:.1f}m ahead"
        elif dist_moto < 25 and abs(moto["y"] - ego["y"]) < 5:
            decision      = "CAUTION"
            speed_target  = 5.0
            event = f"t={t:.1f}s: CAUTION — oncoming motorcycle {dist_moto:.1f}m"
        elif dist_cart < 10 and abs(cart["y"] - ego["y"]) < 4:
            decision      = "CAUTION"
            speed_target  = 4.0
        else:
            decision      = "PROCEED"
            speed_target  = 8.33
        if event and decision != last_decision:
            event_log.append(event)
            print(f"  {event}")
        last_decision = decision

        # ── Speed control ──────────────────────────────────────────────
        ego["spd"] = float(np.clip(
            ego["spd"] + 0.8 * (speed_target - ego["spd"]) * dt, 0, 15))

        # ── Kinematic bicycle integration ──────────────────────────────
        ego["x"] += ego["spd"] * math.cos(ego["h"]) * dt
        ego["y"] += ego["spd"] * math.sin(ego["h"]) * dt
        ego["h"] += (ego["spd"] / wb) * math.tan(steer) * dt
        traj_x.append(ego["x"])
        traj_y.append(ego["y"])

        # Actor updates
        ped["x"]  += ped["spd"]  * math.cos(ped["h"])  * dt
        ped["y"]  += ped["spd"]  * math.sin(ped["h"])  * dt
        moto["x"] += moto["spd"] * math.cos(moto["h"]) * dt
        moto["y"] += moto["spd"] * math.sin(moto["h"]) * dt

        # Log
        log_data["t"].append(t)
        log_data["x"].append(ego["x"])
        log_data["y"].append(ego["y"])
        log_data["speed_mps"].append(ego["spd"])
        log_data["decision"].append(decision)

        # ── Plot update ────────────────────────────────────────────────
        h_ego.set_data( [ego["x"]],  [ego["y"]])
        h_ped.set_data( [ped["x"]],  [ped["y"]])
        h_moto.set_data([moto["x"]], [moto["y"]])
        h_traj.set_data(traj_x, traj_y)

        col = _color(decision)
        h_status.set_text(
            f"t={t:.1f} s  |  Decision: {decision}  |  "
            f"Speed: {ego['spd']*3.6:.1f} km/h  |  Dist: {dist_to_goal:.1f} m")
        h_status.set_color(col)
        ax.set_title(f"Scenario 1 — Village Road  |  {decision}",
                     color=col, fontsize=12)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        t += dt

    log_data["scenario_completed"] = arrived

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== Scenario 1 Complete ===")
    if arrived:
        print(f"  EGO reached goal in {t:.1f} s")
    else:
        print(f"  Scenario ended at t={t:.1f} s (goal not reached)")
    print(f"  Events logged: {len(event_log)}")
    for ev in event_log:
        print(f"    {ev}")

    if collect_metrics_at_end and log_data["t"]:
        try:
            from simulation.collect_metrics import collect_metrics
            collect_metrics(log_data, "village_road", save_csv=True)
        except Exception as exc:
            print(f"  [WARN] collect_metrics: {exc}")

    plt.ioff()
    plt.show()
    return log_data


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ADAS Scenario 1 — Village Road")
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip metrics summary and chart.")
    args = p.parse_args()
    run_scenario(collect_metrics_at_end=not args.no_metrics)
