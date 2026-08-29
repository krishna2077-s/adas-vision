"""
scenario_urban_intersection.py — ADAS Vision Driving Scenario 2: Urban Intersection

Python equivalent of scenario_urban_intersection.m.

Standalone simulation (no UDP bridge required). Creates a 4-way junction with:
  - Cross-traffic cars from the left and right
  - A rickshaw entering from the right
  - Pedestrians crossing at the zebra crossing
  - A cyclist on the far side

Usage:
    python simulation/scenario_urban_intersection.py
    python simulation/scenario_urban_intersection.py --no-metrics
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.vehicle_model import VEH, SIM


DECISION_COLORS = {
    "PROCEED":        "#33CC55",
    "CAUTION":        "#FFCC00",
    "SLOW":           "#FF8800",
    "BRAKE":          "#FF3300",
    "EMERGENCY_STOP": "#FF0000",
}


def _color(d: str) -> str:
    return DECISION_COLORS.get(d, "#CCCCCC")


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ===========================================================================
# Scenario
# ===========================================================================

def run_scenario(collect_metrics_at_end: bool = True) -> dict:
    print("=== Scenario 2: Urban Intersection ===")

    lw = 3.0    # half lane width (m)
    rl = 80.0   # road arm length (m)

    dt    = SIM.dt_s
    t_end = 30.0
    wb    = VEH.wheelbase_m
    max_st = VEH.max_steer_rad
    pp_la  = 8.0
    goal  = np.array([0.0, rl])
    tol   = 5.0

    # ── Actors ────────────────────────────────────────────────────────────
    ego  = {"x": 0.0,      "y": -rl,       "h": math.pi/2, "spd": 8.33}
    car1 = {"x": -rl,      "y":  lw/2,     "spd": 11.1,    "h": 0.0}
    car2 = {"x":  rl,      "y": -lw/2,     "spd":  9.7,    "h": math.pi}
    rick = {"x":  rl*0.6,  "y":  lw/2+1.5, "spd":  5.5,    "h": math.pi}
    ped1 = {"x": -lw,      "y": 0.0,       "spd":  1.4,    "h": 0.0}
    ped2 = {"x":  lw*2,    "y": 0.0,       "spd":  1.1,    "h": math.pi}
    cyc  = {"x":  lw/2,    "y":  rl*0.3,   "spd":  4.2,    "h": math.pi/2}

    # ── Figure ────────────────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_facecolor("#0F0F18")
    ax.set_facecolor("#1A1A26")
    for sp in ax.spines.values():
        sp.set_edgecolor("#303040")
    ax.tick_params(colors="white")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title("Scenario 2 — Urban Intersection", color="#AABBFF", fontsize=12)
    ax.grid(color="#222232", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-rl - 10, rl + 10)
    ax.set_ylim(-rl - 10, rl + 10)

    # Road fills — N-S and E-W arms
    road_c = "#363640"
    ax.fill([-lw*2,  lw*2,  lw*2, -lw*2],
            [-rl-5, -rl-5,  rl+5,  rl+5], color=road_c)
    ax.fill([-rl-5,  rl+5,  rl+5, -rl-5],
            [-lw*2, -lw*2,  lw*2,  lw*2], color=road_c)

    # Lane markings
    lc = "#AAAAAA"
    ax.plot([0, 0], [-rl-5, -lw*2], "--", color=lc, lw=0.9)
    ax.plot([0, 0], [ lw*2,  rl+5], "--", color=lc, lw=0.9)
    ax.plot([-rl-5, -lw*2], [0, 0], "--", color=lc, lw=0.9)
    ax.plot([ lw*2,  rl+5], [0, 0], "--", color=lc, lw=0.9)

    # Zebra crossings (N and S approach)
    for zy in [-lw*2 - 2, lw*2 + 2]:
        xi = -lw * 1.8
        while xi < lw * 1.8:
            ax.fill([xi, xi+0.8, xi+0.8, xi],
                    [zy, zy, zy+0.8, zy+0.8],
                    color="#DDDDDD", alpha=0.45)
            xi += 1.2

    # Direction labels
    for txt, tx, ty in [("N ↑", 4, rl+2), ("S ↓", 4, -rl-6),
                         ("W ←", -rl-7, 4), ("E →", rl+2, 4)]:
        ax.text(tx, ty, txt, color="#8899BB", fontsize=8)

    # Goal
    ax.plot(goal[0], goal[1], "p", color="#FFCC00", ms=16, mec="white", mew=1.5)
    ax.text(goal[0]+2, goal[1]+3, "GOAL", color="#FFCC00", fontsize=9.5)

    # Actor labels (static position near starting spot)
    ax.text(car1["x"],    car1["y"]+3, "Car W→E",  color="#FF8888", fontsize=7.5)
    ax.text(car2["x"]-14, car2["y"]+3, "Car E→W",  color="#FF8888", fontsize=7.5)
    ax.text(rick["x"]-4,  rick["y"]+3, "Rickshaw", color="#FFAA44", fontsize=7.5)

    # Dynamic handles
    h_ego,  = ax.plot([], [], "o", color="#00DD44", ms=13, mec="white", mew=1.8)
    h_car1, = ax.plot([], [], "s", color="#EE3333", ms=12, mec="white", mew=1.4)
    h_car2, = ax.plot([], [], "s", color="#EE3333", ms=12, mec="white", mew=1.4)
    h_rick, = ax.plot([], [], "D", color="#DD8822", ms=11, mec="white", mew=1.4)
    h_ped1, = ax.plot([], [], "^", color="#CC44CC", ms=10, mec="white", mew=1.3)
    h_ped2, = ax.plot([], [], "^", color="#CC44CC", ms=10, mec="white", mew=1.3)
    h_cyc,  = ax.plot([], [], "v", color="#44AAFF", ms=10, mec="white", mew=1.3)
    h_traj, = ax.plot([], [], "--", color="#448844", lw=0.9)

    h_status = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                       color="white", fontsize=10, va="top", fontweight="bold")
    fig.tight_layout()

    _running = [True]
    fig.canvas.mpl_connect("close_event", lambda e: _running.__setitem__(0, False))
    plt.show(block=False)
    fig.canvas.flush_events()

    # ── Simulation ────────────────────────────────────────────────────────
    t = 0.0
    arrived = False
    last_decision = ""
    traj_x: List[float] = [ego["x"]]
    traj_y: List[float] = [ego["y"]]
    event_log: List[str] = []
    in_intersection = False

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

        in_intersection = abs(ego["x"]) < lw*2 and abs(ego["y"]) < lw*2

        # ── Distances to actors ───────────────────────────────────────
        d_c1  = math.hypot(car1["x"] - ego["x"], car1["y"] - ego["y"])
        d_c2  = math.hypot(car2["x"] - ego["x"], car2["y"] - ego["y"])
        d_rk  = math.hypot(rick["x"] - ego["x"], rick["y"] - ego["y"])
        d_p1  = math.hypot(ped1["x"] - ego["x"], ped1["y"] - ego["y"])
        d_p2  = math.hypot(ped2["x"] - ego["x"], ped2["y"] - ego["y"])

        in_int_zone = math.hypot(ego["x"], ego["y"]) < lw*3 + 5

        # ── Decision ──────────────────────────────────────────────────
        event = ""
        if (d_p1 < 10 or d_p2 < 10) and in_int_zone:
            decision     = "BRAKE"
            speed_target = 1.0
            event = (f"t={t:.1f}s: BRAKE — pedestrian in crossing "
                     f"({min(d_p1, d_p2):.1f}m)")
        elif (d_c1 < 20 or d_c2 < 20) and in_int_zone:
            decision     = "SLOW"
            speed_target = 3.0
            event = f"t={t:.1f}s: SLOW — cross-traffic {min(d_c1, d_c2):.1f}m"
        elif d_rk < 18 and in_int_zone:
            decision     = "CAUTION"
            speed_target = 5.0
            event = f"t={t:.1f}s: CAUTION — rickshaw {d_rk:.1f}m"
        elif in_int_zone:
            decision     = "CAUTION"
            speed_target = 5.0
        else:
            decision     = "PROCEED"
            speed_target = 8.33
        if event and decision != last_decision:
            event_log.append(event)
            print(f"  {event}")
        last_decision = decision

        # ── Pure-pursuit toward goal ───────────────────────────────────
        alpha = math.atan2(goal[1] - ego["y"], goal[0] - ego["x"]) - ego["h"]
        alpha = _wrap(alpha)
        steer = math.atan2(2 * wb * math.sin(alpha), pp_la)
        steer = float(np.clip(steer, -max_st, max_st))

        # ── Speed control ─────────────────────────────────────────────
        ego["spd"] = float(np.clip(
            ego["spd"] + 0.8 * (speed_target - ego["spd"]) * dt, 0, 15))

        # ── Kinematic integration ─────────────────────────────────────
        ego["x"]  += ego["spd"] * math.cos(ego["h"]) * dt
        ego["y"]  += ego["spd"] * math.sin(ego["h"]) * dt
        ego["h"]  += (ego["spd"] / wb) * math.tan(steer) * dt
        traj_x.append(ego["x"])
        traj_y.append(ego["y"])

        # Actor updates
        car1["x"] += car1["spd"] * math.cos(car1["h"]) * dt
        car2["x"] += car2["spd"] * math.cos(car2["h"]) * dt
        rick["x"] += rick["spd"] * math.cos(rick["h"]) * dt
        ped1["x"] += ped1["spd"] * math.cos(ped1["h"]) * dt
        ped2["x"] += ped2["spd"] * math.cos(ped2["h"]) * dt
        cyc["y"]  += cyc["spd"]  * math.sin(cyc["h"])  * dt

        # Log
        log_data["t"].append(t)
        log_data["x"].append(ego["x"])
        log_data["y"].append(ego["y"])
        log_data["speed_mps"].append(ego["spd"])
        log_data["decision"].append(decision)

        # ── Plot ──────────────────────────────────────────────────────
        h_ego.set_data( [ego["x"]],  [ego["y"]])
        h_car1.set_data([car1["x"]], [car1["y"]])
        h_car2.set_data([car2["x"]], [car2["y"]])
        h_rick.set_data([rick["x"]], [rick["y"]])
        h_ped1.set_data([ped1["x"]], [ped1["y"]])
        h_ped2.set_data([ped2["x"]], [ped2["y"]])
        h_cyc.set_data( [cyc["x"]],  [cyc["y"]])
        h_traj.set_data(traj_x, traj_y)

        col = _color(decision)
        h_status.set_text(
            f"t={t:.1f} s  |  {decision}  |  "
            f"Speed: {ego['spd']*3.6:.1f} km/h  |  Dist: {dist_to_goal:.1f} m")
        h_status.set_color(col)
        ax.set_title(f"Scenario 2 — Urban Intersection  |  {decision}",
                     color=col, fontsize=12)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        t += dt

    log_data["scenario_completed"] = arrived

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== Scenario 2 Complete ===")
    if arrived:
        print(f"  Goal reached in {t:.1f} s")
    else:
        print(f"  Ended at t={t:.1f} s")
    print(f"  Events: {len(event_log)}")

    if collect_metrics_at_end and log_data["t"]:
        try:
            from simulation.collect_metrics import collect_metrics
            collect_metrics(log_data, "urban_intersection", save_csv=True)
        except Exception as exc:
            print(f"  [WARN] collect_metrics: {exc}")

    plt.ioff()
    plt.show()
    return log_data


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ADAS Scenario 2 — Urban Intersection")
    p.add_argument("--no-metrics", action="store_true")
    args = p.parse_args()
    run_scenario(collect_metrics_at_end=not args.no_metrics)
