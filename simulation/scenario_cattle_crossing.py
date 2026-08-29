"""
scenario_cattle_crossing.py — ADAS Vision Driving Scenario 5: Cattle Crossing

Python equivalent of scenario_cattle_crossing.m.

Standalone simulation (no UDP bridge required). Creates a village road
where TWO COWS suddenly walk onto the road from the left shoulder after
3 seconds. The ego vehicle must detect the obstacle, brake, and wait for
the cows to clear before proceeding.

Usage:
    python simulation/scenario_cattle_crossing.py
    python simulation/scenario_cattle_crossing.py --no-metrics
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
    print("=== Scenario 5: Cattle Crossing ===")

    road_width_m = 6.0
    road_len_m   = 180.0
    lw           = road_width_m / 2     # half-width = 3.0 m

    dt     = SIM.dt_s
    t_end  = 35.0
    wb     = VEH.wheelbase_m
    max_st = VEH.max_steer_rad
    pp_la  = 10.0
    goal   = np.array([160.0, 0.0])
    goal_tol = 4.0

    # ── Actors ────────────────────────────────────────────────────────────
    ego   = {"x": 5.0,  "y": 0.0,       "h": 0.0,         "spd": 11.1}
    cow1  = {"x": 80.0, "y": -lw - 2.0, "spd": 0.8,
             "h": math.pi/2, "active": False}
    cow2  = {"x": 77.0, "y": -lw - 3.0, "spd": 0.7,
             "h": math.pi/2, "active": False}

    # ── Figure ────────────────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("#0C1208")
    ax.set_facecolor("#182010")
    for sp in ax.spines.values():
        sp.set_edgecolor("#304028")
    ax.tick_params(colors="white")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title("Scenario 5 — Cattle Crossing  |  Waiting for event...",
                 color="#AAFFAA", fontsize=12)
    ax.grid(color="#1A2414", linewidth=0.5)
    ax.set_xlim(0, road_len_m + 10)
    ax.set_ylim(-lw - 7, lw + 7)

    # Road fill
    ax.fill([0, road_len_m, road_len_m, 0],
            [-lw, -lw, lw, lw],
            color="#3A4430", alpha=0.75)

    # Road markings
    ax.plot([0, road_len_m], [ lw,  lw], "w-", lw=1.4)
    ax.plot([0, road_len_m], [-lw, -lw], "w-", lw=1.4)
    ax.plot([0, road_len_m], [ 0,   0],  "--", color="#BBAA00", lw=0.8, alpha=0.7)

    # Grass shoulders
    ax.fill([0, road_len_m, road_len_m, 0],
            [-lw-7, -lw-7, -lw, -lw], color="#1E3B10", alpha=0.6)
    ax.fill([0, road_len_m, road_len_m, 0],
            [ lw,  lw, lw+7, lw+7], color="#1E3B10", alpha=0.6)

    # Cattle warning sign at x=65
    ax.plot(65, -lw - 1.5, "^", color="#FFEE00", ms=14,
            mec="black", mew=1.2)
    ax.text(65, -lw - 4, "CATTLE\nXING", color="#FFEE00", fontsize=7.5,
            ha="center", va="top")

    # Goal marker
    ax.plot(goal[0], goal[1], "p", color="#FFCC00", ms=16, mec="white", mew=1.5)
    ax.text(goal[0]+2, goal[1]+1.5, "GOAL", color="#FFCC00", fontsize=9.5)

    # Dynamic handles
    h_ego,  = ax.plot([], [], "o", color="#00DD44", ms=14, mec="white", mew=1.8)
    h_cow1, = ax.plot([], [], "s", color="#996633", ms=16, mec="white", mew=1.8)
    h_cow2, = ax.plot([], [], "s", color="#996633", ms=16, mec="white", mew=1.8)
    h_traj, = ax.plot([], [], "--", color="#55AA55", lw=0.9)

    # Cow text labels (using ASCII to avoid emoji font warnings)
    ax.text(cow1["x"] + 1, cow1["y"] + 0.8, "COW", fontsize=9,
            color="#CC9944", fontweight="bold")
    ax.text(cow2["x"] + 1, cow2["y"] + 0.8, "COW", fontsize=9,
            color="#CC9944", fontweight="bold")

    # Countdown / status overlays
    h_count = ax.text(80, lw + 2.5,
                      f"⚠ Cows enter in {3.0:.1f} s",
                      color="#FFAA33", fontsize=10, ha="center",
                      fontweight="bold")
    h_status = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                       color="white", fontsize=10, va="top", fontweight="bold")

    fig.tight_layout()

    # Close-event callback — lets user close the window early
    _running = [True]
    fig.canvas.mpl_connect("close_event", lambda e: _running.__setitem__(0, False))

    plt.show(block=False)
    fig.canvas.flush_events()

    # ── Simulation ────────────────────────────────────────────────────────
    t = 0.0
    arrived   = False
    waiting   = False
    last_decision = ""
    traj_x: List[float] = [ego["x"]]
    traj_y: List[float] = [ego["y"]]
    event_log: List[str] = []

    log_data = {
        "t": [], "x": [], "y": [], "speed_mps": [],
        "decision": [], "collisions": 0, "replans": 0,
        "replan_latency_ms": [], "scenario_completed": False,
        "min_dist_m": [],
    }

    while t < t_end and _running[0]:
        dist_to_goal = float(np.linalg.norm(goal - np.array([ego["x"], ego["y"]])))
        if dist_to_goal < goal_tol:
            arrived = True
            break

        # ── Cattle event trigger ───────────────────────────────────────
        if t >= 3.0:
            cow1["active"] = True
        if t >= 3.5:
            cow2["active"] = True

        # ── Update cows ────────────────────────────────────────────────
        if cow1["active"] and cow1["y"] < lw + 3:
            cow1["y"] += cow1["spd"] * dt
            if cow1["y"] > lw + 2:
                cow1["active"] = False
        if cow2["active"] and cow2["y"] < lw + 3:
            cow2["y"] += cow2["spd"] * dt
            if cow2["y"] > lw + 2:
                cow2["active"] = False

        # ── Decision ──────────────────────────────────────────────────
        cow_in_road1 = -lw < cow1["y"] < lw
        cow_in_road2 = -lw < cow2["y"] < lw
        cow_in_road  = cow_in_road1 or cow_in_road2

        dist_c1  = math.hypot(cow1["x"] - ego["x"], cow1["y"] - ego["y"])
        dist_c2  = math.hypot(cow2["x"] - ego["x"], cow2["y"] - ego["y"])
        dist_cow = min(dist_c1, dist_c2)
        log_data["min_dist_m"].append(dist_cow)

        event = ""
        if cow_in_road and dist_cow < 8:
            decision     = "EMERGENCY_STOP"
            speed_target = 0.0
            if not waiting:
                event   = (f"t={t:.1f}s: EMERGENCY_STOP — "
                           f"cow {dist_cow:.1f}m in-path")
                waiting = True
        elif cow_in_road and dist_cow < 20:
            decision     = "BRAKE"
            speed_target = 0.5
            if t > 3.5:
                event = (f"t={t:.1f}s: BRAKE — "
                         f"cow {dist_cow:.1f}m ahead (in road)")
        elif dist_cow < 35 and cow_in_road:
            decision     = "SLOW"
            speed_target = 3.0
            event = f"t={t:.1f}s: SLOW — cow {dist_cow:.1f}m ahead"
        elif dist_cow < 50 and t > 2.5:
            decision     = "CAUTION"
            speed_target = 6.0
            if t < 4.5:
                event = (f"t={t:.1f}s: CAUTION — "
                         f"cattle detected {dist_cow:.1f}m ahead")
        else:
            decision     = "PROCEED"
            speed_target = 11.1
            if waiting and not cow_in_road:
                event   = f"t={t:.1f}s: PROCEED — road clear, resuming"
                waiting = False

        # Only log when decision LEVEL changes (not every distance tick)
        if decision != last_decision:
            if event:
                event_log.append(event)
                print(f"  {event}")
            elif decision in ("SLOW", "BRAKE", "EMERGENCY_STOP", "CAUTION"):
                msg = f"t={t:.1f}s: {decision} — cow {dist_cow:.1f}m"
                event_log.append(msg)
                print(f"  {msg}")
            last_decision = decision

        # ── Pure-pursuit: straight to goal ────────────────────────────
        alpha = math.atan2(goal[1] - ego["y"], goal[0] - ego["x"]) - ego["h"]
        alpha = _wrap(alpha)
        steer = math.atan2(2 * wb * math.sin(alpha), pp_la)
        steer = float(np.clip(steer, -max_st, max_st))

        # ── Speed control ─────────────────────────────────────────────
        ego["spd"] = float(np.clip(
            ego["spd"] + 1.2 * (speed_target - ego["spd"]) * dt, 0, 15))

        # ── Kinematic integration ─────────────────────────────────────
        ego["x"] += ego["spd"] * math.cos(ego["h"]) * dt
        ego["y"] += ego["spd"] * math.sin(ego["h"]) * dt
        ego["h"] += (ego["spd"] / wb) * math.tan(steer) * dt
        traj_x.append(ego["x"])
        traj_y.append(ego["y"])

        # Log
        log_data["t"].append(t)
        log_data["x"].append(ego["x"])
        log_data["y"].append(ego["y"])
        log_data["speed_mps"].append(ego["spd"])
        log_data["decision"].append(decision)

        # ── Plot ──────────────────────────────────────────────────────
        h_ego.set_data( [ego["x"]],  [ego["y"]])
        h_cow1.set_data([cow1["x"]], [cow1["y"]])
        h_cow2.set_data([cow2["x"]], [cow2["y"]])
        h_traj.set_data(traj_x, traj_y)

        # Countdown label
        if t < 3.0:
            h_count.set_text(f"⚠ Cows enter in {3.0 - t:.1f} s")
            h_count.set_color("#FFAA33")
            h_count.set_visible(True)
        elif t < 5.0:
            h_count.set_text("⚠ CATTLE ON ROAD")
            h_count.set_color("#FF2222")
            h_count.set_fontsize(13)
            h_count.set_visible(True)
        else:
            h_count.set_visible(False)

        col = _color(decision)
        h_status.set_text(
            f"t={t:.1f} s  |  {decision}  |  "
            f"Speed: {ego['spd']*3.6:.1f} km/h  |  Cow dist: {dist_cow:.1f} m")
        h_status.set_color(col)
        ax.set_title(f"Scenario 5 — Cattle Crossing  |  {decision}",
                     color=col, fontsize=12)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        t += dt

    log_data["scenario_completed"] = arrived

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== Scenario 5 Complete ===")
    if arrived:
        print(f"  Goal reached in {t:.1f} s")
    else:
        print(f"  Ended at t={t:.1f} s")
    print(f"  Events: {len(event_log)}")
    for ev in event_log:
        print(f"    {ev}")

    if collect_metrics_at_end and log_data["t"]:
        try:
            from simulation.collect_metrics import collect_metrics
            collect_metrics(log_data, "cattle_crossing", save_csv=True)
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
        description="ADAS Scenario 5 — Cattle Crossing")
    p.add_argument("--no-metrics", action="store_true")
    args = p.parse_args()
    run_scenario(collect_metrics_at_end=not args.no_metrics)
