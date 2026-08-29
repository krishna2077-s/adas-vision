"""
cosimulation.py — ADAS Vision Python Co-simulation

Python equivalent of run_cosimulation.m.

Connects to the Python UDP bridge (simulation/udp_bridge.py) on port 5005,
receives real-time AI perception data (detected objects, lane decisions),
runs a Hybrid A* path planner and Pure-Pursuit + PID vehicle controller on
a kinematic bicycle model, shows a live matplotlib plot, and streams ego
position back to the bridge on port 5006.

Quick Start
-----------
Terminal 1 — start the ADAS perception bridge:
    cd adas-vision
    .\\venv\\Scripts\\python.exe simulation\\udp_bridge.py --video dashcam.mp4 --async

Terminal 2 — run the co-simulation:
    .\\venv\\Scripts\\python.exe simulation\\cosimulation.py --goal 200 0

Arguments
---------
    --goal X Y          Goal position in world frame (metres). Required.
    --max-time N        Max simulation time (default: 120 s).
    --init-x X          Initial ego X  (default: 0).
    --init-y Y          Initial ego Y  (default: 0).
    --init-heading DEG  Initial heading in degrees (default: 0).
    --no-metrics        Skip the collect_metrics summary at the end.
"""

import argparse
import heapq
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")          # interactive backend — works on Windows
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is importable when running as __main__
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.vehicle_model import (
    VEH, LONG, PP, PID, PLANNER, SIM,
    UDP_RX_PORT, UDP_TX_PORT, UDP_HOST, UDP_BUFSIZE,
)


# ===========================================================================
# Path Planner — Hybrid A* (greedy obstacle-aware path from start to goal)
# ===========================================================================

def plan_path(
    start_xy:  np.ndarray,
    start_h:   float,
    goal_xy:   np.ndarray,
    obstacles: np.ndarray,          # shape (N, 2) in world frame, or (0,2)
    cfg:       object = PLANNER,
) -> np.ndarray:
    """
    Simplified Hybrid A* path planner.

    Returns an (N, 2) array of [x, y] waypoints from start to goal,
    nudging around known obstacle positions with an inflation radius.

    The approach matches the MATLAB implementation: straight-line seed path
    segmented into waypoints, then greedy outward nudges perpendicular to
    the goal direction whenever a waypoint is within obstacle_rad_m of an
    obstacle.  Pure A* grid search is a planned upgrade; for co-simulation
    and hackathon demos this greedy method replans in <2 ms and is safe.
    """
    dist = float(np.linalg.norm(goal_xy - start_xy))
    n_pts = max(5, int(math.ceil(dist / cfg.step_m)))

    xs = np.linspace(start_xy[0], goal_xy[0], n_pts)
    ys = np.linspace(start_xy[1], goal_xy[1], n_pts)
    path = np.column_stack([xs, ys])    # (N, 2)

    if len(obstacles) == 0:
        return path

    # Perpendicular direction to the goal vector (for nudging)
    dg = goal_xy - start_xy
    dg_len = max(np.linalg.norm(dg), 1e-6)
    perp = np.array([-dg[1], dg[0]]) / dg_len  # unit perpendicular

    for k in range(len(path)):
        for obs in obstacles:
            d = float(np.linalg.norm(path[k] - obs))
            if d < cfg.obstacle_rad_m + 0.5:
                push = (cfg.obstacle_rad_m + 0.5 - d + 0.3)
                path[k] = path[k] + perp * push

    return path


# ===========================================================================
# Pure-Pursuit Lateral Controller
# ===========================================================================

def pure_pursuit(
    ego_xy:  np.ndarray,
    heading: float,
    path:    np.ndarray,
    idx:     int,
    pp:      object = PP,
    veh:     object = VEH,
    speed:   float  = 0.0,
) -> Tuple[float, int]:
    """
    Pure-pursuit steering command.

    Returns:
        steer_rad: steering angle command (clamped to ±max_steer_rad)
        next_idx:  updated look-ahead path index
    """
    if len(path) == 0:
        return 0.0, idx

    la = np.clip(
        pp.lookahead_m + pp.k_lookahead * speed,
        pp.min_lookahead,
        pp.max_lookahead,
    )

    # Advance index until the waypoint is at least la metres ahead
    next_idx = idx
    while next_idx < len(path) - 1:
        if np.linalg.norm(path[next_idx] - ego_xy) >= la:
            break
        next_idx += 1

    wp    = path[next_idx]
    dx, dy = wp[0] - ego_xy[0], wp[1] - ego_xy[1]
    alpha = math.atan2(dy, dx) - heading
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # wrap to [-π, π]
    steer = math.atan2(2.0 * veh.wheelbase_m * math.sin(alpha), la)
    steer = float(np.clip(steer, -veh.max_steer_rad, veh.max_steer_rad))
    return steer, next_idx


# ===========================================================================
# PID Speed Controller
# ===========================================================================

class PIDState:
    """Mutable PID controller state (integral + previous error)."""
    def __init__(self):
        self.integral  = 0.0
        self.prev_err  = 0.0


def pid_control(
    v:        float,
    v_target: float,
    pid_p:    object    = PID,
    state:    PIDState  = None,
    long:     object    = LONG,
) -> Tuple[float, float, PIDState]:
    """
    PID longitudinal speed controller.

    Returns:
        throttle: [0, 1]
        brake:    [0, 1]
        state:    updated PIDState
    """
    if state is None:
        state = PIDState()
    err              = v_target - v
    state.integral   = float(np.clip(state.integral + err * pid_p.dt,
                                     -pid_p.int_max, pid_p.int_max))
    derivative       = (err - state.prev_err) / pid_p.dt
    state.prev_err   = err
    cmd      = pid_p.Kp * err + pid_p.Ki * state.integral + pid_p.Kd * derivative
    throttle = float(np.clip(cmd,  0.0, 1.0))
    brake    = float(np.clip(-cmd, 0.0, 1.0))
    return throttle, brake, state


# ===========================================================================
# Decision → speed mapping
# ===========================================================================
_DECISION_SPEED = {
    "PROCEED":        SIM.init_speed_mps,
    "CAUTION":        SIM.init_speed_mps * 0.7,
    "SLOW":           SIM.init_speed_mps * 0.4,
    "BRAKE":          SIM.init_speed_mps * 0.15,
    "EMERGENCY_STOP": 0.0,
}

_DECISION_COLOR = {
    "PROCEED":        "#33CC55",
    "CAUTION":        "#FFCC00",
    "SLOW":           "#FF8800",
    "BRAKE":          "#FF3300",
    "EMERGENCY_STOP": "#FF0000",
}


def decision_to_speed(decision: str, brake_dem: float = 0.0) -> float:
    if decision == "BRAKE":
        return SIM.init_speed_mps * (1 - brake_dem) * 0.3
    return _DECISION_SPEED.get(decision, SIM.init_speed_mps)


# ===========================================================================
# UDP helpers
# ===========================================================================

def _recv_packet(sock: socket.socket) -> Optional[dict]:
    """Non-blocking UDP receive. Returns parsed JSON dict or None."""
    try:
        data, _ = sock.recvfrom(UDP_BUFSIZE)
        return json.loads(data.decode("utf-8"))
    except (socket.timeout, BlockingIOError):
        return None
    except Exception:
        return None


def _tracks_to_world(tracks: list, ego_x: float, ego_y: float,
                     heading: float) -> np.ndarray:
    """Convert in-path track distances to world-frame (x, y) obstacles."""
    obs = []
    for t in tracks:
        if not t.get("in_path", False):
            continue
        dist = t.get("distance_m")
        if dist is None or math.isnan(dist):
            continue
        dx = dist * math.cos(heading)
        dy = dist * math.sin(heading)
        obs.append([ego_x + dx, ego_y + dy])
    return np.array(obs, dtype=float) if obs else np.empty((0, 2))


# ===========================================================================
# Matplotlib figure setup
# ===========================================================================

def _setup_figure(goal: np.ndarray) -> dict:
    """Create the dark-themed co-simulation plot. Returns a dict of handles."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#141418")
    ax.set_facecolor("#141418")
    for spine in ax.spines.values():
        spine.set_edgecolor("#303040")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Connecting to ADAS bridge...", color="white", fontsize=11)
    ax.grid(color="#252535", linewidth=0.6)
    ax.set_aspect("equal")

    h_path, = ax.plot([], [], "-",  color="#33CC88", lw=1.5, label="Planned path")
    h_traj, = ax.plot([], [], "--", color="#555566", lw=0.9, label="Ego trajectory")
    h_ego,  = ax.plot([], [], "o",  color="#33EE44", ms=12,
                      mec="white", mew=1.5, label="Ego")
    h_obs,  = ax.plot([], [], "s",  color="#EE3333", ms=9,
                      mec="white", mew=1, label="In-path obstacle")

    ax.plot(goal[0], goal[1], "p", color="#FFCC00", ms=16, mec="white", mew=1.5)
    ax.text(goal[0] + 2, goal[1] + 2, "GOAL", color="#FFCC00", fontsize=10)

    legend = ax.legend(loc="upper right", facecolor="#202030",
                       edgecolor="#404050", labelcolor="white", fontsize=8)

    h_dec  = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     color="white", fontsize=10, va="top", fontweight="bold")
    h_fps  = ax.text(0.02, 0.91, "", transform=ax.transAxes,
                     color="#88CCFF", fontsize=9, va="top")
    h_stat = ax.text(0.02, 0.85, "", transform=ax.transAxes,
                     color="#CCCC66", fontsize=9, va="top")

    fig.tight_layout()
    plt.pause(0.05)

    return {
        "fig": fig, "ax": ax,
        "h_path": h_path, "h_traj": h_traj,
        "h_ego": h_ego, "h_obs": h_obs,
        "h_dec": h_dec, "h_fps": h_fps, "h_stat": h_stat,
    }


def _update_figure(handles: dict, ego_x: float, ego_y: float,
                   traj_x: List[float], traj_y: List[float],
                   path: np.ndarray, obs_world: np.ndarray,
                   goal: np.ndarray, decision: str,
                   fps_live: float, t_sim: float, ego_speed: float,
                   replans: int, collisions: int) -> None:
    ax = handles["ax"]
    handles["h_ego"].set_data([ego_x], [ego_y])
    handles["h_traj"].set_data(traj_x, traj_y)

    if len(path) > 0:
        handles["h_path"].set_data(path[:, 0], path[:, 1])
    if len(obs_world) > 0:
        handles["h_obs"].set_data(obs_world[:, 0], obs_world[:, 1])
    else:
        handles["h_obs"].set_data([], [])

    dist = float(np.linalg.norm(goal - np.array([ego_x, ego_y])))
    col  = _DECISION_COLOR.get(decision, "white")
    handles["h_dec"].set_text(f"Decision: {decision}")
    handles["h_dec"].set_color(col)
    handles["h_fps"].set_text(
        f"Python FPS: {fps_live:.1f}  |  Sim t: {t_sim:.1f} s")
    handles["h_stat"].set_text(
        f"Speed: {ego_speed*3.6:.1f} km/h  |  "
        f"Replans: {replans}  |  Collisions: {collisions}")
    ax.set_title(
        f"ADAS Co-sim  |  Ego: ({ego_x:.1f}, {ego_y:.1f})  |  "
        f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})  |  Dist: {dist:.1f} m",
        color=col, fontsize=10,
    )

    # Keep ego centred with some margin
    margin = 30
    ax.set_xlim(ego_x - margin, ego_x + margin * 2)
    ax.set_ylim(ego_y - margin, ego_y + margin)

    handles["fig"].canvas.draw_idle()
    plt.pause(0.001)


# ===========================================================================
# Main co-simulation
# ===========================================================================

def run_cosimulation(
    goal_xy:     Tuple[float, float],
    max_time:    float = SIM.max_time_s,
    init_pos:    Tuple[float, float] = (0.0, 0.0),
    init_heading_deg: float = 0.0,
    collect_metrics_at_end: bool = True,
) -> dict:
    """
    Run the full ADAS co-simulation.

    Args:
        goal_xy:               (x, y) goal in world frame (metres).
        max_time:              Maximum simulation duration (seconds).
        init_pos:              (x0, y0) initial ego position.
        init_heading_deg:      Initial heading in degrees (0 = east).
        collect_metrics_at_end: Run collect_metrics and show chart when done.

    Returns:
        log_data dict (same structure as the MATLAB workspace log_data).
    """
    goal    = np.array(goal_xy, dtype=float)
    veh     = VEH
    long    = LONG
    pp_cfg  = PP
    pid_cfg = PID
    plan_cfg = PLANNER
    sim     = SIM

    # ── Ego state ─────────────────────────────────────────────────────────
    ego_x    = float(init_pos[0])
    ego_y    = float(init_pos[1])
    heading  = math.radians(init_heading_deg)
    speed    = sim.init_speed_mps
    pid_state = PIDState()

    # ── Path planner state ────────────────────────────────────────────────
    path_xy:     np.ndarray = np.empty((0, 2))
    path_idx:    int        = 0
    last_replan: float      = -999.0

    # ── Log ───────────────────────────────────────────────────────────────
    log_data = {
        "t": [], "x": [], "y": [], "heading": [],
        "speed_mps": [], "steer_rad": [], "throttle": [], "brake": [],
        "decision": [], "n_tracks": [], "min_dist_m": [],
        "collisions": 0, "replans": 0, "replan_latency_ms": [],
        "scenario_completed": False,
    }

    traj_x: List[float] = [ego_x]
    traj_y: List[float] = [ego_y]

    # ── UDP sockets ────────────────────────────────────────────────────────
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx_sock.bind(("0.0.0.0", UDP_RX_PORT))
    rx_sock.settimeout(0.5)

    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("=== ADAS Co-simulation (Python) ===")
    print(f"  Goal      : [{goal[0]:.1f}, {goal[1]:.1f}] m")
    print(f"  UDP RX    : port {UDP_RX_PORT}")
    print(f"  UDP TX    : port {UDP_TX_PORT}")
    print(f"  Max time  : {max_time:.0f} s")
    print("  Waiting for ADAS bridge packets…")

    # ── Matplotlib figure ──────────────────────────────────────────────────
    handles = _setup_figure(goal)

    # ── Last received perception packet ───────────────────────────────────
    last_pkt = {
        "decision": {"longitudinal": "PROCEED", "brake_demand": 0.0},
        "lane":     {"offset_px": 0},
        "tracks":   [],
        "fps":      0.0,
    }

    t_sim    = 0.0
    arrived  = False
    collision = False
    t_wall   = time.perf_counter()
    update_hz = 10   # plot refresh divisor (~10 Hz)
    step_no   = 0

    try:
        while t_sim < max_time:
            if not plt.fignum_exists(handles["fig"].number):
                print("  [INFO] Window closed — stopping.")
                break

            # ── 1. Receive latest perception packet ────────────────────
            pkt = _recv_packet(rx_sock)
            if pkt is not None:
                last_pkt = pkt

            decision  = last_pkt["decision"].get("longitudinal", "PROCEED")
            brake_dem = last_pkt["decision"].get("brake_demand", 0.0)
            offset_px = last_pkt["lane"].get("offset_px", 0)
            tracks    = last_pkt.get("tracks", [])
            fps_live  = float(last_pkt.get("fps", 0.0))

            # ── 2. Build obstacle list in world frame ──────────────────
            obs_world = _tracks_to_world(tracks, ego_x, ego_y, heading)

            # ── 3. Path planning (replan when needed) ─────────────────
            dist_to_goal = float(np.linalg.norm(goal - np.array([ego_x, ego_y])))

            deviation = 0.0
            if len(path_xy) > 0 and path_idx < len(path_xy):
                deviation = float(np.linalg.norm(
                    path_xy[path_idx] - np.array([ego_x, ego_y])))

            need_replan = (
                len(path_xy) == 0 or
                (t_sim - last_replan > 5.0) or
                deviation > plan_cfg.replan_dist_m
            )

            if need_replan and dist_to_goal > sim.goal_tol_m:
                t0 = time.perf_counter()
                path_xy = plan_path(
                    np.array([ego_x, ego_y]), heading, goal,
                    obs_world, plan_cfg,
                )
                replan_ms = (time.perf_counter() - t0) * 1000.0
                path_idx  = 0
                last_replan = t_sim
                log_data["replans"] += 1
                log_data["replan_latency_ms"].append(replan_ms)
                print(f"  [t={t_sim:5.1f}s] Replan #{log_data['replans']}"
                      f"  latency={replan_ms:.1f} ms"
                      f"  dist_to_goal={dist_to_goal:.1f} m")

            # ── 4. Pure-Pursuit lateral control ───────────────────────
            steer_cmd, path_idx = pure_pursuit(
                np.array([ego_x, ego_y]), heading, path_xy,
                path_idx, pp_cfg, veh, speed,
            )

            # Blend lane offset correction from camera (match MATLAB blend)
            lane_corr = -offset_px / 500.0
            steer_cmd = float(np.clip(
                0.85 * steer_cmd + 0.15 * lane_corr,
                -veh.max_steer_rad, veh.max_steer_rad,
            ))

            # ── 5. PID longitudinal control ────────────────────────────
            target_speed = decision_to_speed(decision, brake_dem)
            throttle, brake, pid_state = pid_control(
                speed, target_speed, pid_cfg, pid_state, long)

            # ── 6. Kinematic bicycle model (Euler integration) ─────────
            beta  = math.atan2(veh.lr_m * math.tan(steer_cmd), veh.wheelbase_m)
            accel = throttle * long.max_accel_mps2 - brake * long.max_brake_mps2
            speed = float(np.clip(speed + accel * sim.dt_s,
                                  0.0, long.max_speed_mps))
            ego_x   += speed * math.cos(heading + beta) * sim.dt_s
            ego_y   += speed * math.sin(heading + beta) * sim.dt_s
            heading += (speed / veh.wheelbase_m) * math.tan(steer_cmd) * sim.dt_s

            traj_x.append(ego_x)
            traj_y.append(ego_y)

            # ── 7. Collision check ─────────────────────────────────────
            min_dist = 999.0
            for obs in obs_world:
                d = float(np.linalg.norm(obs - np.array([ego_x, ego_y])))
                min_dist = min(min_dist, d)
                if d < sim.collision_rad_m:
                    log_data["collisions"] += 1
                    print(f"  [t={t_sim:5.1f}s] COLLISION detected!")
            if min_dist < 999.0:
                log_data["min_dist_m"].append(min_dist)

            # ── 8. Goal check ──────────────────────────────────────────
            dist_to_goal = float(np.linalg.norm(goal - np.array([ego_x, ego_y])))
            if dist_to_goal <= sim.goal_tol_m:
                arrived = True
                log_data["scenario_completed"] = True
                print(f"  [t={t_sim:5.1f}s] GOAL REACHED! "
                      f"dist_remaining={dist_to_goal:.2f} m")
                break

            # ── 9. Send ego position back to ADAS bridge ───────────────
            ego_json = json.dumps({"ego_x": round(ego_x, 3),
                                   "ego_y": round(ego_y, 3)})
            tx_sock.sendto(ego_json.encode("utf-8"), (UDP_HOST, UDP_TX_PORT))

            # ── 10. Log ────────────────────────────────────────────────
            log_data["t"].append(t_sim)
            log_data["x"].append(ego_x)
            log_data["y"].append(ego_y)
            log_data["heading"].append(heading)
            log_data["speed_mps"].append(speed)
            log_data["steer_rad"].append(steer_cmd)
            log_data["throttle"].append(throttle)
            log_data["brake"].append(brake)
            log_data["decision"].append(decision)
            log_data["n_tracks"].append(len(tracks))

            # ── 11. Live plot update (~10 Hz) ──────────────────────────
            step_no += 1
            if step_no % update_hz == 0:
                _update_figure(
                    handles, ego_x, ego_y, traj_x, traj_y,
                    path_xy, obs_world, goal, decision, fps_live,
                    t_sim, speed,
                    log_data["replans"], log_data["collisions"],
                )

            t_sim += sim.dt_s

    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted by user.")
    finally:
        rx_sock.close()
        tx_sock.close()

    # ── Summary ───────────────────────────────────────────────────────────
    total_dist = 0.0
    if len(traj_x) > 1:
        dx = np.diff(traj_x)
        dy = np.diff(traj_y)
        total_dist = float(np.sum(np.sqrt(dx**2 + dy**2)))

    print("\n=== Simulation Complete ===")
    print(f"  Duration         : {t_sim:.1f} s")
    print(f"  Distance covered : {total_dist:.1f} m")
    print(f"  Final speed      : {speed*3.6:.1f} km/h")
    print(f"  Collisions       : {log_data['collisions']}")
    print(f"  Replans          : {log_data['replans']}")
    if log_data["replan_latency_ms"]:
        print(f"  Avg replan ms    : {np.mean(log_data['replan_latency_ms']):.1f} ms")
    print(f"  STATUS           : {'GOAL REACHED' if arrived else 'TIME-OUT'}")

    # ── Metrics ───────────────────────────────────────────────────────────
    if collect_metrics_at_end and log_data["t"]:
        try:
            from simulation.collect_metrics import collect_metrics
            collect_metrics(log_data, "cosimulation", save_csv=True)
        except Exception as exc:
            print(f"  [WARN] collect_metrics failed: {exc}")

    # Keep plot open
    plt.ioff()
    plt.show()
    return log_data


# ===========================================================================
# CLI entry-point
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="ADAS Vision Python Co-simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Drive to (200, 0):
    python simulation/cosimulation.py --goal 200 0

  Custom start position and time limit:
    python simulation/cosimulation.py --goal 150 50 --max-time 90 \\
        --init-x 10 --init-y 5 --init-heading 10
        """,
    )
    p.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"),
                   required=True, help="Goal position in metres.")
    p.add_argument("--max-time", type=float, default=SIM.max_time_s,
                   help=f"Max simulation time (default: {SIM.max_time_s} s).")
    p.add_argument("--init-x", type=float, default=0.0)
    p.add_argument("--init-y", type=float, default=0.0)
    p.add_argument("--init-heading", type=float, default=0.0,
                   help="Initial heading in degrees (0 = east, 90 = north).")
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip the metrics summary and chart.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_cosimulation(
        goal_xy=tuple(args.goal),
        max_time=args.max_time,
        init_pos=(args.init_x, args.init_y),
        init_heading_deg=args.init_heading,
        collect_metrics_at_end=not args.no_metrics,
    )
