"""
collect_metrics.py — ADAS Vision Performance Metrics Collector

Python equivalent of collect_metrics.m.

Computes comprehensive performance metrics from a simulation log_data dict
and optionally saves them to a CSV and shows a bar-chart summary figure.

Usage:
    from simulation.collect_metrics import collect_metrics

    metrics = collect_metrics(log_data, "village_road")
    metrics = collect_metrics(log_data, "cattle_crossing", save_csv=False)

The log_data dict should have the same structure as the one built by
cosimulation.py or the standalone scenario scripts.
"""

import csv
import math
import os
from datetime import datetime
from typing import Dict, Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_metrics(
    log_data: dict,
    scenario_name: str,
    save_csv:     bool = True,
    out_dir:      str  = ".",
    plot_summary: bool = True,
) -> Dict[str, Any]:
    """
    Compute and report all performance metrics for a finished simulation run.

    Args:
        log_data:      Dict produced by cosimulation.py / scenario scripts.
        scenario_name: Label for this run (used in filenames and chart titles).
        save_csv:      Write a ``metrics_<scenario>.csv`` file if True.
        out_dir:       Directory for CSV output (default: current directory).
        plot_summary:  Display a bar-chart summary figure if True.

    Returns:
        Dict of metric name → value.
    """
    print(f"=== collect_metrics: {scenario_name} ===")

    m: Dict[str, Any] = {
        "scenario":  scenario_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    t_arr    = np.asarray(log_data.get("t",          []))
    x_arr    = np.asarray(log_data.get("x",          []))
    y_arr    = np.asarray(log_data.get("y",          []))
    spd_arr  = np.asarray(log_data.get("speed_mps",  []))
    rl_arr   = np.asarray(log_data.get("replan_latency_ms", []))
    dec_list = log_data.get("decision", [])

    # ------------------------------------------------------------------
    # 1. Basic stats
    # ------------------------------------------------------------------
    if len(t_arr) > 0:
        m["duration_s"]  = float(t_arr[-1] - t_arr[0])
        m["n_frames"]    = int(len(t_arr))
        m["control_hz"]  = m["n_frames"] / max(m["duration_s"], 1e-6)
    else:
        m["duration_s"] = m["n_frames"] = m["control_hz"] = 0.0

    # ------------------------------------------------------------------
    # 2. Distance travelled
    # ------------------------------------------------------------------
    if len(x_arr) > 1:
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        m["distance_m"] = float(np.sum(np.sqrt(dx**2 + dy**2)))
    else:
        m["distance_m"] = 0.0

    # ------------------------------------------------------------------
    # 3. Path smoothness — mean absolute curvature (rad/m)
    # ------------------------------------------------------------------
    m["path_smoothness"] = _compute_smoothness(x_arr, y_arr)

    # ------------------------------------------------------------------
    # 4. Speed statistics
    # ------------------------------------------------------------------
    if len(spd_arr) > 0:
        m["mean_speed_kmh"] = float(np.mean(spd_arr) * 3.6)
        m["max_speed_kmh"]  = float(np.max(spd_arr)  * 3.6)
        m["min_speed_kmh"]  = float(np.min(spd_arr)  * 3.6)
        m["idle_fraction"]  = float(np.mean(spd_arr < 0.1))
    else:
        m["mean_speed_kmh"] = m["max_speed_kmh"] = \
        m["min_speed_kmh"]  = m["idle_fraction"] = 0.0

    # ------------------------------------------------------------------
    # 5. Replanning latency
    # ------------------------------------------------------------------
    if len(rl_arr) > 0:
        m["n_replans"]      = int(len(rl_arr))
        m["mean_replan_ms"] = float(np.mean(rl_arr))
        m["max_replan_ms"]  = float(np.max(rl_arr))
        m["p95_replan_ms"]  = float(np.percentile(rl_arr, 95))
    else:
        m["n_replans"] = m["mean_replan_ms"] = \
        m["max_replan_ms"] = m["p95_replan_ms"] = 0.0

    # ------------------------------------------------------------------
    # 6. Collisions
    # ------------------------------------------------------------------
    m["collisions"] = int(log_data.get("collisions", 0))

    # ------------------------------------------------------------------
    # 7. Decision distribution
    # ------------------------------------------------------------------
    levels = ["PROCEED", "CAUTION", "SLOW", "BRAKE", "EMERGENCY_STOP"]
    n_dec  = max(len(dec_list), 1)
    dist   = {lv: sum(1 for d in dec_list if d == lv) / n_dec for lv in levels}
    m["decision_dist"]  = dist
    m["proceed_frac"]   = dist.get("PROCEED", 0.0)

    # ------------------------------------------------------------------
    # 8. Minimum obstacle clearance
    # ------------------------------------------------------------------
    min_dist_arr = log_data.get("min_dist_m", [])
    if len(min_dist_arr) > 0:
        m["min_clearance_m"] = float(np.min(min_dist_arr))
    else:
        m["min_clearance_m"] = 5.0   # default when not tracked

    # ------------------------------------------------------------------
    # 9. Lateral control error (RMS Y deviation — valid for straight-road runs)
    # ------------------------------------------------------------------
    if len(y_arr) > 1:
        m["lateral_rms_m"] = float(math.sqrt(float(np.mean(y_arr**2))))
    else:
        m["lateral_rms_m"] = 0.0

    # ------------------------------------------------------------------
    # 10. Braking harshness
    # ------------------------------------------------------------------
    m["max_decel_mps2"] = 0.0
    m["mean_jerk"]      = 0.0
    if len(spd_arr) > 2 and len(t_arr) > 2:
        dt_arr = np.diff(t_arr)
        dt_arr = np.where(dt_arr < 1e-6, 1e-6, dt_arr)
        accel  = np.diff(spd_arr) / dt_arr
        decel  = -accel[accel < 0]
        if len(decel) > 0:
            m["max_decel_mps2"] = float(np.max(decel))
        if len(accel) > 1:
            jerk = np.diff(accel) / dt_arr[:-1]
            m["mean_jerk"] = float(np.mean(np.abs(jerk)))

    # ------------------------------------------------------------------
    # 11. Scenario completion
    # ------------------------------------------------------------------
    m["scenario_completed"] = int(bool(log_data.get("scenario_completed", False)))

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _print_summary(m)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if save_csv:
        _save_csv(m, out_dir)

    # ------------------------------------------------------------------
    # Plot summary figure
    # ------------------------------------------------------------------
    if plot_summary:
        _plot_summary(m)

    print("\n  collect_metrics done.")
    return m


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_smoothness(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """Mean absolute curvature (rad/m). Lower = smoother path."""
    if len(x_arr) < 3:
        return 0.0
    dx1 = np.diff(x_arr)
    dy1 = np.diff(y_arr)
    dx2 = np.diff(dx1)
    dy2 = np.diff(dy1)
    ds  = np.sqrt(dx1[:-1]**2 + dy1[:-1]**2) + 1e-9
    kap = np.abs(dx1[:-1] * dy2 - dy1[:-1] * dx2) / (ds**3 + 1e-9)
    return float(np.mean(kap))


def _print_summary(m: dict) -> None:
    print("\n--- Metrics Summary ---")
    print(f"  Scenario          : {m['scenario']}")
    print(f"  Duration          : {m['duration_s']:.1f} s")
    print(f"  Distance          : {m['distance_m']:.1f} m")
    print(f"  Mean speed        : {m['mean_speed_kmh']:.1f} km/h")
    print(f"  Path smoothness   : {m['path_smoothness']:.4f} rad/m (lower = smoother)")
    print(f"  Lateral RMS       : {m['lateral_rms_m']:.2f} m")
    print(f"  Min clearance     : {m['min_clearance_m']:.1f} m")
    print(f"  Collisions        : {m['collisions']}")
    print(f"  Replans           : {m['n_replans']}  "
          f"(mean {m['mean_replan_ms']:.1f} ms, P95 {m['p95_replan_ms']:.1f} ms)")
    print(f"  Max decel         : {m['max_decel_mps2']:.2f} m/s²")
    print(f"  Idle fraction     : {m['idle_fraction']*100:.0f}%")
    print(f"  PROCEED fraction  : {m['proceed_frac']*100:.0f}%")
    print(f"  Completed         : {'YES' if m['scenario_completed'] else 'NO'}")


def _save_csv(m: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir,
                         f"metrics_{m['scenario'].replace(' ', '_')}.csv")
    rows = [
        ("metric",            "value",                     "unit"),
        ("scenario",          m["scenario"],               ""),
        ("timestamp",         m["timestamp"],              ""),
        ("duration_s",        f"{m['duration_s']:.2f}",   "s"),
        ("distance_m",        f"{m['distance_m']:.2f}",   "m"),
        ("mean_speed_kmh",    f"{m['mean_speed_kmh']:.2f}","km/h"),
        ("max_speed_kmh",     f"{m['max_speed_kmh']:.2f}","km/h"),
        ("path_smoothness",   f"{m['path_smoothness']:.5f}","rad/m"),
        ("lateral_rms_m",     f"{m['lateral_rms_m']:.3f}","m"),
        ("min_clearance_m",   f"{m['min_clearance_m']:.2f}","m"),
        ("collisions",        str(m["collisions"]),        "count"),
        ("n_replans",         str(m["n_replans"]),         "count"),
        ("mean_replan_ms",    f"{m['mean_replan_ms']:.2f}","ms"),
        ("p95_replan_ms",     f"{m['p95_replan_ms']:.2f}","ms"),
        ("max_decel_mps2",    f"{m['max_decel_mps2']:.3f}","m/s2"),
        ("idle_fraction",     f"{m['idle_fraction']:.3f}","frac"),
        ("proceed_fraction",  f"{m['proceed_frac']:.3f}","frac"),
        ("scenario_completed",str(m["scenario_completed"]),"bool"),
    ]
    try:
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
        print(f"  CSV saved: {fname}")
    except OSError as e:
        print(f"[WARN] CSV save failed: {e}")


def _plot_summary(m: dict) -> None:
    labels = [
        "Speed\n(km/h)", "Smoothness\n(×100)", "Clearance\n(m)",
        "Replan\n(ms)", "Idle\n(%)", "PROCEED\n(%)",
    ]
    vals = [
        m["mean_speed_kmh"],
        m["path_smoothness"] * 100,
        m["min_clearance_m"],
        m["mean_replan_ms"],
        m["idle_fraction"]  * 100,
        m["proceed_frac"]   * 100,
    ]
    colors = [
        "#33CC6E", "#3399FF", "#FFAA33",
        "#E05555", "#CC88FF", "#55DDDD",
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#141418")
    ax.set_facecolor("#1E1E26")
    for spine in ax.spines.values():
        spine.set_edgecolor("#404050")

    x_pos = range(len(vals))
    bars = ax.bar(x_pos, vals, color=colors, edgecolor="none", width=0.55)

    # Value annotations above each bar
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(vals) * 0.02,
                f"{v:.1f}", ha="center", va="bottom",
                color="white", fontsize=9)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.tick_params(axis="y", colors="white")
    ax.yaxis.label.set_color("white")
    ax.set_ylabel("Value (units per label)", color="white")
    ax.grid(axis="y", color="#303040", alpha=0.6)

    col_str = "#E03333" if m["collisions"] > 0 else "#33CC55"
    col_txt = (f"⚠ {m['collisions']} COLLISION(S)"
               if m["collisions"] > 0 else "✓ NO COLLISIONS")
    ax.text(0.98, 0.95, col_txt, transform=ax.transAxes,
            ha="right", va="top", color=col_str, fontsize=11, fontweight="bold")

    ax.set_title(
        f"ADAS Metrics — {m['scenario']}  |  "
        f"Collisions: {m['collisions']}  |  Replans: {m['n_replans']}",
        color="white", fontsize=11, pad=10,
    )
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
