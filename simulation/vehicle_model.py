"""
vehicle_model.py — ADAS Vision Co-simulation Vehicle Parameters

Python equivalent of setup_vehicle_model.m.

All vehicle physics constants, controller gains, path-planner config, and
simulation parameters are defined here as frozen dataclasses. Import these
from any scenario or co-simulation script:

    from vehicle_model import VEH, LONG, PP, PID, PLANNER, SIM
    from vehicle_model import UDP_RX_PORT, UDP_TX_PORT, UDP_HOST

No global state and no workspace pollution — just plain Python objects.
"""

import math
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# UDP Network parameters (must match udp_bridge.py)
# ---------------------------------------------------------------------------
UDP_RX_PORT = 5005   # port that RECEIVES perception JSON from udp_bridge.py
UDP_TX_PORT = 5006   # port that SENDS ego position back to udp_bridge.py
UDP_HOST    = "127.0.0.1"
UDP_TIMEOUT = 0.5    # seconds to wait for a packet before reusing last value
UDP_BUFSIZE = 65535


# ---------------------------------------------------------------------------
# Vehicle geometry — kinematic bicycle model
# ---------------------------------------------------------------------------
@dataclass
class VehicleParams:
    """Geometric and tyre parameters for the kinematic bicycle model."""
    mass_kg:       float = 1450.0   # kerb mass (kg)
    wheelbase_m:   float = 2.70     # front-to-rear axle distance (m)
    lf_m:          float = 1.15     # CG-to-front axle (m)
    lr_m:          float = 1.55     # CG-to-rear  axle (m)
    width_m:       float = 1.85     # vehicle width (m)
    length_m:      float = 4.40     # vehicle length (m)
    cg_height_m:   float = 0.55     # CG height above ground (m)
    max_steer_deg: float = 35.0     # maximum steering angle (degrees)
    # Simplified linear tyre model
    Cf:            float = 80_000.0  # front cornering stiffness (N/rad)
    Cr:            float = 90_000.0  # rear  cornering stiffness (N/rad)
    Iz:            float = 2500.0    # yaw moment of inertia (kg·m²)

    @property
    def max_steer_rad(self) -> float:
        return math.radians(self.max_steer_deg)


# ---------------------------------------------------------------------------
# Longitudinal dynamics (simplified drag + rolling resistance)
# ---------------------------------------------------------------------------
@dataclass
class LongitudinalParams:
    """Simplified longitudinal force model and saturation limits."""
    Cd:             float = 0.30    # aerodynamic drag coefficient
    A_m2:           float = 2.20   # frontal area (m²)
    rho:            float = 1.225  # air density (kg/m³)
    Cr_roll:        float = 0.015  # rolling resistance coefficient
    g:              float = 9.81   # gravity (m/s²)
    max_accel_mps2: float = 2.5    # maximum forward acceleration (m/s²)
    max_brake_mps2: float = 7.0    # maximum braking deceleration (m/s²)
    max_speed_mps:  float = 27.8   # speed cap (~100 km/h)


# ---------------------------------------------------------------------------
# Pure-Pursuit lateral controller
# ---------------------------------------------------------------------------
@dataclass
class PurePursuitParams:
    """Adaptive pure-pursuit look-ahead parameters."""
    lookahead_m:   float = 8.0    # nominal look-ahead distance (m)
    min_lookahead: float = 4.0    # minimum (low-speed)
    max_lookahead: float = 20.0   # maximum (high-speed)
    k_lookahead:   float = 0.5    # velocity scaling factor


# ---------------------------------------------------------------------------
# Longitudinal PID speed controller
# ---------------------------------------------------------------------------
@dataclass
class PIDParams:
    """PID gains and integrator limits for speed control."""
    Kp:      float = 0.60
    Ki:      float = 0.05
    Kd:      float = 0.00
    dt:      float = 0.033   # control step (~30 Hz)
    int_max: float = 5.0     # anti-windup integrator clamp


# ---------------------------------------------------------------------------
# Hybrid A* path planner
# ---------------------------------------------------------------------------
@dataclass
class PlannerParams:
    """Configuration for the Hybrid A* path planner."""
    grid_res_m:     float = 0.5    # grid cell resolution (m)
    heading_bins:   int   = 72     # heading discretisation (360/72 = 5 °/bin)
    steer_angles:   List[float] = field(
        default_factory=lambda: [-25.0, -15.0, -5.0, 0.0, 5.0, 15.0, 25.0])
    step_m:         float = 1.0    # arc length per node expansion (m)
    obstacle_rad_m: float = 2.0    # obstacle inflation radius (m)
    heuristic_w:    float = 1.5    # weighted A* factor (>1 = faster, suboptimal)
    max_iters:      int   = 5000   # maximum node expansions
    replan_dist_m:  float = 3.0    # replan when ego drifts this far off path (m)


# ---------------------------------------------------------------------------
# Simulation control parameters
# ---------------------------------------------------------------------------
@dataclass
class SimParams:
    """Top-level simulation timing and goal/collision thresholds."""
    dt_s:            float = 0.033   # simulation timestep (s), ~30 Hz
    max_time_s:      float = 120.0   # maximum run duration (s)
    init_speed_mps:  float = 8.33    # initial ego speed (~30 km/h)
    goal_tol_m:      float = 3.0     # distance at which goal is "reached"
    collision_rad_m: float = 1.5     # ego body radius for collision detection


# ---------------------------------------------------------------------------
# Singleton default instances — import and use directly
# ---------------------------------------------------------------------------
VEH     = VehicleParams()
LONG    = LongitudinalParams()
PP      = PurePursuitParams()
PID     = PIDParams()
PLANNER = PlannerParams()
SIM     = SimParams()


# ---------------------------------------------------------------------------
# Pretty-print summary (mirrors setup_vehicle_model console output)
# ---------------------------------------------------------------------------
def print_summary() -> None:
    """Print the loaded parameter summary to stdout."""
    print("=== ADAS Vision — Vehicle Model (Python) ===")
    print(f"  Vehicle        : {VEH.mass_kg:.0f} kg, {VEH.wheelbase_m:.2f} m wheelbase")
    print(f"  Controller     : Pure-Pursuit ({PP.lookahead_m:.1f} m lookahead)"
          f" + PID (Kp={PID.Kp:.2f})")
    print(f"  Planner        : Hybrid A* ({PLANNER.grid_res_m:.1f} m grid,"
          f" {len(PLANNER.steer_angles)} steer angles)")
    print(f"  UDP RX         : port {UDP_RX_PORT}   TX: port {UDP_TX_PORT}")
    print("  All parameters ready.")
    print(f"  Next step: python simulation/cosimulation.py --goal 200 0\n")


if __name__ == "__main__":
    print_summary()
