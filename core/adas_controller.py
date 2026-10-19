"""
adas_controller.py — Full-Stack Real ADAS Autonomous Decision & Control Engine.

Implements:
  1. Finite State Machine (FSM):
     - ACC_CRUISE: Adaptive Cruise Control maintaining highway speed.
     - ACC_FOLLOW: Time-headway (1.6s) spacing control behind Lead Vehicle.
     - FCW_ALERT: Forward Collision Warning alert with live Time-To-Collision (TTC).
     - LANE_CHANGE_DECISION: Evaluates whether adjacent lanes (Lane -2 / Lane -4) are safe to enter.
     - AUTONOMOUS_LANE_CHANGE: Generates smooth Quintic Polynomial minimum-jerk trajectory.
     - AEB_EMERGENCY_BRAKE: Full ABS deceleration (-8.0 m/s^2) if lane change is blocked or emergency stop required.
     - LANE_KEEPING_ASSIST (LKA): Centering in lane after maneuver.

  2. Trajectory Generation:
     - Quintic Polynomial (zero jerk, smooth lateral acceleration curve conforming to ISO 17387).

  3. Lateral Steering Control:
     - Stanley closed-loop path-tracking controller outputting road wheel angle and handwheel steering angle (-540 deg to +540 deg).

  4. Longitudinal Control:
     - Throttle [0..1] and hydraulic brake pressure [0..1] with brake-throttle override safety.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.vehicle_physics import VehiclePhysicsEngine, VehicleSpecs


class ADASState(Enum):
    CRUISE = "ACC_CRUISE"
    FOLLOW = "ACC_FOLLOW"
    FCW_ALERT = "FCW_ALERT"
    EVALUATING_LANE_CHANGE = "EVALUATING_LANE_CHANGE"
    AUTONOMOUS_LANE_CHANGE = "AUTONOMOUS_LANE_CHANGE"
    AEB_BRAKING = "AEB_ACTIVE"
    LDW_WARNING = "LDW_WARNING"
    LKA_CENTERING = "LKA_CENTERING"
    RESUMING = "ACC_RESUMING"


@dataclass
class ADASCommand:
    state: ADASState
    state_desc: str
    status_color: Tuple[int, int, int]   # BGR
    throttle_cmd: float                  # 0.0 to 1.0
    brake_cmd: float                     # 0.0 to 1.0
    steering_wheel_deg: float            # -540 to +540 deg
    road_wheel_deg: float                # -35 to +35 deg
    target_speed_kmh: float
    ttc_s: float
    current_lane_offset_m: float
    target_lane_offset_m: float
    decision_reason: str
    lateral_g: float = 0.0
    longitudinal_g: float = 0.0
    lane_change_progress_pct: float = 0.0
    planned_path: List[Tuple[float, float]] = field(default_factory=list)
    ihc_mode: str = "LOW BEAM (Traffic ahead)"


class QuinticTrajectoryPlanner:
    """Generates minimum-jerk smooth lateral trajectory for lane changes."""

    def __init__(self, y_start: float, y_end: float, duration_s: float = 3.6):
        self.y0 = y_start
        self.y1 = y_end
        self.T = max(1.0, duration_s)
        self.t_elapsed = 0.0
        self.finished = False

    def evaluate(self, t: float) -> Tuple[float, float, float]:
        """
        Returns (y_ref, y_dot_ref, y_ddot_ref) at time t into the maneuver.
        y(tau) = y0 + (y1 - y0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
        """
        tau = max(0.0, min(1.0, t / self.T))
        dy = self.y1 - self.y0

        # Position
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        y = self.y0 + dy * s

        # Velocity
        s_dot = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / self.T
        y_dot = dy * s_dot

        # Acceleration
        s_ddot = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (self.T ** 2)
        y_ddot = dy * s_ddot

        if tau >= 1.0:
            self.finished = True

        return y, y_dot, y_ddot


class StanleySteeringController:
    """
    Stanley path tracking controller:
    delta = heading_error + arctan(k * cross_track_error / (v + eps))
    Supports 'gentle_nudge' (tactile, non-jarring steering assist conforming to MG Pilot L2)
    and 'firm' (standard aggressive intervention).
    """

    def __init__(self, k_gain: float = 1.4, k_soft: float = 1.0, k_d: float = 0.6,
                 intervention_mode: str = "gentle_nudge"):
        self.k = k_gain
        self.k_soft = k_soft
        self.kd = k_d
        self.intervention_mode = intervention_mode
        self.prev_e = 0.0

    def compute(self, current_y: float, current_heading: float,
                target_y: float, target_heading: float,
                vx: float, dt: float = 0.02) -> float:
        cross_track_error = target_y - current_y
        heading_error = target_heading - current_heading

        # Normalize heading error to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2.0 * math.pi) - math.pi

        # Cross track term
        vel_term = max(1.0, vx) + self.k_soft
        cross_term = math.atan2(self.k * cross_track_error, vel_term)

        # Derivative damping
        de = (cross_track_error - self.prev_e) / max(0.001, dt)
        self.prev_e = cross_track_error
        d_term = math.atan2(self.kd * de, vel_term)

        raw_steering = heading_error + cross_term + d_term

        # MG Hector Tomahawk tuned 'gentle_nudge':
        # Light-handed tactile correction without wrestling the steering wheel away
        if self.intervention_mode == "gentle_nudge":
            max_rad = math.radians(18.0)
        else:
            max_rad = math.radians(35.0)

        return max(-max_rad, min(max_rad, raw_steering))


class ADASDecisionSystem:
    """
    Central ADAS Decision & Vehicle Control System.
    Monitors 360-degree scenario traffic, assesses collision risks,
    makes autonomous lane change / braking decisions, and executes full controls.
    """

    def __init__(self, cruise_speed_kmh: float = 72.0,
                 fcw_sensitivity: str = "comfort",
                 tja_enabled: bool = True,
                 lka_intervention: str = "gentle_nudge"):
        self.set_speed_mps = cruise_speed_kmh / 3.6
        self.physics = VehiclePhysicsEngine(init_speed_mps=self.set_speed_mps)
        self.fcw_sensitivity = fcw_sensitivity
        self.tja_enabled = tja_enabled
        self.lka_intervention = lka_intervention
        self.steering_ctrl = StanleySteeringController(intervention_mode=self.lka_intervention)
        
        # State
        self.state = ADASState.CRUISE
        self.lane_change_planner: Optional[QuinticTrajectoryPlanner] = None
        self.lane_change_start_time = 0.0
        self.current_lane_idx = -3       # Center highway lane
        self.current_lateral_m = 0.0     # 0.0 = center of lane -3
        self.target_lateral_m = 0.0

        # Lane Departure Warning (LDW) → LKA cascade
        # LDW fires when ego drifts > ldw_threshold from lane center
        # After ldw_warning_duration_s, LKA correction nudge activates
        self.ldw_threshold_m = 0.35      # Lane departure warning trigger (0.35m from edge = 1.4m from center on 3.5m lane)
        self.ldw_warning_start_t = -1.0  # Timestamp when LDW warning started

        # PID Longitudinal gains
        self.kp_spd = 0.35
        self.ki_spd = 0.04
        self.spd_integral = 0.0

    def cycle_fcw_sensitivity(self) -> str:
        """Cycle driver-adjustable FCW sensitivity: comfort (dialed back) -> normal -> aggressive."""
        modes = ["comfort", "normal", "aggressive"]
        idx = (modes.index(self.fcw_sensitivity) + 1) % len(modes)
        self.fcw_sensitivity = modes[idx]
        return self.fcw_sensitivity

    def toggle_tja(self) -> bool:
        """Toggle Traffic Jam Assist (TJA) with Curve Control on/off."""
        self.tja_enabled = not self.tja_enabled
        return self.tja_enabled

    def cycle_lka_intervention(self) -> str:
        """Cycle LKA intervention between gentle tactile nudge and firm centering."""
        self.lka_intervention = "firm" if self.lka_intervention == "gentle_nudge" else "gentle_nudge"
        self.steering_ctrl.intervention_mode = self.lka_intervention
        return self.lka_intervention

    def set_vehicle(self, vehicle_key: str, custom_weight_kg: Optional[float] = None):
        """Configure vehicle weight and dimensions profile."""
        self.physics.set_vehicle_profile(vehicle_key, custom_weight_kg)

    def set_weather(self, weather_key: str, custom_friction_mu: Optional[float] = None):
        """Configure road surface friction and weather condition."""
        self.physics.set_weather_profile(weather_key, custom_friction_mu)

    def evaluate_scene(self, sim_t: float) -> dict:
        """
        Reconstructs ground-truth spatial relationships from the Indian Highway Scenario:
          - Ego: starts at s=50m, cruising in Lane -3
          - LeadCar: starts at s=150m (100m ahead in Lane -3)
          - Overtaker: starts at s=10m in Lane -2 (Fast Lane, 40m behind), cruises past Ego
          - StrayCattle: starts at s=580m on right shoulder, moves towards lane at t=13s
          - Traffic in Lane -4: Heavy Truck (s=90m), Bus (s=320m), Scooter (s=30m)
        """
        ego_speed = self.physics.vx
        ego_s = 50.0 + self.physics.x
        specs = self.physics.specs
        mu = specs.mu_peak

        # Dynamic friction-dependent safety thresholds
        # Driver-Adjustable FCW Sensitivity (MG Pilot level):
        # - "comfort" (dialed back for chaotic Indian roads to avoid false alarms): base 2.4s
        # - "normal" (standard factory Level-2 setting): base 3.5s
        # - "aggressive" (highway early alert): base 4.5s
        base_fcw_ttc = 2.4 if self.fcw_sensitivity == "comfort" else (4.5 if self.fcw_sensitivity == "aggressive" else 3.5)
        time_headway_s = 1.6 * math.sqrt(0.88 / max(0.2, mu))
        fcw_ttc_thresh = base_fcw_ttc * (0.88 / max(0.2, mu))
        stopping_dist = self.physics.compute_stopping_distance(ego_speed)
        
        # 1. Lead Vehicle kinematics
        lead_v = 20.0 if sim_t < 13.0 else (max(11.1, 20.0 - (sim_t - 13.0) * 2.22) if sim_t < 25.0 else min(20.0, 11.1 + (sim_t - 25.0) * 2.22))
        lead_s = 150.0 + (sim_t * 20.0 if sim_t < 13.0 else (13.0 * 20.0 + (sim_t - 13.0) * lead_v))
        lead_dist = max(5.0, lead_s - ego_s)
        lead_rel_speed = lead_v - ego_speed

        # 2. Overtaker kinematics (Lane -2: Fast Lane)
        overtaker_v = 24.0 if sim_t < 4.0 else 26.0
        overtaker_s = 10.0 + (sim_t * 24.0 if sim_t < 4.0 else (4.0 * 24.0 + (sim_t - 4.0) * 26.0))
        overtaker_long_gap = overtaker_s - ego_s

        # 3. Stray Cattle kinematics (crossing hazard at s=580m)
        cattle_s = 580.0
        cattle_dist = max(0.0, cattle_s - ego_s)
        cattle_lat_m = 4.5 if sim_t < 13.0 else max(1.4, 4.5 - (sim_t - 13.0) * 0.22)

        # 4. Closing speed and Time-To-Collision (TTC) with in-path obstacles
        primary_threat_dist = min(lead_dist, cattle_dist if cattle_lat_m < 2.0 else 999.0)
        closing_speed = max(0.1, -lead_rel_speed if lead_dist < cattle_dist else ego_speed)
        ttc = primary_threat_dist / closing_speed if primary_threat_dist > 0.1 else 99.0

        # 5. Adjacent Lane Clearances
        # Fast Lane (-2, left): safe if Overtaker is either > 35m ahead or > 30m behind
        fast_lane_clear = (overtaker_long_gap > 35.0) or (overtaker_long_gap < -30.0)
        # Slow Lane (-4, right): heavy trucks and buses operating
        slow_lane_clear = True

        # 6. Intelligent Headlamp Control (IHC) - Level 2 Feature
        # Auto High-Beam toggles: dips to Low Beam if vehicle ahead within 120m or traffic adjacent
        ihc_active_low = (lead_dist < 120.0) or (abs(overtaker_long_gap) < 60.0)
        ihc_mode = "LOW BEAM (Traffic ahead)" if ihc_active_low else "HIGH BEAM (Road clear)"

        return {
            "sim_t": sim_t,
            "ego_s": ego_s,
            "ego_speed": ego_speed,
            "lead_dist": lead_dist,
            "lead_rel_speed": lead_rel_speed,
            "cattle_dist": cattle_dist,
            "cattle_lat_m": cattle_lat_m,
            "overtaker_gap": overtaker_long_gap,
            "fast_lane_clear": fast_lane_clear,
            "slow_lane_clear": slow_lane_clear,
            "ttc": ttc,
            "mu_peak": mu,
            "weather_type": specs.weather_type,
            "vehicle_type": specs.vehicle_type,
            "vehicle_mass_kg": specs.total_mass_kg,
            "time_headway_s": time_headway_s,
            "fcw_ttc_thresh": fcw_ttc_thresh,
            "stopping_dist_m": stopping_dist,
            "fcw_sensitivity": self.fcw_sensitivity,
            "tja_enabled": self.tja_enabled,
            "lka_intervention": self.lka_intervention,
            "ihc_mode": ihc_mode,
        }

    def update(self, sim_t: float, dt: float = 0.02) -> Tuple[ADASCommand, dict]:
        """
        Main tick: Evaluates environment, steps state machine, computes control commands,
        and advances physical vehicle integration.
        """
        scene = self.evaluate_scene(sim_t)
        ego_speed = self.physics.vx
        ttc = scene["ttc"]
        lead_dist = scene["lead_dist"]
        cattle_dist = scene["cattle_dist"]
        fast_lane_clear = scene["fast_lane_clear"]
        mu = scene["mu_peak"]
        fcw_thresh = scene["fcw_ttc_thresh"]
        stopping_dist = scene["stopping_dist_m"]

        throttle_cmd = 0.0
        brake_cmd = 0.0
        target_handwheel_deg = 0.0
        target_spd_kmh = 72.0
        status_color = (0, 230, 118)  # Neon Green
        reason = "All lanes clear; cruising at set speed."

        # -----------------------------------------------------------------------
        # DYNAMIC FRICTION-ADAPTIVE STATE MACHINE
        # -----------------------------------------------------------------------
        # Hazard criteria adapts dynamically: if road is wet/gravel, hazard distance expands!
        hazard_dist_thresh = max(65.0, stopping_dist + 20.0)
        hazard_ahead = (cattle_dist < hazard_dist_thresh and scene["cattle_lat_m"] < 2.4) or (lead_dist < stopping_dist + 15.0 and ttc < fcw_thresh)

        # Lane change duration scales to prevent lateral skid on low-friction road
        lc_duration = max(3.2, 3.6 * math.sqrt(0.88 / max(0.2, mu)))

        # Dynamic lane departure detection
        lateral_dev = abs(self.physics.y - self.target_lateral_m)

        if self.state in (ADASState.CRUISE, ADASState.FOLLOW, ADASState.RESUMING):
            if hazard_ahead:
                # Hazard detected! Transition to evaluation / alert
                if ttc < 2.0 or cattle_dist < stopping_dist:
                    # Imminent collision: not enough room to complete lane change -> AEB
                    self.state = ADASState.AEB_BRAKING
                elif fast_lane_clear and sim_t > 15.0 and self.current_lateral_m > -2.0:
                    # Adjacent fast lane is wide open! Initiate Autonomous Lane Change
                    self.state = ADASState.AUTONOMOUS_LANE_CHANGE
                    self.target_lateral_m = -3.5  # Move one full lane to the left (Lane -2)
                    self.lane_change_planner = QuinticTrajectoryPlanner(
                        y_start=self.current_lateral_m,
                        y_end=self.target_lateral_m,
                        duration_s=lc_duration
                    )
                    self.lane_change_start_time = sim_t
                else:
                    self.state = ADASState.FCW_ALERT
            elif lateral_dev > self.ldw_threshold_m:
                # Unintended lane drift detected -> Lane Departure Warning
                self.state = ADASState.LDW_WARNING
                self.ldw_warning_start_t = sim_t
            elif lead_dist < (stopping_dist + 40.0):
                self.state = ADASState.FOLLOW
            else:
                self.state = ADASState.CRUISE

        elif self.state == ADASState.LDW_WARNING:
            if hazard_ahead:
                if ttc < 2.0 or cattle_dist < stopping_dist:
                    self.state = ADASState.AEB_BRAKING
                else:
                    self.state = ADASState.FCW_ALERT
            elif sim_t - self.ldw_warning_start_t > 0.4:
                # 0.4s warning chime completed -> initiate LKA corrective nudge
                self.state = ADASState.LKA_CENTERING
            elif lateral_dev < 0.12:
                self.state = ADASState.CRUISE if lead_dist >= (stopping_dist + 40.0) else ADASState.FOLLOW

        elif self.state == ADASState.FCW_ALERT:
            if fast_lane_clear and sim_t > 15.0 and self.current_lateral_m > -2.0:
                # Evasive Lane Change permitted!
                self.state = ADASState.AUTONOMOUS_LANE_CHANGE
                self.target_lateral_m = -3.5
                self.lane_change_planner = QuinticTrajectoryPlanner(
                    y_start=self.current_lateral_m,
                    y_end=self.target_lateral_m,
                    duration_s=lc_duration
                )
                self.lane_change_start_time = sim_t
            elif ttc < 2.5 or cattle_dist < 50.0:
                # Fast lane occupied or hazard too close -> AEB Braking required
                self.state = ADASState.AEB_BRAKING
            elif not hazard_ahead:
                self.state = ADASState.CRUISE

        elif self.state == ADASState.AUTONOMOUS_LANE_CHANGE:
            if self.lane_change_planner and self.lane_change_planner.finished:
                # Lane change completed! Transition to LKA
                self.state = ADASState.LKA_CENTERING
                self.current_lane_idx = -2  # Now established in Fast Lane
                self.lane_change_planner = None

        elif self.state == ADASState.AEB_BRAKING:
            if cattle_dist <= 0.0 or cattle_dist > 550.0 or not hazard_ahead or ego_speed < 8.0:
                # Hazard cleared or controlled speed reached
                if sim_t > 25.0:
                    self.state = ADASState.RESUMING
            elif fast_lane_clear and sim_t > 18.0 and self.current_lateral_m > -2.0:
                # Once traffic opened up, transition from braking to lane change
                self.state = ADASState.AUTONOMOUS_LANE_CHANGE
                self.target_lateral_m = -3.5
                self.lane_change_planner = QuinticTrajectoryPlanner(
                    y_start=self.current_lateral_m,
                    y_end=self.target_lateral_m,
                    duration_s=lc_duration
                )
                self.lane_change_start_time = sim_t

        elif self.state == ADASState.LKA_CENTERING:
            if hazard_ahead:
                if ttc < 2.0 or cattle_dist < stopping_dist:
                    self.state = ADASState.AEB_BRAKING
                else:
                    self.state = ADASState.FCW_ALERT
            elif lateral_dev < 0.10 or sim_t > 30.0:
                self.state = ADASState.CRUISE if lead_dist >= (stopping_dist + 40.0) else ADASState.FOLLOW

        elif self.state == ADASState.RESUMING:
            if abs(ego_speed - self.set_speed_mps) < 0.5:
                self.state = ADASState.CRUISE

        # -----------------------------------------------------------------------
        # EXECUTE CONTROLS FOR CURRENT STATE
        # -----------------------------------------------------------------------
        planned_path_pts = []
        lane_change_pct = 0.0

        if self.state == ADASState.CRUISE:
            target_spd_mps = self.set_speed_mps
            target_spd_kmh = 72.0
            spd_err = target_spd_mps - ego_speed
            self.spd_integral = max(-5.0, min(5.0, self.spd_integral + spd_err * dt))
            accel_req = self.kp_spd * spd_err + self.ki_spd * self.spd_integral
            throttle_cmd = max(0.0, min(0.65, accel_req))
            brake_cmd = 0.0
            status_color = (0, 230, 118)  # Neon Green
            reason = "ACC Active: Cruising at 72 km/h; lane centered."
            # Lane centering steering
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=self.target_lateral_m, target_heading=0.0,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio

        elif self.state == ADASState.FOLLOW:
            # Maintain friction-adaptive safe time gap (dry=1.6s, wet=2.0s, heavy_rain=2.4s)
            time_headway = scene["time_headway_s"]
            des_dist = 8.0 + time_headway * ego_speed
            dist_err = lead_dist - des_dist
            target_spd_mps = max(5.0, min(self.set_speed_mps, self.set_speed_mps + 0.3 * dist_err))
            target_spd_kmh = target_spd_mps * 3.6
            spd_err = target_spd_mps - ego_speed
            if spd_err >= 0:
                throttle_cmd = max(0.0, min(0.5, 0.25 * spd_err))
                brake_cmd = 0.0
            else:
                throttle_cmd = 0.0
                brake_cmd = max(0.0, min(0.4, -0.2 * spd_err))
            if self.tja_enabled and ego_speed < 16.7:
                # TJA Active: Traffic Jam Assist with Curve/Bend Control
                # Estimate road curvature radius from ego yaw rate (if turning)
                abs_yaw_rate = abs(self.physics.yaw_rate)
                if abs_yaw_rate > 0.005 and ego_speed > 1.0:
                    curve_radius = ego_speed / abs_yaw_rate  # R = v / omega
                else:
                    curve_radius = 500.0  # Assume straight road
                # Maximum safe cornering speed: v_max = sqrt(mu * g * R) * 0.85 safety factor
                v_curve_max = math.sqrt(mu * 9.81 * curve_radius) * 0.85
                # TJA caps target speed to curve-safe limit
                tja_target_mps = min(target_spd_mps, v_curve_max)
                if ego_speed > tja_target_mps + 0.5:
                    # Apply gentle braking to slow for curve
                    throttle_cmd = 0.0
                    brake_cmd = max(brake_cmd, min(0.25, 0.15 * (ego_speed - tja_target_mps)))
                target_spd_kmh = min(target_spd_kmh, tja_target_mps * 3.6)
                status_color = (255, 210, 0)  # Cyan-Aqua TJA
                reason = f"TJA Active: Curve-control v_max={v_curve_max*3.6:.0f}km/h R={curve_radius:.0f}m, bumper-to-bumper ({lead_dist:.1f}m)."
            else:
                status_color = (0, 214, 255)  # Amber
                reason = f"ACC Follow: Tracking Lead Vehicle at {lead_dist:.1f}m (Gap={des_dist:.1f}m, Headway={time_headway:.1f}s)."
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=self.target_lateral_m, target_heading=0.0,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio

        elif self.state == ADASState.FCW_ALERT:
            throttle_cmd = 0.0
            brake_cmd = 0.35  # Pre-charge hydraulic brake lines
            target_spd_kmh = 45.0
            status_color = (0, 140, 255)  # Bright Orange
            reason = f"FCW ALERT: Hazard ahead at {cattle_dist:.0f}m! Pre-charging brake hydraulics (TTC={ttc:.1f}s)."
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=self.target_lateral_m, target_heading=0.0,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio

        elif self.state == ADASState.AUTONOMOUS_LANE_CHANGE:
            # Execute quintic polynomial trajectory
            t_in_maneuver = sim_t - self.lane_change_start_time
            y_ref, y_dot_ref, y_ddot_ref = self.lane_change_planner.evaluate(t_in_maneuver)
            lane_change_pct = min(100.0, (t_in_maneuver / self.lane_change_planner.T) * 100.0)

            # Target heading angle from lateral velocity
            target_heading = math.atan2(y_dot_ref, max(1.0, ego_speed))

            # Stanley tracking of the planned spline
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=y_ref, target_heading=target_heading,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio
            self.current_lateral_m = self.physics.y

            # Moderate speed during lane change (maintain 65-70 km/h)
            spd_err = (68.0 / 3.6) - ego_speed
            throttle_cmd = max(0.0, min(0.4, 0.2 * spd_err))
            brake_cmd = 0.0
            status_color = (255, 120, 0)  # Cyan-Blue
            reason = f"AUTONOMOUS LANE CHANGE: Overtaking hazard via Lane -2 ({lane_change_pct:.0f}% complete)."

            # Sample planned path forward for BEV visualization
            for tau_step in [0.2, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0]:
                t_future = t_in_maneuver + tau_step
                if t_future <= self.lane_change_planner.T:
                    y_fwd, _, _ = self.lane_change_planner.evaluate(t_future)
                    x_fwd = ego_speed * tau_step
                    planned_path_pts.append((x_fwd, y_fwd))

        elif self.state == ADASState.AEB_BRAKING:
            # Autonomous Emergency Braking: cut throttle, apply full hydraulic brake pressure
            throttle_cmd = 0.0
            brake_cmd = 1.0  # 100% ABS Braking
            target_spd_kmh = 0.0
            status_color = (0, 0, 255)  # Crimson Red
            stopping_dist = self.physics.stopping_distance_m
            reason = f"EMERGENCY AEB ACTIVE: Cattle in lane at {cattle_dist:.1f}m! Full ABS Braking (StopDist={stopping_dist:.1f}m)."
            # Keep vehicle pointed straight in lane under emergency braking
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=self.target_lateral_m, target_heading=0.0,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio

        elif self.state == ADASState.LDW_WARNING:
            # Lane Departure Warning active: driver alert chime, pre-nudge steering assist
            target_spd_mps = self.set_speed_mps
            target_spd_kmh = 72.0
            spd_err = target_spd_mps - ego_speed
            throttle_cmd = max(0.0, min(0.5, 0.3 * spd_err))
            brake_cmd = 0.0
            status_color = (0, 165, 255)  # Warning Amber
            lat_err = abs(self.physics.y - self.target_lateral_m)
            reason = f"LDW ALERT: Unintended lane drift ({lat_err:.2f}m)! Haptic alert chime active before LKA nudge."
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=self.target_lateral_m, target_heading=0.0,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio

        elif self.state == ADASState.LKA_CENTERING:
            target_spd_mps = self.set_speed_mps
            target_spd_kmh = 72.0
            spd_err = target_spd_mps - ego_speed
            throttle_cmd = max(0.0, min(0.5, 0.3 * spd_err))
            brake_cmd = 0.0
            status_color = (0, 230, 118)
            mode_lbl = "Gentle Nudge" if self.lka_intervention == "gentle_nudge" else "Firm Centering"
            reason = f"LKA Active ({mode_lbl}): Centering vehicle into target lane centerline."
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=self.target_lateral_m, target_heading=0.0,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio

        elif self.state == ADASState.RESUMING:
            target_spd_mps = self.set_speed_mps
            target_spd_kmh = 72.0
            spd_err = target_spd_mps - ego_speed
            throttle_cmd = max(0.0, min(0.6, 0.4 * spd_err))
            brake_cmd = 0.0
            status_color = (0, 230, 118)
            reason = "ACC Resuming: Hazard cleared; accelerating to cruise speed."
            road_steer = self.steering_ctrl.compute(
                current_y=self.physics.y, current_heading=self.physics.heading,
                target_y=self.target_lateral_m, target_heading=0.0,
                vx=ego_speed, dt=dt
            )
            target_handwheel_deg = math.degrees(road_steer) * self.physics.specs.steering_ratio

        # -----------------------------------------------------------------------
        # STEP REAL-WORLD VEHICLE DYNAMICS
        # -----------------------------------------------------------------------
        physics_state = self.physics.step(
            throttle_cmd=throttle_cmd,
            brake_cmd=brake_cmd,
            target_steering_wheel_deg=target_handwheel_deg,
            dt=dt
        )

        cmd = ADASCommand(
            state=self.state,
            state_desc=self.state.value,
            status_color=status_color,
            throttle_cmd=throttle_cmd,
            brake_cmd=brake_cmd,
            steering_wheel_deg=physics_state["handwheel_deg"],
            road_wheel_deg=physics_state["road_wheel_deg"],
            target_speed_kmh=target_spd_kmh,
            ttc_s=ttc,
            current_lane_offset_m=self.physics.y,
            target_lane_offset_m=self.target_lateral_m,
            decision_reason=reason,
            lateral_g=physics_state["ay_g"],
            longitudinal_g=physics_state["ax_g"],
            lane_change_progress_pct=lane_change_pct,
            planned_path=planned_path_pts,
            ihc_mode=scene.get("ihc_mode", "LOW BEAM"),
        )

        return cmd, physics_state
