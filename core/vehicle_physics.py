"""
vehicle_physics.py — High-Fidelity Real-World Vehicle Dynamics & Physics Engine.

Models the physics of an average modern passenger sedan (e.g., Honda City / Hyundai Verna / Skoda Slavia):
  1. Longitudinal Dynamics:
     - Engine traction force & transmission gear torque curves.
     - Hydraulic disc braking with ABS slip regulation and road-tire friction limits (mu = 0.85).
     - Aerodynamic drag force: F_aero = 0.5 * rho * Cd * A * v^2.
     - Rolling resistance: F_roll = Crr * m * g.
     - Dynamic normal load transfer under acceleration and braking:
       Fz_front = (m*g*lr - m*ax*hcg) / L
       Fz_rear  = (m*g*lf + m*ax*hcg) / L
     - Stopping distance computation including hydraulic lag time.

  2. Lateral Dynamics (Kinematic & Dynamic Bicycle Model):
     - Wheelbase L = 2.70 m, steering gear ratio 15.5:1.
     - Front & rear slip angles: alpha_f, alpha_r.
     - Cornering stiffness Cf = 85,000 N/rad, Cr = 95,000 N/rad.
     - Yaw rate (r), lateral velocity (vy), and lateral acceleration (ay = vy_dot + vx * r).
     - Handwheel steering angle (-540 deg to +540 deg) mapped to road wheel angle (-35 deg to +35 deg).
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Vehicle Weight & Class Presets
# ---------------------------------------------------------------------------
VEHICLE_PROFILES = {
    "hector_ev": {
        "name": "MG Hector Tomahawk EV (Essence)",
        "desc": "69.2 kWh MAGIC Battery, 201 bhp, 310 Nm, Level-2 ADAS",
        "kerb_mass_kg": 1840.0,
        "payload_kg": 100.0,
        "total_mass_kg": 1940.0,
        "wheelbase_m": 2.81,
        "lf_m": 1.32,
        "lr_m": 1.49,
        "cg_height_m": 0.53,
        "Cd": 0.32,
        "frontal_area_m2": 2.75,
        "steering_ratio": 14.4,
        "max_handwheel_deg": 504.0,
        "track_width_m": 1.61,
        "ground_clearance_mm": 230.0,
        "power_bhp": 201.0,
        "torque_nm": 310.0,
        "weight_dist_front_pct": 53.0,
    },
    "hector_phev": {
        "name": "MG Hector Tomahawk PHEV (Essence)",
        "desc": "1.5L Atkinson + Motor, 20.5 kWh, 198 bhp, 237 Nm",
        "kerb_mass_kg": 1780.0,
        "payload_kg": 100.0,
        "total_mass_kg": 1880.0,
        "wheelbase_m": 2.81,
        "lf_m": 1.26,
        "lr_m": 1.55,
        "cg_height_m": 0.55,
        "Cd": 0.32,
        "frontal_area_m2": 2.75,
        "steering_ratio": 14.4,
        "max_handwheel_deg": 504.0,
        "track_width_m": 1.61,
        "ground_clearance_mm": 230.0,
        "power_bhp": 198.0,
        "torque_nm": 237.0,
        "weight_dist_front_pct": 55.0,
    },
    "sedan": {
        "name": "C-Segment Sedan",
        "desc": "Honda City / Hyundai Verna class",
        "kerb_mass_kg": 1450.0,
        "payload_kg": 100.0,
        "total_mass_kg": 1550.0,
        "wheelbase_m": 2.70,
        "lf_m": 1.10,
        "lr_m": 1.60,
        "cg_height_m": 0.52,
        "Cd": 0.31,
        "frontal_area_m2": 2.22,
        "steering_ratio": 15.5,
        "max_handwheel_deg": 542.5,
        "track_width_m": 1.82,
    },
    "hatchback": {
        "name": "B-Segment Hatchback",
        "desc": "Maruti Swift / Hyundai i20 class",
        "kerb_mass_kg": 1050.0,
        "payload_kg": 100.0,
        "total_mass_kg": 1150.0,
        "wheelbase_m": 2.45,
        "lf_m": 1.00,
        "lr_m": 1.45,
        "cg_height_m": 0.50,
        "Cd": 0.32,
        "frontal_area_m2": 2.15,
        "steering_ratio": 15.5,
        "max_handwheel_deg": 542.5,
        "track_width_m": 1.74,
    },
    "compact_suv": {
        "name": "Compact SUV",
        "desc": "Hyundai Creta / Kia Seltos class",
        "kerb_mass_kg": 1600.0,
        "payload_kg": 120.0,
        "total_mass_kg": 1720.0,
        "wheelbase_m": 2.61,
        "lf_m": 1.12,
        "lr_m": 1.49,
        "cg_height_m": 0.60,
        "Cd": 0.34,
        "frontal_area_m2": 2.45,
        "steering_ratio": 15.5,
        "max_handwheel_deg": 542.5,
        "track_width_m": 1.79,
    },
    "heavy_suv": {
        "name": "Full-Size SUV (Loaded)",
        "desc": "Mahindra Scorpio / Toyota Fortuner class",
        "kerb_mass_kg": 2200.0,
        "payload_kg": 250.0,
        "total_mass_kg": 2450.0,
        "wheelbase_m": 2.75,
        "lf_m": 1.20,
        "lr_m": 1.55,
        "cg_height_m": 0.68,
        "Cd": 0.38,
        "frontal_area_m2": 2.80,
        "steering_ratio": 16.0,
        "max_handwheel_deg": 560.0,
        "track_width_m": 1.85,
    },
}

# ---------------------------------------------------------------------------
# Weather & Road Surface Friction Presets
# ---------------------------------------------------------------------------
WEATHER_PROFILES = {
    "dry": {
        "name": "Dry Highway",
        "desc": "Dry Asphalt (Optimal Grip)",
        "mu_peak": 0.88,
        "mu_slide": 0.72,
        "Crr": 0.015,
        "rain_intensity": 0.0,
    },
    "wet": {
        "name": "Monsoon Rain",
        "desc": "Wet Asphalt (Reduced Grip)",
        "mu_peak": 0.55,
        "mu_slide": 0.44,
        "Crr": 0.020,
        "rain_intensity": 0.5,
    },
    "heavy_rain": {
        "name": "Heavy Downpour",
        "desc": "Standing Water (Hydroplane Risk)",
        "mu_peak": 0.38,
        "mu_slide": 0.28,
        "Crr": 0.025,
        "rain_intensity": 1.0,
    },
    "gravel": {
        "name": "Gravel / Shoulder",
        "desc": "Unpaved Loose Road Margin",
        "mu_peak": 0.45,
        "mu_slide": 0.35,
        "Crr": 0.038,
        "rain_intensity": 0.0,
    },
}


@dataclass
class VehicleSpecs:
    """Standard physical parameters of an average C-segment passenger car."""
    # Active profile labels
    vehicle_type: str = "sedan"
    weather_type: str = "dry"

    # Inertial properties
    kerb_mass_kg: float = 1450.0       # Kerb mass (kg)
    payload_mass_kg: float = 100.0     # Driver + luggage (kg)
    total_mass_kg: float = 1550.0      # Total operating mass (kg)
    Iz_kgm2: float = 2450.0            # Yaw moment of inertia (kg*m^2)

    # Geometry
    wheelbase_m: float = 2.70          # L (front-to-rear axle distance)
    lf_m: float = 1.10                 # Distance from CG to front axle (m)
    lr_m: float = 1.60                 # Distance from CG to rear axle (m)
    track_width_m: float = 1.82        # Track width (m)
    cg_height_m: float = 0.52          # Center of Gravity height (m)

    # Steering
    steering_ratio: float = 15.5       # Steering wheel to road wheel angle ratio
    max_road_wheel_deg: float = 35.0   # Max road wheel steering angle (deg)
    max_handwheel_deg: float = 542.5   # Max handwheel angle = 35 * 15.5 (deg)
    max_steer_rate_deg_s: float = 380.0# Max human / servo steering rate (deg/s)

    # Tires & Aerodynamics
    Cf_N_rad: float = 85000.0          # Front axle cornering stiffness (N/rad)
    Cr_N_rad: float = 95000.0          # Rear axle cornering stiffness (N/rad)
    mu_peak: float = 0.88              # Road surface peak friction coefficient
    mu_slide: float = 0.72             # Sliding friction coefficient
    Cd: float = 0.31                   # Aerodynamic drag coefficient
    frontal_area_m2: float = 2.22      # Frontal area (m^2)
    air_density: float = 1.225         # Air density at sea level (kg/m^3)
    Crr: float = 0.015                 # Rolling resistance coefficient
    gravity_mps2: float = 9.81         # Acceleration due to gravity (m/s^2)

    # Powertrain & Chassis Limits (MG Hector Tomahawk EV defaults)
    power_bhp: float = 201.0           # Peak motor power (bhp)
    torque_nm: float = 310.0           # Peak instantaneous motor torque (Nm)
    ground_clearance_mm: float = 230.0 # Battery pack ground clearance (mm)

    # Braking Performance Limits
    hydraulic_lag_s: float = 0.12      # Brake system hydraulic response delay (s)
    max_comfort_brake_mps2: float = 2.8# Comfortable passenger braking (~0.29 g)
    max_aeb_brake_mps2: float = 8.2    # Emergency maximum ABS braking (~0.84 g)


class VehiclePhysicsEngine:
    """Simulates real-world vehicle physical response to throttle, brake, and steering inputs."""

    def __init__(self, specs: VehicleSpecs = None, init_speed_mps: float = 20.0):
        self.specs = specs or VehicleSpecs()
        
        # State vector: [x, y, heading_psi, vx, vy, yaw_rate_r]
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.vx = max(0.0, init_speed_mps)
        self.vy = 0.0
        self.yaw_rate = 0.0

        # Control inputs (filtered)
        self.road_wheel_angle_deg = 0.0
        self.throttle_pct = 0.0
        self.brake_pct = 0.0

        # Physical telemetry
        self.ax_mps2 = 0.0
        self.ay_mps2 = 0.0
        self.pitch_deg = 0.0
        self.fz_front_n = (self.specs.total_mass_kg * self.specs.gravity_mps2 * self.specs.lr_m) / self.specs.wheelbase_m
        self.fz_rear_n = (self.specs.total_mass_kg * self.specs.gravity_mps2 * self.specs.lf_m) / self.specs.wheelbase_m
        self.friction_utilization_pct = 0.0
        self.stopping_distance_m = 0.0

    def compute_stopping_distance(self, speed_mps: float, brake_intensity: float = 1.0) -> float:
        """
        Computes real physical stopping distance (meters):
        d_stop = d_lag + d_braking = (v * t_lag) + (v^2 / (2 * mu * g))
        """
        if speed_mps <= 0.1:
            return 0.0
        t_lag = self.specs.hydraulic_lag_s
        max_decel = self.specs.mu_peak * self.specs.gravity_mps2
        eff_decel = max(1.5, max_decel * brake_intensity)
        d_lag = speed_mps * t_lag
        d_brake = (speed_mps ** 2) / (2.0 * eff_decel)
        return d_lag + d_brake

    def set_vehicle_profile(self, profile_key: str, custom_weight_kg: Optional[float] = None):
        """Update vehicle weight, dimensions, and aerodynamic profile."""
        if profile_key in VEHICLE_PROFILES:
            prof = VEHICLE_PROFILES[profile_key]
            self.specs.vehicle_type = profile_key
            self.specs.kerb_mass_kg = prof["kerb_mass_kg"]
            self.specs.payload_mass_kg = prof["payload_kg"]
            self.specs.total_mass_kg = prof["total_mass_kg"]
            self.specs.wheelbase_m = prof["wheelbase_m"]
            self.specs.lf_m = prof["lf_m"]
            self.specs.lr_m = prof["lr_m"]
            self.specs.cg_height_m = prof["cg_height_m"]
            self.specs.Cd = prof["Cd"]
            self.specs.frontal_area_m2 = prof["frontal_area_m2"]
            self.specs.Iz_kgm2 = prof["total_mass_kg"] * (prof["wheelbase_m"] ** 2) / 4.5
            if "steering_ratio" in prof:
                self.specs.steering_ratio = prof["steering_ratio"]
            if "max_handwheel_deg" in prof:
                self.specs.max_handwheel_deg = prof["max_handwheel_deg"]
            if "track_width_m" in prof:
                self.specs.track_width_m = prof["track_width_m"]
            if "power_bhp" in prof:
                self.specs.power_bhp = prof["power_bhp"]
            if "torque_nm" in prof:
                self.specs.torque_nm = prof["torque_nm"]
            if "ground_clearance_mm" in prof:
                self.specs.ground_clearance_mm = prof["ground_clearance_mm"]
        if custom_weight_kg is not None and custom_weight_kg > 300.0:
            self.specs.total_mass_kg = float(custom_weight_kg)

    def set_weather_profile(self, weather_key: str, custom_friction_mu: Optional[float] = None):
        """Update road surface friction (mu) and rolling resistance based on weather."""
        if weather_key in WEATHER_PROFILES:
            w = WEATHER_PROFILES[weather_key]
            self.specs.weather_type = weather_key
            self.specs.mu_peak = w["mu_peak"]
            self.specs.mu_slide = w["mu_slide"]
            self.specs.Crr = w["Crr"]
        if custom_friction_mu is not None and 0.05 <= custom_friction_mu <= 1.2:
            self.specs.mu_peak = float(custom_friction_mu)
            self.specs.mu_slide = float(custom_friction_mu * 0.8)

    def cycle_weather_profile(self) -> str:
        keys = list(WEATHER_PROFILES.keys())
        idx = (keys.index(self.specs.weather_type) + 1) % len(keys)
        new_key = keys[idx]
        self.set_weather_profile(new_key)
        return new_key

    def cycle_vehicle_profile(self) -> str:
        keys = list(VEHICLE_PROFILES.keys())
        idx = (keys.index(self.specs.vehicle_type) + 1) % len(keys)
        new_key = keys[idx]
        self.set_vehicle_profile(new_key)
        return new_key

    def step(self, throttle_cmd: float, brake_cmd: float, target_steering_wheel_deg: float, dt: float = 0.02) -> dict:
        """
        Integrates vehicle dynamics over timestep dt (seconds).
        Inputs:
          throttle_cmd: [0.0, 1.0] (0% to 100%)
          brake_cmd:    [0.0, 1.0] (0% to 100%)
          target_steering_wheel_deg: [-540, +540] deg
        """
        dt = max(0.001, min(0.1, dt))
        s = self.specs

        # 1. Steering Handwheel to Road Wheel mapping with servo slew-rate limitation
        target_road_wheel_deg = max(
            -s.max_road_wheel_deg,
            min(s.max_road_wheel_deg, target_steering_wheel_deg / s.steering_ratio)
        )
        max_delta = s.max_steer_rate_deg_s * dt / s.steering_ratio
        delta_steer = target_road_wheel_deg - self.road_wheel_angle_deg
        delta_steer = max(-max_delta, min(max_delta, delta_steer))
        self.road_wheel_angle_deg += delta_steer
        delta_rad = math.radians(self.road_wheel_angle_deg)

        # 2. Smooth pedal inputs
        self.throttle_pct = max(0.0, min(1.0, throttle_cmd))
        self.brake_pct = max(0.0, min(1.0, brake_cmd))

        # 3. Resistive environmental forces
        # Aerodynamic Drag: F_drag = 0.5 * rho * Cd * A * v^2
        f_aero = 0.5 * s.air_density * s.Cd * s.frontal_area_m2 * (self.vx ** 2)
        # Rolling Resistance: F_roll = Crr * m * g
        f_roll = s.Crr * s.total_mass_kg * s.gravity_mps2 if self.vx > 0.1 else 0.0

        # 4. Longitudinal Drive & Brake Forces
        if self.brake_pct > 0.01:
            # Brake takes precedence (brake-throttle override safety)
            f_drive = 0.0
            # ABS modulated braking deceleration limit strictly bounded by tire-road friction mu * g
            max_achievable_decel = s.mu_peak * s.gravity_mps2
            max_brake_force = s.total_mass_kg * max_achievable_decel * self.brake_pct
            f_brake = max_brake_force
            f_net_long = -(f_brake + f_aero + f_roll)
        else:
            # Powertrain traction force — uses vehicle profile torque_nm and power_bhp
            # For EV: single-speed reduction gear ratio ~9.0, tire radius ~0.34m (215/55 R18)
            # For ICE/PHEV: effective 1st-gear multiplication ~4.2, tire radius ~0.34m
            tire_radius_m = 0.34  # 215/55 R18: 0.215*0.55 + 0.2286 = 0.347m ≈ 0.34m
            motor_torque_nm = s.torque_nm
            motor_power_w = s.power_bhp * 745.7  # bhp -> watts
            # Gear ratio: EV single-speed ~9.0; ICE effective ~4.2 at typical cruising gear
            gear_ratio = 9.0 if "ev" in s.vehicle_type else 4.2
            # Torque-limited traction at wheel: F = T * gear_ratio * efficiency / tire_radius
            f_torque_limited = motor_torque_nm * gear_ratio * 0.92 / tire_radius_m
            # Power-limited traction at speed: F = P / v (caps force at high speed)
            v_clamp = max(1.0, self.vx)
            f_power_limited = motor_power_w * 0.90 / v_clamp  # 90% drivetrain efficiency
            # Actual max traction is the lesser of torque limit and power limit
            max_traction_n = min(f_torque_limited, f_power_limited)
            # Traction cannot exceed tire grip (friction circle longitudinal component)
            max_tire_traction = s.mu_peak * s.total_mass_kg * s.gravity_mps2 * 0.55  # ~55% on drive axle
            max_traction_n = min(max_traction_n, max_tire_traction)
            f_drive = self.throttle_pct * max_traction_n
            f_net_long = f_drive - f_aero - f_roll

        self.ax_mps2 = f_net_long / s.total_mass_kg

        # Clamp stopping when speed reaches 0
        if self.vx <= 0.05 and self.ax_mps2 < 0.0:
            self.ax_mps2 = 0.0
            self.vx = 0.0
        else:
            self.vx = max(0.0, self.vx + self.ax_mps2 * dt)

        # 5. Dynamic Normal Load Transfer (Nose-dive under braking, squat under acceleration)
        # Delta Fz = m * ax * (cg_height / wheelbase)
        delta_fz = s.total_mass_kg * self.ax_mps2 * (s.cg_height_m / s.wheelbase_m)
        self.fz_front_n = max(1000.0, (s.total_mass_kg * s.gravity_mps2 * s.lr_m / s.wheelbase_m) - delta_fz)
        self.fz_rear_n = max(1000.0, (s.total_mass_kg * s.gravity_mps2 * s.lf_m / s.wheelbase_m) + delta_fz)

        # Pitch angle (suspension compression, radians -> deg)
        # Typical suspension pitch stiffness k_pitch ~ 85,000 N*m/rad
        k_pitch = 85000.0
        pitch_rad = (s.total_mass_kg * self.ax_mps2 * s.cg_height_m) / k_pitch
        self.pitch_deg = math.degrees(pitch_rad)

        # 6. Lateral Dynamics (Bicycle model)
        if self.vx > 0.5:
            # Slip angles
            alpha_f = delta_rad - math.atan2(self.vy + s.lf_m * self.yaw_rate, self.vx)
            alpha_r = -math.atan2(self.vy - s.lr_m * self.yaw_rate, self.vx)

            # Lateral tire forces with friction saturation limit
            max_lateral_front = s.mu_peak * self.fz_front_n
            max_lateral_rear = s.mu_peak * self.fz_rear_n
            fy_f = max(-max_lateral_front, min(max_lateral_front, s.Cf_N_rad * alpha_f))
            fy_r = max(-max_lateral_rear, min(max_lateral_rear, s.Cr_N_rad * alpha_r))

            # Lateral acceleration and yaw acceleration
            vy_dot = (fy_f * math.cos(delta_rad) + fy_r) / s.total_mass_kg - self.vx * self.yaw_rate
            yaw_accel = (s.lf_m * fy_f * math.cos(delta_rad) - s.lr_m * fy_r) / s.Iz_kgm2

            self.vy += vy_dot * dt
            self.yaw_rate += yaw_accel * dt
            self.ay_mps2 = vy_dot + self.vx * self.yaw_rate
        else:
            # Kinematic fallback at low speeds
            self.vy = 0.0
            self.yaw_rate = (self.vx / s.wheelbase_m) * math.tan(delta_rad)
            self.ay_mps2 = self.vx * self.yaw_rate

        # 7. World Pose Integration
        self.heading += self.yaw_rate * dt
        # Normalize heading to [-pi, pi]
        self.heading = (self.heading + math.pi) % (2.0 * math.pi) - math.pi

        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        self.x += (self.vx * cos_h - self.vy * sin_h) * dt
        self.y += (self.vx * sin_h + self.vy * cos_h) * dt

        # 8. Friction circle utilization
        total_accel_g = math.sqrt(self.ax_mps2 ** 2 + self.ay_mps2 ** 2) / s.gravity_mps2
        self.friction_utilization_pct = min(100.0, (total_accel_g / s.mu_peak) * 100.0)

        # 9. Predictive stopping distance
        self.stopping_distance_m = self.compute_stopping_distance(self.vx, brake_intensity=1.0)

        # Actual handwheel angle in degrees
        actual_handwheel_deg = self.road_wheel_angle_deg * s.steering_ratio

        # Speed-sensitive Electronic Power Steering (EPS) feedback:
        # Matches MG Hector specification: light maneuverability in slow urban driving and firming up at higher speeds
        aligning_torque_nm = abs(math.sin(delta_rad) * self.fz_front_n * 0.035)
        speed_factor = min(1.0, max(0.0, (self.vx * 3.6 - 15.0) / 65.0))  # 0 at <=15 km/h, 1 at >=80 km/h
        eps_assist_pct = 0.82 - 0.28 * speed_factor  # 82% assist at low speed down to 54% assist at highway
        steering_torque_nm = max(0.6, (aligning_torque_nm / s.steering_ratio) * (1.0 - eps_assist_pct) * 8.0)

        return {
            "x": self.x,
            "y": self.y,
            "heading_rad": self.heading,
            "heading_deg": math.degrees(self.heading),
            "speed_mps": self.vx,
            "speed_kmh": self.vx * 3.6,
            "ax_mps2": self.ax_mps2,
            "ax_g": self.ax_mps2 / s.gravity_mps2,
            "ay_mps2": self.ay_mps2,
            "ay_g": self.ay_mps2 / s.gravity_mps2,
            "road_wheel_deg": self.road_wheel_angle_deg,
            "handwheel_deg": actual_handwheel_deg,
            "steering_torque_nm": steering_torque_nm,
            "pitch_deg": self.pitch_deg,
            "fz_front_n": self.fz_front_n,
            "fz_rear_n": self.fz_rear_n,
            "friction_util_pct": self.friction_utilization_pct,
            "stopping_dist_m": self.stopping_distance_m,
            "throttle_pct": self.throttle_pct * 100.0,
            "brake_pct": self.brake_pct * 100.0,
        }
