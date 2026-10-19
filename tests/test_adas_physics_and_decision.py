"""
test_adas_physics_and_decision.py — Mathematical and functional verification test suite.

Verifies:
  1. Vehicle physics equations (braking load transfer, aerodynamic drag, stopping distance).
  2. ADAS decision state machine transitions (ACC -> FCW -> Autonomous Lane Change -> LKA).
  3. Steering handwheel control and lateral stability limits.
  4. Real-world stopping distance from 72 km/h.
"""

import math
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.vehicle_physics import VehiclePhysicsEngine, VehicleSpecs, VEHICLE_PROFILES
from core.adas_controller import ADASDecisionSystem, ADASState


def test_vehicle_braking_physics():
    """Verify emergency braking deceleration, load transfer, and stopping distance."""
    specs = VehicleSpecs()
    physics = VehiclePhysicsEngine(specs=specs, init_speed_mps=20.0)  # 72 km/h

    # Initial static load
    fz_front_static = (specs.total_mass_kg * specs.gravity_mps2 * specs.lr_m) / specs.wheelbase_m
    fz_rear_static = (specs.total_mass_kg * specs.gravity_mps2 * specs.lf_m) / specs.wheelbase_m

    # Execute 100% emergency brake step
    state = physics.step(throttle_cmd=0.0, brake_cmd=1.0, target_steering_wheel_deg=0.0, dt=0.02)

    # Deceleration should be between -7.0 and -9.0 m/s^2 (ABS peak friction)
    assert -9.0 <= state["ax_mps2"] <= -7.0, f"Unexpected ax: {state['ax_mps2']}"

    # Load transfer: Front axle load must increase (nose dive), rear must decrease
    assert state["fz_front_n"] > fz_front_static, "Front axle load did not increase under braking"
    assert state["fz_rear_n"] < fz_rear_static, "Rear axle load did not decrease under braking"

    # Pitch angle must be negative (nose dipping down)
    assert state["pitch_deg"] < 0.0, "Vehicle did not pitch down under braking"

    # Stopping distance from 72 km/h (20 m/s) should be between 22m and 30m
    stop_dist = state["stopping_dist_m"]
    assert 22.0 <= stop_dist <= 30.0, f"Stopping distance out of realistic range: {stop_dist}"
    print(f"[PASS] Braking Physics: Decel={state['ax_mps2']:.2f} m/s^2, StopDist={stop_dist:.1f} m, FrontLoad={state['fz_front_n']:.0f} N")


def test_adas_steering_and_lane_change():
    """Verify autonomous lane change initiation, trajectory, and steering angles."""
    adas = ADASDecisionSystem(cruise_speed_kmh=72.0)
    
    # Run simulation ticks
    states_observed = set()
    steer_angles = []

    dt = 0.05
    for step_i in range(int(35.0 / dt)):
        t = step_i * dt
        cmd, phys = adas.update(sim_t=t, dt=dt)
        states_observed.add(cmd.state)
        steer_angles.append(cmd.steering_wheel_deg)

    # Must observe Cruise, Lane Change, and LKA Centering
    assert ADASState.CRUISE in states_observed, "Did not enter CRUISE"
    assert ADASState.AUTONOMOUS_LANE_CHANGE in states_observed, "Did not execute AUTONOMOUS_LANE_CHANGE"
    assert ADASState.LKA_CENTERING in states_observed, "Did not execute LKA_CENTERING"

    # Max steering wheel angle must be reasonable for highway lane change (within +/- 90 degrees)
    max_steer = max(abs(ang) for ang in steer_angles)
    assert 5.0 < max_steer < 120.0, f"Steering handwheel angle unrealistic: {max_steer} deg"

    # Lateral displacement should have successfully shifted into adjacent lane (~3.5m)
    final_lateral = adas.physics.y
    assert abs(final_lateral - (-3.5)) < 0.8, f"Vehicle did not reach target lane: {final_lateral} m"
    print(f"[PASS] ADAS Decision & Steering: LaneChange completed (Lateral={final_lateral:.2f}m, MaxSteer={max_steer:.1f} deg)")


def test_weather_friction_and_vehicle_weight_adaptation():
    """Verify physics equations and ADAS threshold scaling under dry vs monsoon vs downpour conditions."""
    adas = ADASDecisionSystem(cruise_speed_kmh=72.0, fcw_sensitivity="normal")
    v_mps = 20.0  # 72 km/h

    # 1. Dry Highway (mu = 0.88)
    adas.set_weather("dry")
    d_dry = adas.physics.compute_stopping_distance(v_mps)
    scene_dry = adas.evaluate_scene(sim_t=5.0)
    assert 24.0 <= d_dry <= 26.5, f"Dry stopping distance out of range: {d_dry:.1f} m"
    assert abs(scene_dry["time_headway_s"] - 1.6) < 0.05, f"Expected 1.6s headway on dry: {scene_dry['time_headway_s']}"
    assert abs(scene_dry["fcw_ttc_thresh"] - 3.5) < 0.05, f"Expected 3.5s FCW TTC on dry: {scene_dry['fcw_ttc_thresh']}"

    # 2. Monsoon Rain (mu = 0.55)
    adas.set_weather("wet")
    d_wet = adas.physics.compute_stopping_distance(v_mps)
    scene_wet = adas.evaluate_scene(sim_t=5.0)
    assert 37.0 <= d_wet <= 41.0, f"Monsoon wet stopping distance out of range: {d_wet:.1f} m"
    # Stopping distance must be > 50% greater than dry
    assert d_wet > d_dry * 1.5, f"Monsoon stopping distance did not scale properly: {d_wet:.1f} vs {d_dry:.1f}"
    # Headway must expand to ~2.0s
    assert 1.95 <= scene_wet["time_headway_s"] <= 2.10, f"Monsoon headway incorrect: {scene_wet['time_headway_s']}"
    # FCW warning threshold must expand to ~5.6s
    assert 5.4 <= scene_wet["fcw_ttc_thresh"] <= 5.8, f"Monsoon FCW thresh incorrect: {scene_wet['fcw_ttc_thresh']}"

    # 3. Heavy Downpour (mu = 0.38)
    adas.set_weather("heavy_rain")
    d_heavy = adas.physics.compute_stopping_distance(v_mps)
    scene_heavy = adas.evaluate_scene(sim_t=5.0)
    assert 53.0 <= d_heavy <= 58.0, f"Heavy rain stopping distance out of range: {d_heavy:.1f} m"
    # Stopping distance must more than double (> 100% increase)
    assert d_heavy > d_dry * 2.1, f"Heavy rain stopping distance did not double: {d_heavy:.1f} vs {d_dry:.1f}"
    # Headway must expand to ~2.4s
    assert 2.35 <= scene_heavy["time_headway_s"] <= 2.55, f"Heavy rain headway incorrect: {scene_heavy['time_headway_s']}"
    # FCW warning threshold must expand to ~8.1s
    assert 7.8 <= scene_heavy["fcw_ttc_thresh"] <= 8.4, f"Heavy rain FCW thresh incorrect: {scene_heavy['fcw_ttc_thresh']}"

    # 4. Vehicle Mass & Load Transfer Sensitivity
    adas.set_vehicle("sedan")
    adas.set_weather("dry")
    state_sedan = adas.physics.step(throttle_cmd=0.0, brake_cmd=1.0, target_steering_wheel_deg=0.0, dt=0.02)

    adas.set_vehicle("heavy_suv")
    state_suv = adas.physics.step(throttle_cmd=0.0, brake_cmd=1.0, target_steering_wheel_deg=0.0, dt=0.02)

    assert state_suv["fz_front_n"] > state_sedan["fz_front_n"] * 1.4, "SUV axle load not scaled with vehicle mass"
    print(f"[PASS] Friction & Weight Adaptation: Dry={d_dry:.1f}m -> Wet={d_wet:.1f}m -> HeavyRain={d_heavy:.1f}m | SUV FrontLoad={state_suv['fz_front_n']:.0f} N")


def test_profile_cycling_and_custom_overrides():
    """Verify runtime hotkey cycling and custom parameter overrides."""
    physics = VehiclePhysicsEngine()

    # Weather cycling: dry -> wet -> heavy_rain -> gravel -> dry
    assert physics.specs.weather_type == "dry"
    assert physics.cycle_weather_profile() == "wet"
    assert physics.cycle_weather_profile() == "heavy_rain"
    assert physics.cycle_weather_profile() == "gravel"
    assert physics.cycle_weather_profile() == "dry"

    # Vehicle cycling across all profiles
    keys = list(VEHICLE_PROFILES.keys())
    start_idx = keys.index(physics.specs.vehicle_type)
    for i in range(1, len(keys) + 1):
        expected_key = keys[(start_idx + i) % len(keys)]
        next_k = physics.cycle_vehicle_profile()
        assert next_k == expected_key, f"Expected {expected_key}, got {next_k}"

    # Custom weight and friction overrides
    physics.set_vehicle_profile("hector_ev", custom_weight_kg=1990.0)
    assert physics.specs.total_mass_kg == 1990.0
    physics.set_weather_profile("dry", custom_friction_mu=0.62)
    assert abs(physics.specs.mu_peak - 0.62) < 1e-4

    print("[PASS] Profile Cycling & Custom Overrides: Clean state rotation and custom mass/mu assignment")


def test_mg_hector_tomahawk_specs_and_adas_settings():
    """Verify exact MG Hector Tomahawk EV & PHEV engineering specs and driver ADAS adjustments."""
    adas = ADASDecisionSystem(cruise_speed_kmh=72.0)
    adas.set_vehicle("hector_ev")
    specs = adas.physics.specs

    # Footprint & Geometry
    assert specs.wheelbase_m == 2.81, f"Wheelbase incorrect: {specs.wheelbase_m}"
    assert abs(specs.lf_m - 1.32) < 0.02, f"Front axle dist incorrect: {specs.lf_m}"
    assert abs(specs.lr_m - 1.49) < 0.02, f"Rear axle dist incorrect: {specs.lr_m}"
    assert specs.total_mass_kg == 1940.0, f"Total mass incorrect: {specs.total_mass_kg}"
    assert specs.cg_height_m == 0.53, f"CG height incorrect: {specs.cg_height_m}"
    assert specs.steering_ratio == 14.4, f"Steering ratio incorrect: {specs.steering_ratio}"
    assert specs.max_handwheel_deg == 504.0, f"Max handwheel deg incorrect: {specs.max_handwheel_deg}"

    # Weight distribution static check (53% Front : 47% Rear)
    fz_f = (specs.total_mass_kg * specs.gravity_mps2 * specs.lr_m) / specs.wheelbase_m
    fz_r = (specs.total_mass_kg * specs.gravity_mps2 * specs.lf_m) / specs.wheelbase_m
    front_ratio = fz_f / (fz_f + fz_r)
    assert 0.52 <= front_ratio <= 0.54, f"Weight distribution out of 53% front spec: {front_ratio}"

    # Driver-Adjustable FCW Sensitivity: Comfort (2.4s) vs Normal (3.5s) vs Aggressive (4.5s)
    adas.fcw_sensitivity = "comfort"
    scene_comfort = adas.evaluate_scene(sim_t=5.0)
    assert abs(scene_comfort["fcw_ttc_thresh"] - 2.4) < 0.05, f"Comfort FCW threshold incorrect: {scene_comfort['fcw_ttc_thresh']}"

    adas.cycle_fcw_sensitivity()  # switches to normal
    assert adas.fcw_sensitivity == "normal"
    scene_normal = adas.evaluate_scene(sim_t=5.0)
    assert abs(scene_normal["fcw_ttc_thresh"] - 3.5) < 0.05, f"Normal FCW threshold incorrect: {scene_normal['fcw_ttc_thresh']}"

    adas.cycle_fcw_sensitivity()  # switches to aggressive
    assert adas.fcw_sensitivity == "aggressive"
    scene_agg = adas.evaluate_scene(sim_t=5.0)
    assert abs(scene_agg["fcw_ttc_thresh"] - 4.5) < 0.05, f"Aggressive FCW threshold incorrect: {scene_agg['fcw_ttc_thresh']}"

    # Gentle Tactile LKA Nudge Intervention
    assert adas.lka_intervention == "gentle_nudge"
    steer_gentle = adas.steering_ctrl.compute(current_y=0.0, current_heading=0.0, target_y=-8.0, target_heading=0.0, vx=5.0)
    assert abs(abs(steer_gentle) - math.radians(18.0)) < 1e-3, f"Gentle LKA should clamp to 18 deg: {math.degrees(steer_gentle)}"

    adas.cycle_lka_intervention()  # switches to firm
    steer_firm = adas.steering_ctrl.compute(current_y=0.0, current_heading=0.0, target_y=-8.0, target_heading=0.0, vx=5.0)
    assert abs(abs(steer_firm) - math.radians(35.0)) < 1e-3, f"Firm LKA should permit full 35 deg: {math.degrees(steer_firm)}"

    # TJA Curve / Congestion Control Toggle
    assert adas.tja_enabled is True
    adas.toggle_tja()
    assert adas.tja_enabled is False
    adas.toggle_tja()
    assert adas.tja_enabled is True

    print(f"[PASS] MG Hector Tomahawk EV: Mass={specs.total_mass_kg}kg, Dist={front_ratio*100:.1f}%F, L={specs.wheelbase_m}m, LKA=GentleNudge, FCW=Comfort/Normal/Aggressive verified")


def test_powertrain_acceleration_and_torque():
    """Verify MG Hector EV (201 bhp, 310 Nm) and PHEV (198 bhp, 237 Nm) 0-100 km/h acceleration."""
    # 1. Hector EV: spec 0-100 in ~8.2s
    phys_ev = VehiclePhysicsEngine(init_speed_mps=0.0)
    phys_ev.set_vehicle_profile("hector_ev")
    t_ev = 0.0
    dt = 0.02
    while phys_ev.vx < (100.0 / 3.6) and t_ev < 15.0:
        phys_ev.step(throttle_cmd=1.0, brake_cmd=0.0, target_steering_wheel_deg=0.0, dt=dt)
        t_ev += dt

    assert 7.5 <= t_ev <= 9.0, f"Hector EV 0-100 time unrealistic: {t_ev:.2f}s (expected ~8.2s)"

    # 2. Hector PHEV: spec 0-100 in ~9.5s
    phys_phev = VehiclePhysicsEngine(init_speed_mps=0.0)
    phys_phev.set_vehicle_profile("hector_phev")
    t_phev = 0.0
    while phys_phev.vx < (100.0 / 3.6) and t_phev < 15.0:
        phys_phev.step(throttle_cmd=1.0, brake_cmd=0.0, target_steering_wheel_deg=0.0, dt=dt)
        t_phev += dt

    assert 9.0 <= t_phev <= 11.0, f"Hector PHEV 0-100 time unrealistic: {t_phev:.2f}s"
    print(f"[PASS] Powertrain 0-100 km/h: EV={t_ev:.2f}s (spec 8.2s), PHEV={t_phev:.2f}s (spec ~9.5s)")


def test_tja_curve_speed_limiting():
    """Verify Traffic Jam Assist (TJA) limits speed when road curvature requires slowing down."""
    adas = ADASDecisionSystem(cruise_speed_kmh=72.0, tja_enabled=True)
    adas.physics.x = 180.0  # Lead car is at 20m gap
    adas.physics.vx = 15.0  # 54 km/h in follow zone
    adas.physics.yaw_rate = 0.35  # Sharp curve / roundabout entry (R ≈ 43m)

    cmd, _ = adas.update(sim_t=5.0, dt=0.02)
    assert cmd.state == ADASState.FOLLOW
    # Curve-safe speed limit v_max = sqrt(0.88 * 9.81 * 43) * 0.85 ≈ 50 km/h
    assert cmd.target_speed_kmh < 65.0, f"TJA did not cap speed in sharp curve: {cmd.target_speed_kmh}"
    assert "Curve-control" in cmd.decision_reason
    print(f"[PASS] TJA Curve Speed Limiting: Target capped to {cmd.target_speed_kmh:.1f} km/h on curve")


def test_lane_departure_warning_cascade():
    """Verify unintended lane departure triggers LDW chime first, then gentle LKA corrective nudge."""
    adas = ADASDecisionSystem(cruise_speed_kmh=72.0)
    adas.physics.y = 0.45  # Inadvertent drift > 0.35m from lane center

    # Initial tick should enter LDW_WARNING
    cmd1, _ = adas.update(sim_t=5.0, dt=0.02)
    assert cmd1.state == ADASState.LDW_WARNING, f"Expected LDW_WARNING, got {cmd1.state}"
    assert "LDW ALERT" in cmd1.decision_reason

    # After 0.5s of warning without correction, LKA should take over with tactile nudge
    cmd2, _ = adas.update(sim_t=5.6, dt=0.02)
    assert cmd2.state == ADASState.LKA_CENTERING, f"Expected LKA_CENTERING, got {cmd2.state}"
    assert "LKA Active" in cmd2.decision_reason
    print("[PASS] LDW -> LKA Cascade: Unintended drift triggered LDW alert chime, followed by LKA corrective nudge")


def test_intelligent_headlamp_control():
    """Verify Level-2 Intelligent Headlamp Control (IHC) automatic beam switching."""
    adas = ADASDecisionSystem()

    # Clear road ahead (> 120m): Auto High-Beam
    scene_clear = adas.evaluate_scene(sim_t=5.0)
    assert "HIGH BEAM" in scene_clear["ihc_mode"]

    # Close traffic ahead (< 120m): Auto Low-Beam dip
    adas.physics.x = 100.0  # Moves ego closer to lead car (lead_dist = 100m)
    scene_traffic = adas.evaluate_scene(sim_t=5.0)
    assert "LOW BEAM" in scene_traffic["ihc_mode"]
    print(f"[PASS] Intelligent Headlamp Control (IHC): {scene_clear['ihc_mode']} -> {scene_traffic['ihc_mode']}")


def test_speed_sensitive_steering_eps():
    """Verify Electric Power Steering (EPS) offers light parking feel and firms up at highway speeds."""
    phys = VehiclePhysicsEngine(init_speed_mps=3.0)  # Urban speed (10.8 km/h)
    phys.set_vehicle_profile("hector_ev")
    s_low = phys.step(throttle_cmd=0.2, brake_cmd=0.0, target_steering_wheel_deg=90.0, dt=0.02)

    phys.vx = 25.0  # Highway speed (90 km/h)
    s_high = phys.step(throttle_cmd=0.2, brake_cmd=0.0, target_steering_wheel_deg=90.0, dt=0.02)

    assert s_low["steering_torque_nm"] < s_high["steering_torque_nm"], "Steering did not firm up at highway speed"
    assert s_low["steering_torque_nm"] <= 1.0, f"Low speed steering too stiff: {s_low['steering_torque_nm']}"
    assert s_high["steering_torque_nm"] >= 1.4, f"Highway steering too light: {s_high['steering_torque_nm']}"
    print(f"[PASS] Speed-Sensitive EPS: Low-speed={s_low['steering_torque_nm']:.2f} Nm (light) -> Highway={s_high['steering_torque_nm']:.2f} Nm (firm)")


if __name__ == "__main__":
    test_vehicle_braking_physics()
    test_adas_steering_and_lane_change()
    test_weather_friction_and_vehicle_weight_adaptation()
    test_profile_cycling_and_custom_overrides()
    test_mg_hector_tomahawk_specs_and_adas_settings()
    test_powertrain_acceleration_and_torque()
    test_tja_curve_speed_limiting()
    test_lane_departure_warning_cascade()
    test_intelligent_headlamp_control()
    test_speed_sensitive_steering_eps()
    print("\n=======================================================")
    print("  ALL 10 VERIFICATION TESTS PASSED RIGOROUSLY!")
    print("=======================================================\n")


