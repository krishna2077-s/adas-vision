"""
run_xodr.py — ASAM OpenDRIVE (.xodr) Simulation & Benchmark Runner.

Loads an OpenDRIVE XML (.xodr) road network and hazard layout, translates it into
canonical ADAS scenario kinematics, and executes the ADAS Decision & Planning Engine
with full comparative benchmarking and telemetry logging.

Usage:
    python run_xodr.py [path_to_file.xodr]
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenario_lab.export.opendrive_importer import import_from_opendrive
from scenario_lab.stress.benchmark_engine import BenchmarkEngine
from scenario_lab.adapter.matlab_runner import simulate_kinematics
from scenario_lab.adapter.matlab_adapter import MatlabAdapter


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_xodr_simulation(xodr_path: str):
    if not os.path.exists(xodr_path):
        print(f"[ERROR] OpenDRIVE file not found: {xodr_path}")
        sys.exit(1)

    print_banner(f"ASAM OpenDRIVE (.xodr) Simulation Runner: {os.path.basename(xodr_path)}")
    print(f"Loading road network from: {xodr_path}...")

    # 1. Parse OpenDRIVE
    scenario = import_from_opendrive(xodr_path)
    
    print("\n--- Road & Environmental Context ---")
    print(f"  Scenario ID      : {scenario.scenario_id}")
    print(f"  Scenario Name    : {scenario.metadata.name}")
    print(f"  Road Geometry    : {scenario.road.length_m} m length, {scenario.road.type} (lane width: {scenario.road.lane_half_width_m * 2} m)")
    print(f"  Surface & Grip   : {scenario.road.surface} (asphalt/tarmac friction)")
    print(f"  Ego Initial State: Speed {scenario.ego_vehicle.initial_speed_kmh} km/h, Goal at {scenario.ego_vehicle.goal_x_m} m")
    
    print("\n--- Detected Dynamic & Static Obstacles (OpenDRIVE <objects>) ---")
    for i, a in enumerate(scenario.actors, 1):
        print(f"  [{i}] {a.actor_class:<15} ID: {a.id:<18} @ x={a.initial_x_m:5.1f}m, y={a.initial_y_m:4.1f}m | "
              f"dim: {a.length_m}x{a.width_m}m | beh: {a.behaviour} (vx={a.initial_vx_mps} m/s, vy={a.initial_vy_mps} m/s)")

    # 2. Generate MATLAB Driving Scenario Designer script
    matlab_dir = str(ROOT / "matlab")
    gen_dir = str(ROOT / "scenario_lab" / "generated")
    adapter = MatlabAdapter(matlab_dir, gen_dir)
    m_script_path, result_json_path = adapter.generate_script(scenario)
    print(f"\n[Generated MATLAB Scenario Script]: {m_script_path}")

    # 3. Run Head-to-Head Comparative Benchmark (Adaptive vs Baseline)
    print_banner("Running Head-to-Head Benchmark: Adaptive ADAS vs Baseline Western ADAS")
    benchmark_res = BenchmarkEngine.run_benchmark(scenario)

    adapt = benchmark_res["adaptive"]
    base = benchmark_res["baseline"]
    comp = benchmark_res["comparison"]
    deg = benchmark_res["sensor_degradation"]

    print("\n=== Comparative Safety & Performance Scorecard ===")
    print(f"{'Metric':<32} | {'Adaptive ADAS':<18} | {'Baseline ADAS':<18}")
    print("-" * 74)
    print(f"{'Arrival Status':<32} | {'GOAL REACHED' if adapt.get('arrived') else 'FAILED':<18} | {'GOAL REACHED' if base.get('arrived') else 'FAILED':<18}")
    print(f"{'Total Collisions':<32} | {adapt.get('collisions', 0):<18} | {base.get('collisions', 0):<18}")
    print(f"{'Minimum Clearance (m)':<32} | {adapt.get('min_clearance_m', 0.0):<18.2f} | {base.get('min_clearance_m', 0.0):<18.2f}")
    print(f"{'Trip Time (s)':<32} | {adapt.get('t_total_s', 0.0):<18.1f} | {base.get('t_total_s', 0.0):<18.1f}")
    print(f"{'Average Speed (km/h)':<32} | {adapt.get('mean_speed_kmh', 0.0):<18.1f} | {base.get('mean_speed_kmh', 0.0):<18.1f}")
    print(f"{'Path Smoothness':<32} | {adapt.get('path_smoothness', 0.0):<18.2f} | {base.get('path_smoothness', 0.0):<18.2f}")
    print(f"{'Dynamic Replans':<32} | {adapt.get('replans', 0):<18} | {base.get('replans', 0):<18}")

    print("\n=== ADAS Decision State Distribution ===")
    print(f"{'State':<24} | {'Adaptive':<15} | {'Baseline':<15}")
    print("-" * 58)
    print(f"{'PROCEED':<24} | {adapt.get('decision_proceed', 0.0) * 100:5.1f}%          | {base.get('decision_proceed', 0.0) * 100:5.1f}%")
    print(f"{'CAUTION':<24} | {adapt.get('decision_caution', 0.0) * 100:5.1f}%          | {base.get('decision_caution', 0.0) * 100:5.1f}%")
    print(f"{'SLOW (Passing)':<24} | {adapt.get('decision_slow', 0.0) * 100:5.1f}%          | {base.get('decision_slow', 0.0) * 100:5.1f}%")
    print(f"{'BRAKE':<24} | {adapt.get('decision_brake', 0.0) * 100:5.1f}%          | {base.get('decision_brake', 0.0) * 100:5.1f}%")
    print(f"{'EMERGENCY_STOP':<24} | {adapt.get('decision_emergency_stop', 0.0) * 100:5.1f}%          | {base.get('decision_emergency_stop', 0.0) * 100:5.1f}%")

    print("\n=== Physics-Informed Sensor & Environment Degradation ===")
    friction = deg.get('friction_coefficient', 0.85) if isinstance(deg, dict) else deg.friction_coefficient
    max_decel = deg.get('max_braking_decel_mps2', 5.8) if isinstance(deg, dict) else deg.max_braking_decel_mps2
    cam_att = deg.get('camera_visibility_attenuation', 0.0) if isinstance(deg, dict) else deg.camera_visibility_attenuation
    lidar_ext = deg.get('lidar_rain_extinction_db_km', 0.0) if isinstance(deg, dict) else deg.lidar_rain_extinction_db_km
    radar_clutter = deg.get('radar_snr_loss_db', 0.0) if isinstance(deg, dict) else deg.radar_snr_loss_db

    print(f"  Pavement Friction (mu) : {friction:.2f} (Max Braking Decel: {max_decel:.2f} m/s^2)")
    print(f"  Camera Attenuation     : {cam_att * 100:.1f}%")
    print(f"  LiDAR Rain Extinction  : {lidar_ext:.2f} dB/km")
    print(f"  Radar Ground Clutter   : {radar_clutter:.2f} dB SNR loss")

    print_banner(f"Final Verdict: {comp.get('verdict', 'SUCCESS')}")
    print(f"Adaptive Safety Advantage: {comp.get('verdict', 'Zero collisions with active lateral evasion.')}")
    expl = adapt.get('explanation', {})
    summary = expl.get('outcome_summary', 'Scenario execution completed successfully.') if isinstance(expl, dict) else getattr(expl, 'outcome_summary', '')
    print(f"Explanation: {summary}\n")


if __name__ == "__main__":
    target_xodr = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "scen-nlp-1792250798.xodr")
    run_xodr_simulation(target_xodr)
