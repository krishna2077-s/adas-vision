# ADAS Vision — MATLAB Simulation Project
## Adaptive Path Planning & Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads
### SIH 2026 Problem Statement

---

## 🚀 Quick Start

```matlab
% 1. Open MATLAB R2024a or newer
% 2. Set working directory:
cd('C:\Users\Manish khandelwal\Downloads\adas-vision\adas-vision\matlab')

% 3. Auto-setup:
startup

% 4. Run everything:
main_adas
```

---

## 📋 What This Simulates

A complete autonomous driving pipeline validated across **5 realistic Indian road scenarios**:

| # | Scenario | Key Challenge |
|---|----------|---------------|
| 1 | **Village Road** | Unmarked road, oncoming motorcycle, crossing pedestrian |
| 2 | **Urban Intersection** | Unsignalised 4-way, auto-rickshaw, cross-traffic |
| 3 | **Highway Merge** | Truck cut-in at speed differential, wrong-way bike |
| 4 | **Dense Market** | Narrow street, pushcarts, darting shoppers |
| 5 | **Cattle Crossing** | Sudden cow entry → emergency stop → resume |

---

## 🏗️ Architecture

```
Multi-sensor Perception  →  Trajectory Prediction  →  Dynamic Occupancy Grid
                                                               ↓
                         R1-R7 Decision Engine  ←  Hybrid A* Path Planning
                                  ↓
                    Pure Pursuit + PID  →  Kinematic Bicycle Model
```

**Key algorithms:**
- **Hybrid A*** path planning (Navigation Toolbox `plannerHybridAStar`)
- **R1-R7 temporal ratchet** decision engine (Stateflow / MATLAB)
- **CV/CTRA trajectory prediction** (3s lookahead, 15 steps)
- **Speed-adaptive Pure Pursuit** lateral control
- **PID** longitudinal speed control

---

## 📁 File Structure

```
matlab/
├── main_adas.m                   ← START HERE (interactive menu)
├── startup.m                     ← Auto-setup on MATLAB open
├── setup_vehicle_model.m         ← Vehicle parameters
│
├── Scenarios (5 Indian road scenarios)
│   ├── scenario_village_road.m
│   ├── scenario_urban_intersection.m
│   ├── scenario_highway_merge.m
│   ├── scenario_dense_market.m
│   └── scenario_cattle_crossing.m
│
├── Core Pipeline
│   ├── adaptive_scenario_loop.m  ← Unified closed-loop engine
│   ├── plan_path.m               ← Hybrid A* planner
│   ├── predict_trajectories.m    ← Short-term prediction (CV/CTRA)
│   ├── build_occupancy_grid.m    ← Dynamic binary occupancy map
│   ├── decision_with_ratchet.m   ← R1-R7 + temporal ratchet
│   └── replan_trigger.m          ← Real-time replan condition
│
├── Benchmarking
│   ├── run_all_scenarios.m       ← Run all 5 scenarios
│   ├── run_baseline_vs_adaptive.m← Adaptive vs baseline comparison
│   └── collect_metrics.m         ← KPI computation + CSV + charts
│
├── Advanced Features
│   ├── build_stateflow_model.m   ← Simulink + Stateflow model
│   ├── generate_roadrunner_scenes.m ← OpenDRIVE scene export
│   ├── generate_demo_video.m     ← MP4 demo video
│   └── run_cosimulation.m        ← Live Python-MATLAB co-sim
│
└── roadrunner/                   ← Generated OpenDRIVE files
    ├── village_road.xodr
    └── urban_intersection.xodr
```

---

## 🔧 Requirements

### MATLAB
- **MATLAB R2024a** or newer (R2025b installed)
- **Navigation Toolbox** — `plannerHybridAStar` (fallback available)
- **Automated Driving Toolbox** — `drivingScenario`, scene export
- **Stateflow** — decision state machine (MATLAB function fallback)

### Python (for live co-simulation only)
- Python 3.9+, PyTorch, OpenCV
- See `../requirements.txt`

---

## 📊 Output Metrics

After running `run_all_scenarios`, the system reports:

| Metric | Description |
|--------|-------------|
| `duration_s` | Scenario completion time |
| `distance_m` | Total distance travelled |
| `path_smoothness` | Mean absolute curvature (rad/m) |
| `min_clearance_m` | Minimum obstacle clearance |
| `collisions` | Collision count (target: 0) |
| `n_replans` | Number of path replans |
| `mean_replan_ms` | Average replanning latency |
| `scenario_completed` | Goal reached (0/1) |

CSV outputs: `metrics_<scenario>.csv`, `metrics_all_scenarios_summary.csv`

---

## 🎬 Demo Video

```matlab
generate_demo_video   % → outputs 'adas_demo_video.mp4'
```

---

## 🗺️ RoadRunner Scenes

```matlab
generate_roadrunner_scenes   % → outputs village_road.xodr, urban_intersection.xodr
```

To open in MATLAB:
```matlab
scenario = drivingScenario;
roadNetwork(scenario, 'roadrunner/village_road.xodr');
```

---

## 📖 Documentation

- [TECHNICAL_REPORT_MATLAB.md](TECHNICAL_REPORT_MATLAB.md) — Full system report
- [MATLAB_INSTALL_GUIDE.md](MATLAB_INSTALL_GUIDE.md) — Installation steps

---

*ADAS Vision — SIH 2026*
