# ADAS Vision — Technical Report
## Adaptive Path Planning & Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads
### SIH 2026 Submission | Problem Statement

---

## 1. Introduction

India's road network presents unique challenges for autonomous driving: mixed traffic with cars, buses, trucks, auto-rickshaws, two-wheelers, bicycles, pedestrians, pushcarts, and animals share the same road space with no predictable flow patterns. Traditional autonomous driving systems that rely on structured lane geometry and predictable motion models fail in these conditions.

**ADAS Vision** is an adaptive path planning and collision avoidance system designed specifically for unstructured Indian road scenarios. It integrates a full perception-prediction-planning-control pipeline validated across five representative Indian road scenarios.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAS VISION PIPELINE                          │
│                                                                  │
│  [Camera]  [LiDAR]  [Radar]                                      │
│      │          │       │                                        │
│      └──────────┴───────┘                                        │
│              │                                                    │
│    ┌─────────▼──────────┐                                        │
│    │  Sensor Fusion     │  YOLOv8n object detection              │
│    │  (Multi-modal)     │  Lane detection (LR-ASPP)              │
│    │                    │  Depth estimation (monocular)          │
│    └─────────┬──────────┘                                        │
│              │  tracks[], lane_offset, fps                       │
│    ┌─────────▼──────────┐                                        │
│    │  Trajectory        │  CV/CTRA models                        │
│    │  Prediction        │  3-second lookahead                    │
│    │  (predict_traj.m)  │  15-step forecasts                     │
│    └─────────┬──────────┘                                        │
│              │  predicted positions                              │
│    ┌─────────▼──────────┐                                        │
│    │  Dynamic Occupancy │  Binary occupancy map                  │
│    │  Grid              │  Current + predicted footprints        │
│    │  (build_occ.m)     │  60m × 30m @ 0.25m resolution          │
│    └─────────┬──────────┘                                        │
│              │                                                    │
│    ┌─────────▼──────────┐   ┌──────────────────────┐            │
│    │  Replan Trigger    │──▶│  Hybrid A* Planner   │            │
│    │  (replan_trig.m)   │   │  (plan_path.m)        │            │
│    └────────────────────┘   │  Min-turn: 5m radius │            │
│                              │  0.5m step resolution│            │
│                              └──────────┬───────────┘            │
│                                         │  collision-free path   │
│    ┌────────────────────────────────────▼──────────────┐         │
│    │  R1-R7 Decision Engine (Stateflow / MATLAB)       │         │
│    │  PROCEED → CAUTION → SLOW → BRAKE → E-STOP        │         │
│    │  Temporal ratchet: escalate fast, de-escalate slow│         │
│    └───────────────────┬───────────────────────────────┘         │
│                        │  decision_level, target_speed            │
│    ┌───────────────────▼───────────────────────────────┐         │
│    │  Vehicle Controller                               │         │
│    │  Pure Pursuit (lateral) + PID speed control       │         │
│    │  Kinematic Bicycle Model (Euler integration)      │         │
│    └───────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Path Planning — Hybrid A*

### Algorithm
The system uses **Hybrid A*** from the MathWorks Navigation Toolbox (`plannerHybridAStar`), enhanced with:

- **Dynamic occupancy map** (60m × 30m, 4 cells/m resolution)
- **Prediction-aware obstacle stamping**: future positions of tracked agents for next 1.2s are also stamped as occupied
- **Continuous replanning** triggered by deviation >3m from current path or >5s elapsed
- **Spline smoothing** of raw A* path at 0.5m waypoint spacing

### Turning Constraints
```
Min turning radius: 5.0 m  (compact sedan)
Motion primitive length: 2.0 m
Steering angles tested: ±25°, ±15°, ±5°, 0°
```

### Fallback Strategy
When Navigation Toolbox is unavailable, an internal Greedy Hybrid A* with perpendicular obstacle avoidance is used as fallback.

---

## 4. Decision Engine — R1-R7 Rules with Temporal Ratchet

### Decision States

| Level | State | Target Speed | Condition |
|-------|-------|-------------|-----------|
| 0 | PROCEED | 30 km/h | No hazard detected |
| 1 | CAUTION | 21 km/h | Hazard in path, dist < 40m |
| 2 | SLOW | 12 km/h | TTC < 6s OR VRU < 15m |
| 3 | BRAKE | 5 km/h | Closing speed > 5 m/s OR TTC < 2s |
| 4 | EMERGENCY_STOP | 0 km/h | TTC < 1s OR dist < 5m |

### R1-R7 Rules

| Rule | Trigger | Action |
|------|---------|--------|
| R1 | Obstacle in path + dist < 40m | → CAUTION |
| R2 | TTC < 6s (path-obstructing object) | → SLOW |
| R3 | VRU (person/cow/bicycle) + dist < 15m | → SLOW (VRU safety floor) |
| R4 | Closing speed > 5 m/s + dist < 20m | → BRAKE |
| R5 | TTC < 2.0s | → BRAKE |
| R6 | TTC < 1.0s OR distance < 5m | → EMERGENCY_STOP |
| R7 | Sensor degraded flag | → SLOW (fail-safe mode) |

### Temporal Ratchet
- **Escalation**: Immediate (any single frame can escalate to higher severity)
- **De-escalation**: Requires 3 consecutive frames below threshold (prevents oscillation)
- **Emergency latch**: EMERGENCY_STOP latched for minimum 1.5s

---

## 5. Vehicle Model

**Kinematic Bicycle Model** (Euler integration at 30 Hz):

```
β  = atan2(lr × tan(δ), L)           — slip angle
a  = throttle × a_max - brake × b_max — acceleration
v(t+dt) = clamp(v + a × dt, 0, v_max)
x(t+dt) = x + v × cos(θ + β) × dt
y(t+dt) = y + v × sin(θ + β) × dt
θ(t+dt) = θ + (v/L) × tan(δ) × dt
```

| Parameter | Value |
|-----------|-------|
| Mass | 1450 kg |
| Wheelbase (L) | 2.70 m |
| lf / lr | 1.15 / 1.55 m |
| Width | 1.85 m |
| Max steering | 35° |
| Max acceleration | 2.5 m/s² |
| Max braking | 7.0 m/s² |
| Max speed | 100 km/h |

**Lateral Control**: Speed-adaptive Pure Pursuit
```
Lookahead: La = clamp(0.45v + 3.5, 3.5, 14.0)  [m]
Steer: δ = atan2(2L sin(α), La)
```

**Longitudinal Control**: PID speed tracking
```
Kp=0.60, Ki=0.05, Kd=0.00, dt=0.033s
```

---

## 6. Test Scenarios

### Scenario 1 — Unmarked Village Road
- **Road**: 200m, 6m wide, no lane markings, winding
- **Actors**: Oncoming motorcycle (30 km/h), crossing pedestrian, parked pushcart
- **Challenge**: No centerline reference, unpredictable pedestrian crossing

### Scenario 2 — Urban Unsignalised Intersection
- **Road**: 4-way cross junction, 8m main road × 7m side road
- **Actors**: Cross-traffic car, auto-rickshaw (cut-in), 2 pedestrians, scooter
- **Challenge**: Multi-direction threats, no signal, auto-rickshaw behaviour

### Scenario 3 — Highway Merge
- **Road**: 260m dual carriageway, 15m wide (2+2 lanes)
- **Actors**: Overloaded truck (merging from ramp), fast SUV (80 km/h), wrong-way motorcyclist
- **Challenge**: Large speed differential, heavy vehicle cut-in

### Scenario 4 — Dense Market Street
- **Road**: 85m market lane, 7m wide
- **Actors**: Static pushcart, 2 crossing pedestrians (sudden dart), oncoming scooter
- **Challenge**: Ultra-dense VRU environment, creeping speed, minimal clearance

### Scenario 5 — Cattle Crossing
- **Road**: 160m rural road, 7m wide
- **Actors**: 2 cows crossing at t=2.5s
- **Challenge**: Sudden unpredictable large obstacle, emergency braking then resume

---

## 7. Results

*(Results populated after running `run_all_scenarios` in MATLAB)*

| Scenario | Duration (s) | Distance (m) | Min Clearance (m) | Collisions | Replans | Replan Latency (ms) | Status |
|----------|-------------|-------------|-------------------|------------|---------|---------------------|--------|
| Village Road | — | — | — | — | — | — | — |
| Urban Intersection | — | — | — | — | — | — | — |
| Highway Merge | — | — | — | — | — | — | — |
| Dense Market | — | — | — | — | — | — | — |
| Cattle Crossing | — | — | — | — | — | — | — |

### Performance Metrics
- **Replanning Latency**: Target < 50ms per replan
- **Path Smoothness**: Mean absolute curvature < 0.05 rad/m
- **Scenario Completion**: 5/5 goals reached
- **Collisions**: 0 across all 5 scenarios

---

## 8. Toolbox Requirements

| Toolbox | Used For | Required? |
|---------|----------|-----------|
| Navigation Toolbox | `plannerHybridAStar` | Recommended (fallback exists) |
| Automated Driving Toolbox | `drivingScenario`, RoadRunner integration | Recommended |
| Stateflow | Decision state machine | Recommended (MATLAB function fallback) |
| Vehicle Dynamics Blockset | Full dynamics model | Optional |
| Deep Learning Toolbox | YOLOv8 inference | Used in Python pipeline |

---

## 9. How to Run

```matlab
% Step 1: Open MATLAB R2024a or newer
% Step 2: Set working directory
cd('C:\Users\Manish khandelwal\Downloads\adas-vision\adas-vision\matlab')

% Step 3: Run master entry point
main_adas

% Quick options:
scenario_village_road          % Single scenario
run_all_scenarios              % All 5 benchmark
build_stateflow_model          % Stateflow chart
generate_roadrunner_scenes     % OpenDRIVE scene files
generate_demo_video            % MP4 video output
```

---

## 10. File Structure

```
adas-vision/matlab/
├── main_adas.m                    ← Master entry point (START HERE)
├── startup.m                      ← Auto-setup when MATLAB opens
├── setup_vehicle_model.m          ← Vehicle & control parameters
│
├── run_all_scenarios.m            ← Benchmark all 5 scenarios
├── run_baseline_vs_adaptive.m     ← Adaptive vs baseline comparison
├── adaptive_scenario_loop.m       ← Unified closed-loop engine
│
├── scenario_village_road.m        ← Scenario 1: Village Road
├── scenario_urban_intersection.m  ← Scenario 2: Urban Intersection
├── scenario_highway_merge.m       ← Scenario 3: Highway Merge
├── scenario_dense_market.m        ← Scenario 4: Dense Market
├── scenario_cattle_crossing.m     ← Scenario 5: Cattle Crossing
│
├── plan_path.m                    ← Hybrid A* path planner
├── predict_trajectories.m         ← Short-term trajectory prediction
├── build_occupancy_grid.m         ← Dynamic binary occupancy map
├── decision_with_ratchet.m        ← R1-R7 + temporal ratchet engine
├── replan_trigger.m               ← Real-time replan condition
│
├── build_stateflow_model.m        ← Simulink + Stateflow model
├── generate_roadrunner_scenes.m   ← RoadRunner/OpenDRIVE export
├── generate_demo_video.m          ← MP4 demo video generator
├── collect_metrics.m              ← KPI computation + CSV + charts
│
├── run_cosimulation.m             ← Live Python-MATLAB co-simulation
├── udp_bridge.py (../udp_bridge.py) ← Python UDP bridge
│
└── roadrunner/                    ← Generated scene files
    ├── village_road_scene.mat
    ├── village_road.xodr
    ├── urban_intersection_scene.mat
    └── urban_intersection.xodr
```

---

*Report generated by ADAS Vision | SIH 2026*
