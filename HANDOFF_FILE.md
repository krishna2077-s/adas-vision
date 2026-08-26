# ADAS Vision: Developer Handoff Guide

> **Last Updated:** 2026-08-26  
> **Updated By:** Krishna & Akshit (`krishna2077-s / akshit-mat`)  
> **Repo:** https://github.com/krishna2077-s/adas-vision  
> **Target Hardware:** Any consumer PC/Laptop (Optimized for CPU, OpenVINO, or NVIDIA CUDA GPUs)

---

## ⚡ Current Project Status & Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. AI PERCEPTION & VISION ENGINE  : 100% COMPLETE & VERIFIED                │
│ 2. PYTHON ↔ MATLAB UDP BRIDGE     : 100% WORKING (Bidirectional ports 5005/5006)
│ 3. 5/5 INDIAN ROAD SCENARIOS      : 100% COMPLETE & TESTED IN MATLAB        │
│ 4. HYBRID A* & CONTROL PIPELINE   : 100% WORKING (Replan latency ~0.1-0.2ms)│
│ 5. AUTOMATED METRICS & CSV EXPORT : 100% WORKING (run_all_scenarios.m)      │
│ 6. ROADRUNNER 3D SCENES (.rrscene): 🔄 IN PROGRESS                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Phase | Description | Status |
|---|---|:---:|
| **Phase 1** | Environment setup (venv, CUDA PyTorch, dependencies) | ✅ COMPLETE |
| **Phase 2** | Traffic sign model training (`train_signs.py` → `gtsrb_sign_cnn.pth`) | ✅ COMPLETE — **99.9% val accuracy** |
| **Phase 3** | Folder refactor (`core/`, `perception/`, `simulation/`, `evaluation/`, `training/`) | ✅ COMPLETE |
| **Phase 4** | Universal launcher (`run.py`) handling all module imports | ✅ COMPLETE |
| **Phase 5** | UDP co-simulation bridge (`run.py bridge`, `run_cosimulation.m`) | ✅ COMPLETE & TESTED |
| **Phase 6** | Path planner (Hybrid A* + spline) & Kinematic Bicycle Controller | ✅ COMPLETE |
| **Phase 7** | All 5 Mandatory Indian Driving Scenarios | ✅ ALL 5 COMPLETE |
| **Phase 8** | Multi-Scenario Benchmark Suite (`run_all_scenarios.m` + CSV export) | ✅ COMPLETE |
| **Phase 9** | RoadRunner 3D visual scenes (`.rrscene` files) | 🔄 IN PROGRESS |
| **Phase 10** | Technical Report & Video Demonstration | ⏳ READY TO ASSEMBLE |

---

## 🚀 Quickstart: How to Run the Entire System

### 1. Unified Python Launcher (`run.py`)
Run all commands from the root directory:

```powershell
# 1. Run main ADAS Vision on dashcam (or any internet video)
python run.py main --video dashcam.mp4 --async

# 2. Run the UDP Bridge for MATLAB Co-simulation
python run.py bridge --video dashcam.mp4 --async

# 3. Test the UDP Bridge Output (in a separate terminal)
python run.py test

# 4. Launch the Gradio Web Demo
python run.py demo
```

---

### 2. MATLAB Co-Simulation & Scenarios (`matlab/` folder)

Open **MATLAB R2026a**, set the Current Folder to `adas-vision/matlab`, and run any of the following:

#### A. Run Automated Benchmark Across All 5 Scenarios
```matlab
run_all_scenarios
```
* Runs all 5 scenarios back-to-back.
* Prints a consolidated comparative performance table.
* Automatically exports `metrics_all_scenarios_summary.csv`.
* Displays a multi-panel comparison bar chart.

#### B. Run Individual Scenarios Standalone
```matlab
scenario_village_road          % Scenario 1: Unmarked road, oncoming motorcycle, VRUs
scenario_urban_intersection    % Scenario 2: 4-way junction, rickshaw, cyclists, pedestrians
scenario_highway_merge         % Scenario 3: Commercial truck cut-in, fast overtaking SUV
scenario_dense_market          % Scenario 4: 4m street, pushcarts, darting shoppers, scooter
scenario_cattle_crossing       % Scenario 5: Sudden cow intrusion, emergency stop
```

#### C. Run Live Co-Simulation with Python Perception
```matlab
% 1. Start Python bridge in PowerShell:  python run.py bridge --video dashcam.mp4 --async
% 2. In MATLAB Command Window:
run_cosimulation([200, 0])
```

---

## 🗂️ Project Directory Structure

```
adas-vision/
├── run.py                       # Universal root launcher (adds subfolders to sys.path)
├── HANDOFF_FILE.md              # Current developer handoff guide
├── README.md                    # Project documentation & architecture overview
├── dashcam.mp4                  # Sample test video
│
├── core/                        # Central system orchestration
│   ├── main.py                  # Main ADAS pipeline entrypoint
│   ├── config.py                # Global parameters, thresholds, and model paths
│   ├── decision_engine.py       # Deterministic R1–R7 safety rule engine + ratchet
│   ├── frame_guard.py           # Camera frame health and corrupt-buffer protector
│   └── drive_logger.py          # Real-time event & decision logger (.jsonl)
│
├── perception/                  # Vision & neural perception modules
│   ├── object_detection.py      # Synchronous YOLOv8 detector
│   ├── async_detector.py        # Background GPU worker thread for YOLO
│   ├── lane_detection.py        # Classical CV (Gaussian/Canny/Hough) painted lane detector
│   ├── learned_road_detection.py# LRASPP MobileNetV3 CNN (IDD trained for unpainted roads)
│   ├── road_detection.py        # Classical color/texture flood-fill fallback
│   ├── tracker.py               # Constant-velocity SORT tracker with TTC estimation
│   ├── traffic_light_state.py   # HSV state classifier (Red/Amber/Green)
│   └── traffic_sign_recognition.py # 43-class GTSRB sign classifier CNN
│
├── simulation/                  # Co-simulation & advisory controllers
│   ├── udp_bridge.py            # Streams perception JSON (port 5005) & receives ego pose (port 5006)
│   ├── sensor_fusion.py         # Bayesian camera-radar fusion
│   ├── forward_collision_warning.py # Dynamic FCW audio/visual alert state
│   └── perception_bev.py        # 2D camera to Bird's-Eye-View projection
│
├── matlab/                      # MATLAB / Simulink simulation suite
│   ├── startup.m                # Auto-configuration and path setup script
│   ├── setup_vehicle_model.m    # Bicycle dynamics, PID, & Pure Pursuit parameter setup
│   ├── run_cosimulation.m       # Master 20 Hz co-simulation loop with live HUD animation
│   ├── run_all_scenarios.m      # Automated test runner for all 5 scenarios + CSV exporter
│   ├── collect_metrics.m        # Computes smoothness, latency, clearance, collisions -> CSV
│   ├── plan_path.m              # Hybrid A* kinematic path planner + cubic spline
│   ├── predict_trajectories.m   # Short-term motion predictor with VRU/cattle uncertainty
│   ├── decision_with_ratchet.m  # Stateful R1–R7 decision logic with temporal ratchet
│   ├── stateflow_reference.m    # Direct reference for Simulink Stateflow porting
│   ├── parse_udp_json.m         # Simulink MATLAB Function block UDP JSON parser
│   ├── scenario_village_road.m  # Scenario 1: Unmarked village road
│   ├── scenario_urban_intersection.m # Scenario 2: Unsignalised urban intersection
│   ├── scenario_highway_merge.m # Scenario 3: Highway merge with commercial vehicle cut-in
│   ├── scenario_dense_market.m  # Scenario 4: Congested market with pushcarts & VRUs
│   └── scenario_cattle_crossing.m # Scenario 5: Cattle crossing emergency braking
│
├── evaluation/                  # Testing & validation benchmarks
│   ├── test_udp_bridge.py       # Standalone Python UDP listener verification tool
│   ├── test_adas.py             # 31 unit tests validating decision rules
│   ├── safety_audit.py          # Zero-false-negative safety auditor
│   └── bench_speed.py           # Per-module millisecond latency profiler
│
├── training/                    # Model training & fine-tuning pipelines
│   ├── train_signs.py           # GTSRB sign classifier trainer (99.9% val accuracy)
│   └── train_bdd.py             # Drivable road segmentation fine-tuner
│
├── export/                      # Hardware acceleration exporters
│   ├── export_onnx.py           # ONNX exporter for road segmentation
│   ├── export_openvino.py       # OpenVINO FP16/INT8 exporter
│   └── export_yolo_onnx.py      # YOLO ONNX exporter
│
└── demo/                        # Interactive web demos
    └── web_demo.py              # Gradio web application
```

---

## 🧠 Perception & Decision Pipeline Rules (R1–R7)

The decision engine operates under a deterministic priority ladder with a **Temporal Ratchet** (fast alert escalation, hysteresis de-escalation) and a **Vulnerable Road User (VRU) Safety Floor**:

| Priority | Rule | Condition | Action |
|---|---|---|---|
| **1** | **R1 (EMERGENCY_STOP)** | Distance $\le 5.0\text{ m}$ OR $\text{TTC} \le 1.2\text{ s}$ ($\dot{d} > 0$) | Max Brake (1.0), Target speed: $0\text{ km/h}$, Steering: HOLD |
| **2** | **R2 (BRAKE)** | $\text{TTC} \le 2.5\text{ s}$ ($\dot{d} > 0$) [+$0.8\text{s}$ margin for VRUs/Cattle] | Proportional Brake, Target speed: $0\text{ km/h}$ |
| **3** | **R3 (BRAKE)** | Static high-risk obstacle in-path $\le 8.0\text{ m}$ | Controlled Stop |
| **4** | **R4 (SLOW)** | Red/Amber traffic light OR Stop Sign $\le 25.0\text{ m}$ | Target speed: $20\text{ km/h}$ |
| **5** | **R5 (SLOW)** | Medium-risk obstacle in-path $\le 20.0\text{ m}$ | Target speed: $20\text{ km/h}$ |
| **6** | **R6 (CAUTION)** | $\text{TTC} \le 4.0\text{ s}$ OR obstacle approaching path | Target speed: $30\text{ km/h}$ |
| **7** | **R7 (PROCEED)** | Path clear, confidence $> 0.5$ | Nominal cruising speed ($50\text{ km/h}$) |

---

## 🚧 Team Deliverables Checklist for Submission

1. **RoadRunner 3D Scenes (Person 5):**
   - Install RoadRunner desktop application.
   - Build and export `village_road.rrscene` (unmarked narrow road, trees, dirt edges) and `urban_intersection.rrscene` (4-way junction, crosswalks, buildings).
   - Save files into `scenes/` folder.

2. **Simulink Model Diagram (Person 2):**
   - If judges require a `.slx` model file, wire `parse_udp_json.m` → `decision_with_ratchet.m` → `plan_path.m` → `Bicycle Model` inside Simulink.

3. **Technical Report (5–8 Pages):**
   - Compile problem statement, system architecture, R1–R7 ladder, and the CSV tables generated from `run_all_scenarios`.

4. **Demonstration Video (3–5 Minutes):**
   - Record screen showing side-by-side: Python vision bounding boxes/HUD and MATLAB/RoadRunner dynamic vehicle response.
