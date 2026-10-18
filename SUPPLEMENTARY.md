# ADAS Vision — Supplementary Technical Material

> **Smart India Hackathon Submission**  
> **Team:** Krishna & Akshit (`krishna2077-s / akshit-mat`)  
> **Repository:** https://github.com/krishna2077-s/adas-vision  
> **Advisory / Research Only — never wired to any vehicle's controls.**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Full System Architecture](#2-full-system-architecture)
3. [Mathematical & Algorithmic Formulation](#3-mathematical--algorithmic-formulation)
4. [Decision Rules — R1–R7 Safety Spine](#4-decision-rules--r1r7-safety-spine)
5. [Benchmark Validation — 5 Indian Road Scenarios](#5-benchmark-validation--5-indian-road-scenarios)
6. [Inference Backend Benchmarks](#6-inference-backend-benchmarks)
7. [Hardware Deployment Guide](#7-hardware-deployment-guide)
8. [Training Roadmap](#8-training-roadmap)
9. [Quickstart & CLI Reference](#9-quickstart--cli-reference)
10. [Validation & Safety Audit Protocol](#10-validation--safety-audit-protocol)

---

## 1. Executive Summary

Autonomous navigation in developing countries faces challenges fundamentally distinct from Western highway environments. Indian roads feature **unstructured geometry**, frequent **absence of painted lane markings**, **highly heterogeneous mixed traffic** (two-wheelers, auto-rickshaws, commercial trucks, pushcarts, and stray cattle), and **unpredictable actor trajectories**. Traditional pipelines that depend on structured HD maps and constant-velocity lane-following fail catastrophically in these conditions.

**ADAS Vision** is an end-to-end, closed-loop autonomous driving architecture specifically engineered for unstructured Indian road conditions, integrating:

1. **Multi-tier Perception Ladder** — Deep neural drivable area segmentation (LRASPP MobileNetV3 fine-tuned on the Indian Driving Dataset) + YOLOv8 + classical texture/chroma fallback for completely unpainted roads.
2. **Class-Aware Trajectory Prediction** — Social-force inspired motion forecasting with class-dependent lateral uncertainty inflation ($\pm2.0\text{ m}$ for cattle, $\pm1.5\text{ m}$ for pedestrians).
3. **Dynamic Occupancy Grids & Real-Time Hybrid A\* Replanning** — Vectorized $60\text{ m} \times 30\text{ m}$ spatial grid updating at $30\text{ Hz}$ with sub-millisecond ($0.04$–$0.2\text{ ms}$) kinematically feasible spline-smoothed path replanning.
4. **Deterministic R1–R7 Safety Spine with Temporal Ratchet** — Formally verifiable state arbitration featuring $N$-of-$M$ voting escalation, asymmetric de-escalation dampening, and a non-negotiable Vulnerable Road User (VRU) proximity safety floor.
5. **Non-linear Kinematic Bicycle Dynamics** — Closed-loop Pure Pursuit lateral tracking and PID longitudinal control.

### Project Completion Status

| Component | Status |
|---|:---:|
| AI Perception & Vision Engine (Modules 1–4, 10–11) | ✅ 100% Complete |
| Python ↔ MATLAB UDP Bridge | ✅ 100% Working |
| All 5 Indian Road Scenarios (MATLAB) | ✅ 100% Tested |
| Hybrid A\* & Control Pipeline | ✅ Replan latency ~0.1–0.2 ms |
| Automated Metrics & CSV Export | ✅ Working |
| Model Export (ONNX + OpenVINO INT8) | ✅ Complete |
| Traffic Sign Classifier | ✅ **99.9% val accuracy** on GTSRB |

---

## 2. Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          1. SENSOR & PERCEPTION LAYER                           │
│  • Monocular Dashcam Video / Camera Stream                                      │
│  • Deep Drivable Surface Segmentation (IDD-trained LRASPP CNN)                  │
│  • YOLOv8n Multi-Class Detector (Pedestrians, Two-Wheelers, Cattle, Rickshaws)  │
│  • SORT-lite 2D Tracker with Inter-frame Kalman Kinematics                      │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Tracks & Drivable Bounds
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          2. PREDICTION & WORLD MODEL                            │
│  • Constant Velocity + Class-Dependent Lateral Uncertainty Inflation            │
│  • Dynamic 2D Binary Occupancy Grid (60m × 30m, Δ=0.25m)                       │
│  • Stamp Current Footprints + Forecast Trajectory Cones                         │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Dynamic Grid + Predictions
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      3. DECISION ENGINE (R1–R7 Safety Spine)                    │
│  • Hierarchical: R1 (E-STOP) → R2/R3 (BRAKE) → R4/R5 (SLOW) → R6 → R7        │
│  • N-of-M Voting Temporal Ratchet & 15-frame Emergency Latch                   │
│  • VRU Proximity Safety Floor (d ≤ 15m ⟹ min CAUTION)                          │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Velocity Target + Hazard Bounds
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        4. ADAPTIVE PATH PLANNER (Hybrid A*)                     │
│  • Real-time Replan Trigger: Path-Prediction Intersection Check (< 2.0m buffer) │
│  • Hybrid A* Kinematic Search on Binary Occupancy Map                           │
│  • Cubic Spline Waypoint Smoothing (Δs = 0.5m)                                 │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Waypoints & Target Velocity
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           5. VEHICLE DYNAMICS & CONTROL                         │
│  • Pure Pursuit Steering (lookahead = 6.0m)                                     │
│  • PID Longitudinal Throttle/Brake with Emergency Override (−7.0 m/s²)          │
│  • Non-linear Kinematic Bicycle Dynamics Integration                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Perception Tier Detail (Module 1)

The lane/road module operates as a **priority cascade** — each tier activates only when the tier above it fails:

```
Tier 1: Painted Lane Markings (lane_detection.py)
  ↓ confidence = 0.0 (no paint detected)
Tier 2: Learned Drivable Area (learned_road_detection.py — LRASPP on IDD)
  ↓ not available (no weights / torch missing)
Tier 3: Classical Road Detection (road_detection.py — colour/texture heuristic)
  ↓ all fail
DEGRADED mode — engine holds last known state, outputs CAUTION
```

This ensures the system **always has a best-effort answer** and **never silently fails**.

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Kinematic Bicycle Model

The vehicle state vector $\mathbf{x} = [x, y, \psi, v]^T$ evolves according to:

$$\dot{x} = v \cos(\psi + \beta)$$
$$\dot{y} = v \sin(\psi + \beta)$$
$$\dot{\psi} = \frac{v}{L} \tan(\delta) \cos(\beta)$$
$$\dot{v} = a$$

where $L = 2.70\text{ m}$ is the wheelbase, $\delta$ is the steering angle ($|\delta| \le 35°$), and side-slip angle $\beta = \arctan\!\left(\frac{l_r}{L}\tan(\delta)\right)$.

### 3.2 Pure Pursuit Lateral Steering

Given lookahead distance $L_a = 6.0\text{ m}$ and goal waypoint $(x_g, y_g)$ along the Hybrid A\* spline:

$$\alpha = \text{atan2}(y_g - y,\ x_g - x) - \psi$$
$$\delta = \text{atan2}\!\left(\frac{2L\sin(\alpha)}{L_a}\right)$$

### 3.3 Class-Dependent Trajectory Uncertainty Inflation

For tracked actor $i$ at $(x_i, y_i)$ with velocity $(v_{x,i}, v_{y,i})$ over prediction horizon $T = 3.0\text{ s}$:

$$x_i(t) = x_i + v_{x,i}\,t, \qquad y_i(t) = y_i + v_{y,i}\,t$$

Lateral uncertainty covariance $\sigma_y$ by traffic class:

$$\sigma_y = \begin{cases}
2.0\text{ m} & \text{class} \in \{\text{cow, animal}\} \\
1.5\text{ m} & \text{class} \in \{\text{pedestrian, VRU}\} \\
1.0\text{ m} & \text{class} \in \{\text{motorcycle, bicycle}\} \\
0.5\text{ m} & \text{class} \in \{\text{car, truck, bus}\}
\end{cases}$$

Cattle and pedestrians receive the highest uncertainty because their trajectories are most erratic on Indian roads.

### 3.4 Replan Trigger Condition

Path replanning is triggered instantaneously if **any** of the following holds:

$$t - t_{\text{last}} \ge 200\text{ ms}$$
$$\lor \quad \min_{j,k} \|\mathbf{p}_{\text{path}}(s_j) - \mathbf{p}_{\text{pred},k}(t)\| < 2.0\text{ m}$$
$$\lor \quad \mathcal{M}(\mathbf{p}_{\text{path}}) = 1$$

where $\mathcal{M}$ is the occupancy map lookup. This ensures the path is replanned either periodically or immediately when a predicted trajectory intersects within the $2.0\text{ m}$ safety buffer.

### 3.5 Time-to-Collision (TTC)

Per-track TTC estimate used by the decision engine:

$$\text{TTC} = \frac{d}{\dot{d}} \quad \text{if } \dot{d} > 0 \text{ (closing)}, \quad \text{else } \infty$$

where $d$ is smoothed distance (exponential moving average, $\alpha = 0.3$) and $\dot{d}$ is closing speed. Only used when $\dot{d} > 0$ — a receding object is never classified as a hazard.

### 3.6 Monocular Distance Estimation

Object distance from bounding box height using calibrated focal length:

$$d = \frac{f \cdot H_{\text{real}}}{h_{\text{px}}}$$

where $f$ is the focal length in pixels, $H_{\text{real}}$ is the known real-world height of the object class (e.g., $H_{\text{car}} = 1.5\text{ m}$, $H_{\text{pedestrian}} = 1.7\text{ m}$), and $h_{\text{px}}$ is the bounding box height in pixels.

---

## 4. Decision Rules — R1–R7 Safety Spine

The decision engine is a **deterministic, first-match-wins priority table**. Collision rules always outrank lane guidance. A single noisy frame cannot flip the decision — the temporal ratchet requires $N$-of-$M$ frames to escalate and de-escalates only one level per `HOLD_FRAMES`.

| Priority | Rule | Trigger Condition | Longitudinal Action | Speed Target |
|:---:|---|---|---|:---:|
| **1** | **R1 — EMERGENCY STOP** | $d \le 5.0\text{ m}$ **OR** $\text{TTC} \le 1.2\text{ s}$ (closing) | Max brake (1.0) + HOLD steering | $0\text{ km/h}$ |
| **2** | **R2 — BRAKE** | $\text{TTC} \le 2.5\text{ s}$ (+ $0.8\text{ s}$ margin for VRUs/cattle) | Proportional brake | $0\text{ km/h}$ |
| **3** | **R3 — BRAKE** | Static high-risk obstacle in-path $\le 8.0\text{ m}$ | Controlled stop | $0\text{ km/h}$ |
| **4** | **R4 — SLOW** | Red/Amber light **OR** Stop sign $\le 25.0\text{ m}$ | Decelerate | $20\text{ km/h}$ |
| **5** | **R5 — SLOW** | Medium-risk obstacle in-path $\le 20.0\text{ m}$ | Reduce speed | $20\text{ km/h}$ |
| **6** | **R6 — CAUTION** | $\text{TTC} \le 4.0\text{ s}$ OR obstacle approaching path | Monitor | $30\text{ km/h}$ |
| **7** | **R7 — PROCEED** | Path clear, lane confidence $> 0.5$ | Cruise | $50\text{ km/h}$ |

**Safety invariants (by construction):**
- Collision rules (R1–R3) always outrank lane/sign rules (R4–R7)
- Lateral steering actions can **never** raise the longitudinal threat level
- Only *confirmed* tracks (seen $N$ of last $M$ frames) influence decisions — a one-frame YOLO false positive cannot trigger braking
- 15-frame emergency latch: once `EMERGENCY_STOP` is committed, it cannot be released until the threat has been absent for 15 consecutive frames
- VRU safety floor: any pedestrian/cyclist/cattle within $15\text{ m}$ forces minimum `CAUTION` regardless of TTC

---

## 5. Benchmark Validation — 5 Indian Road Scenarios

All scenarios executed in MATLAB (`run_all_scenarios.m`) with automated metric collection. Baseline = traditional lane-locked ADAS without adaptive replanning.

### Scenario Results Summary

| Scenario | Mode | Duration (s) | Path Roughness (rad/m) | Min Clearance (m) | Collisions | Result |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **1. Unmarked Village Road** | Baseline | 24.0 | 0.0018 | 1.62 | 0 | PASS |
| | **Ours (Adaptive)** | **23.8** | **0.0009** ↓50% | **3.60** ↑122% | **0** | ✅ PASS |
| **2. Urban Intersection** | Baseline | 26.2 | 0.0022 | 1.10 | 0 | PASS |
| | **Ours (Adaptive)** | **25.3** | **0.0011** ↓50% | **2.45** ↑123% | **0** | ✅ PASS |
| **3. Highway Merge** | Baseline | 19.5 | 0.0015 | 1.20 | 0 | PASS |
| | **Ours (Adaptive)** | **18.9** | **0.0006** ↓60% | **2.80** ↑133% | **0** | ✅ PASS |
| **4. Dense Market Street** | Baseline | 32.0 | 0.0035 | 0.20 | **1** | ❌ FAIL |
| | **Ours (Adaptive)** | **28.4** | **0.0003** ↓91% | **0.44** ↑120% | **0** | ✅ PASS |
| **5. Cattle Crossing** | Baseline | 31.0 | 0.0028 | 1.80 | 0 | PASS |
| | **Ours (Adaptive)** | **30.5** | **0.0008** ↓71% | **3.90** ↑117% | **0** | ✅ PASS |

### Detailed Scenario Findings

**Scenario 1 — Unmarked Village Road** (`scenario_village_road.m`)  
Narrow 6m single-carriageway, no lane markings, S-curves, oncoming motorcycle (30 km/h), crossing pedestrian (4 km/h), parked pushcart. *93 events logged.* System detected the oncoming motorcycle at 24.6m, transitioned to `CAUTION` at $t=7.2\text{ s}$, and maintained **3.6m lateral clearance** — 2.2× the baseline — without panic braking.

**Scenario 2 — Unsignalised Urban Intersection** (`scenario_urban_intersection.m`)  
4-way junction, cross-traffic cars (20 km/h), turning auto-rickshaw, cyclists, multiple pedestrians in zebra zone. *191 events logged.* Cross-traffic triggered `SLOW` at 12.6m ($t=8.0\text{ s}$). Pedestrian entry at $t=8.6\text{ s}$ (9.9m away) triggered progressive `BRAKE` → controlled standstill at 6.0m buffer. Vehicle resumed and reached goal in 25.3s with **0 collisions**.

**Scenario 3 — High-Speed Highway Merge** (`scenario_highway_merge.m`)  
2-lane highway (70 km/h design speed), commercial truck merging abruptly at 25 km/h, high-speed SUV overtaking at 80 km/h. System identified closing gap ($38\text{ m} \to 18\text{ m}$), triggered `CAUTION` then progressive `BRAKE` to match truck velocity without tailgating, while correctly holding lane (SUV in adjacent lane, not in path).

**Scenario 4 — Dense Congested Market Street** (`scenario_dense_market.m`)  
Ultra-narrow 3.5m corridor, static pushcarts (thelas), wandering shoppers, oncoming scooter filtering through. *57 events logged.* Creeping speed (12–14 km/h). When scooter squeezed into corridor at $t=15.9\text{ s}$ (4.4m → 3.9m), the engine executed immediate **`EMERGENCY_STOP`**, eliminating collision risk. Baseline clipped static obstacles → **collision** at 0.20m. Ours: **0.44m clearance, 0 collisions**.

**Scenario 5 — Sudden Cattle Crossing** (`scenario_cattle_crossing.m`)  
Straight road, two cattle walking onto road at $t=3.0\text{ s}$ (3 km/h). *224 events logged.* System detected cattle early at 44.6m ($t=2.5\text{ s}$, `CAUTION`). `BRAKE` engaged at 19.9m, vehicle stopped at **11.4m buffer** (comfortable for cattle's unpredictable movement). Goal reached in 30.5s, **0 contact**.

### Key Empirical Findings (Live Co-Simulation)

Live closed-loop co-simulation over a 200m goal run via Python ↔ MATLAB UDP bridge:

| Metric | Value |
|---|---|
| Goal position | $(200.0,\ 0.0)\text{ m}$ |
| Duration | $23.7\text{ s}$ |
| Distance covered | $197.1\text{ m}$ |
| Cruising speed | $30.0\text{ km/h}$ |
| Control cycles / replans | **717 replans** at ~30 Hz |
| Mean replanning latency | **$0.1$–$0.2\text{ ms}$** (initial peak: 20.7ms) |
| Real-time deadline (50ms) | ✅ Always met |
| Collisions | **0** |

---

## 6. Inference Backend Benchmarks

The drivable area model (LRASPP MobileNetV3-Large, 768×432 input) supports four inference backends, auto-selected at runtime from fastest to slowest:

| Backend | Latency | Speedup vs PyTorch | Hardware |
|---|:---:|:---:|---|
| PyTorch (CPU) | ~185 ms | 1× | Baseline, always available |
| ONNX Runtime | ~95 ms | **~1.9×** | CPU-optimized ONNX graph |
| OpenVINO FP16 | ~63 ms | **~2.9×** | Intel iGPU (UHD 620/770) |
| OpenVINO INT8 | ~32 ms | **~5.8×** | Intel iGPU, NNCF-quantized |

*Measured on Intel Core i7-8650U + UHD 620. No discrete GPU.*

### End-to-End Pipeline FPS (i7-8650U laptop, no CUDA)

| Configuration | FPS |
|---|:---:|
| Lanes only (classical) | ~45 FPS |
| Lanes + YOLO (synchronous) | ~14 FPS |
| Lanes + YOLO (async, Module 14) | ~22 FPS |
| Lanes + learned road (ONNX) | ~18 FPS |
| Lanes + learned road (OpenVINO INT8) | ~28 FPS |
| Full pipeline (all 14 modules) | ~12 FPS |

### Projected GPU Performance

| Hardware | Est. Dual-Net FPS | Notes |
|---|:---:|---|
| RTX 4050 (CUDA + async) | **60–100+ FPS** | PyTorch-CUDA auto-detected |
| RTX 4050 (TensorRT engine) | **100+ FPS** | `yolo export format=engine half=True` |
| Jetson Orin Nano 8GB | ~30–60 FPS | JetPack + TensorRT, ~7–15 W |
| Jetson Orin NX 16GB | ~60–100+ FPS | Headroom for higher-res models |

---

## 7. Hardware Deployment Guide

### Auto-Detection Logic

The codebase requires **zero code changes** for GPU deployment. `PREFER_CUDA = True` is the default:

```python
# config.py
PREFER_CUDA = True   # auto-uses CUDA when torch.cuda.is_available()
```

Both `object_detection.py` (YOLO) and `learned_road_detection.py` (LRASPP) route to CUDA automatically. On a CPU-only machine, OpenVINO/ONNX paths engage unchanged.

### CUDA Setup (Desktop / RTX GPU)

```bash
# 1. CUDA-enabled PyTorch (adjust cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt ultralytics

# 2. Verify CUDA is active
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Run with async detection (YOLO on GPU background thread)
python main.py --video dashcam.mp4 --async
```

### TensorRT Export (Maximum Performance on RTX / Jetson)

```bash
# Export YOLO to TensorRT engine ON the target device (engines are device-specific)
yolo export model=yolov8n.pt format=engine half=True imgsz=640
```

Then in `config.py`:
```python
YOLO_TRT_MODEL = "yolov8n.engine"   # used automatically when CUDA is present
```

### OpenVINO Export (Intel iGPU — Laptop / NUC)

```bash
# Export road model to OpenVINO FP16 + INT8
python export/export_onnx.py          # step 1: PyTorch → ONNX
python export/export_openvino.py      # step 2: ONNX → FP16/INT8 IR
```

Files produced: `drivable_idd_lraspp_768x432_ov_fp16.{xml,bin}` and `*_ov_int8.{xml,bin}`.

### On-Camera Edge (Axis ARTPEC-8)

An alternative architecture: run YOLO detection **on the camera's DLPU** as per-tensor INT8 TFLite via the larod runtime (`export/export_yolo_tflite.py`), so object detection never competes for a host GPU. The road model and decision logic run on the host consuming camera detections. Best fit for camera-centric deployments.

---

## 8. Training Roadmap

### A. Drivable Area Segmentation (LRASPP)

**Architecture:** LRASPP MobileNetV3-Large, 3 output classes (0=background, 1=drivable, 2=alt-drivable), input 768×432.

**Datasets:**
- **IDD** (Indian Driving Dataset) — 10k images, Indian roads, JSON polygon annotations
- **BDD100K** — 70k images, global roads, pre-built drivable area masks

**Adverse augmentation pipeline** (`training/adverse_aug.py`):

| Augmentation | Simulates |
|---|---|
| `RandomBrightnessContrast(brightness_limit=(-0.5, 0.1))` | Night driving |
| `RandomFog(fog_coef_range=(0.2, 0.6))` | Haze / fog |
| `MotionBlur(blur_limit=(7, 21))` | Camera vibration |
| `RandomShadow` | Tree shadows, overpasses |
| `GaussNoise(std_range=(0.05, 0.15))` | Low-light sensor noise |

**Training commands:**

```bash
# CPU fine-tuning (from pretrained weights, ~3 hrs on i7)
python training/train_local.py

# GPU training on BDD100K (weighted sampler for night/adverse frames)
python training/train_bdd.py --epochs 50 --batch-size 16

# Kaggle GPU training (T4 x2, ~2–4 hrs)
# Upload kaggle/notebooks/sih_adas_train.ipynb to Kaggle
# Attach: krishna22b/adas-weights  +  krishna22b/idd-segmentation
```

**Export to ONNX:**

```bash
python export/export_onnx.py
# Produces: drivable_idd_lraspp_768x432.onnx
# Validates numerical parity against PyTorch output
```

### B. Traffic Sign Classifier (GTSRB CNN)

Lightweight custom CNN, 43 GTSRB classes, ~0.5M parameters. Achieves **99.9% validation accuracy**.

```bash
python training/train_signs.py
# Output: gtsrb_sign_cnn.pth
# Completes in ~5 min on RTX 4050
```

### C. Configuration Tuning

| What to change | Where | Parameter |
|---|---|---|
| Detection sensitivity | `config.py` | `OBJ_CONF_THRESHOLD` |
| Braking distance | `decision_engine.py` | `_rule_1` (5.0m), `_rule_2` (TTC 2.5s) |
| New object class | `config.py` | Add to `CLASS_REAL_HEIGHTS` + `RELEVANT_CLASSES` |
| Inference backend | `config.py` | `LEARNED_BACKEND = "auto"/"onnx"/"openvino"/"torch"` |
| ROI trapezoid | `config.py` | `ROI_TOP_Y`, `ROI_BOTTOM_LEFT_X`, etc. |

---

## 9. Quickstart & CLI Reference

### Installation

```bash
# Full install (GPU + OpenVINO)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Minimal install (CPU only)
pip install opencv-python numpy ultralytics torchvision
```

### Main CLI (`python main.py`)

```bash
# Webcam, all modules
python main.py --camera

# Video file, all modules
python main.py --video dashcam.mp4

# With async YOLO (GPU background thread) + save output
python main.py --video dashcam.mp4 --async --save output.mp4

# Lanes only (no YOLO — useful if ultralytics not installed)
python main.py --video dashcam.mp4 --no-objects

# With drive logging + driver monitoring
python main.py --video dashcam.mp4 --log drive.jsonl --driver-cam 1

# Debug overlay (shows ROI trapezoid + raw Hough lines)
python main.py --video dashcam.mp4 --debug
```

### Universal Launcher (`python run.py`)

```bash
python run.py main    --video dashcam.mp4 --async  # Main ADAS pipeline
python run.py bridge  --video dashcam.mp4 --async  # UDP bridge for MATLAB
python run.py test                                  # Test UDP bridge output
python run.py demo                                  # Gradio web demo
```

### Keyboard Controls (while running)

| Key | Action |
|---|---|
| `Q` | Quit |
| `D` | Toggle debug overlay |
| `P` | Pause / resume |
| `S` | Save screenshot |

### MATLAB Co-Simulation

```matlab
% 1. Start Python bridge in PowerShell:
%    python run.py bridge --video dashcam.mp4 --async

% 2. In MATLAB (set CWD to adas-vision/matlab):
run_all_scenarios           % Run all 5 scenarios + CSV export
run_cosimulation([200, 0])  % Live co-simulation to goal (200m, 0m)
scenario_cattle_crossing    % Individual scenario
```

---

## 10. Validation & Safety Audit Protocol

Run these **in order** on any new hardware before reporting results.

### Step 1 — Unit Tests (Decision Logic)

```bash
python -m pytest evaluation/test_adas.py -q
# Expected: 31 passed, 0 failed
```

### Step 2 — Confirm Correct Backend

```bash
python -c "
from perception.object_detection import ObjectDetector as O
from perception.learned_road_detection import LearnedRoadDetector as R
import logging; logging.disable(30)
print('YOLO backend   :', O(1280, 720).backend)
r = R(1280, 720)
print('Road backend   :', r.backend)
"
# On CUDA machine: YOLO cuda  /  Road torch (device=cuda)
# On CPU machine:  YOLO cpu   /  Road openvino_int8 (or onnx)
```

### Step 3 — Safety Spine Audit

```bash
python evaluation/safety_audit.py --video dashcam.mp4
# Expected: VERDICT spine HOLDS, 0 violations
# If violations > 0: DO NOT report performance numbers — investigate first
```

### Step 4 — Throughput Check

```bash
python main.py --video dashcam.mp4 --async
# Watch the FPS counter + DEGRADED chip in the HUD
# Clean result: steady FPS, DEGRADED rarely appears near objects
```

### Step 5 — Speed Benchmark

```bash
python evaluation/bench_speed.py --video dashcam.mp4
# Prints per-module latency table (ms)
```

---

*ADAS Vision — Smart India Hackathon research prototype. Advisory / simulation only.*  
*Repository: https://github.com/krishna2077-s/adas-vision*
