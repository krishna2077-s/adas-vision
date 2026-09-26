# ADAS Vision — In-Depth Technical Documentation

> **Advisory / Research Only** — This system is a research prototype. It must never be wired to any vehicle's controls.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Module Reference](#3-module-reference)
   - [Module 1 — Lane Detection](#module-1--lane-detection)
   - [Module 1b — Classical Road Detection](#module-1b--classical-road-detection)
   - [Module 1c — Learned Road Detection](#module-1c--learned-road-detection)
   - [Module 2 — Object Detection](#module-2--object-detection)
   - [Module 3 — Decision Engine](#module-3--decision-engine)
   - [Module 4 — Multi-Object Tracker](#module-4--multi-object-tracker)
   - [Module 5–9 — Advisory Simulation Stack](#module-59--advisory-simulation-stack)
   - [Module 10 — Forward Collision Warning](#module-10--forward-collision-warning)
   - [Module 11 — Scene Understanding](#module-11--scene-understanding)
   - [Module 12 — Drive Logger](#module-12--drive-logger)
   - [Module 14 — Async Object Detection](#module-14--async-object-detection)
4. [Data Pipeline & Training](#4-data-pipeline--training)
5. [Model Export & Inference Backends](#5-model-export--inference-backends)
6. [Configuration Reference](#6-configuration-reference)
7. [Running the System](#7-running-the-system)
8. [Performance Benchmarks](#8-performance-benchmarks)
9. [Project File Structure](#9-project-file-structure)
10. [SIH Context & Contributions](#10-sih-context--contributions)

---

## 1. Project Overview

ADAS Vision is a **multi-module Advanced Driver Assistance System** built entirely in Python. It processes dashcam video (or a live webcam) in real time and produces a continuous stream of advisory driving decisions — lane guidance, hazard warnings, speed advisories, forward collision alerts, and traffic sign/light awareness.

### Key Design Principles

| Principle | Implementation |
|---|---|
| **Graceful degradation** | Every module is optional; the system runs even if PyTorch, YOLO, or OpenVINO are absent |
| **No false authority** | Safety rules can only escalate decisions, never reduce them |
| **Honest failure** | If no lane markings detected → reports `NO LANE`, never invents one |
| **Indian-road first** | Trained on IDD (Indian Driving Dataset); handles unmarked roads, heterogeneous traffic |
| **Layered perception** | Paint lanes → learned road model → classical road fallback, in priority order |

### Quick Numbers

- **Real-time performance:** ~14–30 FPS on a laptop CPU (i7-8650U)
- **Drivable area model:** LRASPP MobileNetV3-Large, val mIoU ~0.92 on IDD
- **Object detection:** YOLOv8n, COCO 80-class
- **Export formats:** PyTorch `.pth`, ONNX, OpenVINO FP16/INT8

---

## 2. System Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              Input Source                │
                         │       (Video file / Webcam feed)         │
                         └────────────────────┬────────────────────┘
                                              │ raw frame
                    ┌─────────────────────────▼──────────────────────────┐
                    │                   core/main.py                      │
                    │              (orchestration loop)                   │
                    └──┬──────────┬──────────┬──────────┬───────────────┘
                       │          │          │          │
              ┌────────▼──┐  ┌────▼────┐  ┌─▼──────┐  ┌▼───────────┐
              │ Module 1  │  │Module 2 │  │Mod 11  │  │  Mod 5-9   │
              │  Lanes    │  │ Objects │  │Scene   │  │  Sim Stack │
              │ (3 layers)│  │ (YOLO)  │  │(Lights │  │ (Advisory) │
              └─────┬─────┘  └────┬────┘  │ Signs) │  └────────────┘
                    │             │       └────────┘
              ┌─────▼─────────────▼──────────────────────────────────┐
              │                Module 4: Tracker                      │
              │         (stable object IDs across frames)             │
              └──────────────────────┬───────────────────────────────┘
                                     │ tracks + lane_result
              ┌──────────────────────▼───────────────────────────────┐
              │              Module 3: Decision Engine                │
              │    Policy table R1–R7 → temporal ratchet → HUD       │
              └──────────────────────┬───────────────────────────────┘
                                     │ DrivingDecision
              ┌──────────────────────▼───────────────────────────────┐
              │          Module 10: Forward Collision Warning         │
              └──────────────────────┬───────────────────────────────┘
                                     │ annotated frame
              ┌──────────────────────▼───────────────────────────────┐
              │          Module 12: Drive Logger (opt-in)             │
              └──────────────────────────────────────────────────────┘
```

---

## 3. Module Reference

### Module 1 — Lane Detection
**File:** [`perception/lane_detection.py`](perception/lane_detection.py)

Detects painted lane markings using classical computer vision.

**Pipeline:**
1. Resize frame to 50% (`LANE_PROC_SCALE`) for speed
2. Convert to HSV → isolate white and yellow paint regions
3. Gaussian blur → Canny edge detection
4. Trapezoid ROI mask (configurable fractions of frame size)
5. Probabilistic Hough Transform → raw line segments
6. Filter by slope (`MIN_SLOPE`/`MAX_SLOPE`) → classify left/right
7. Temporal smoothing (exponential moving average) → stable lane lines
8. Compute `lane_center_x` for lateral guidance

**Outputs:** `LaneResult(left_line, right_line, lane_center_x, confidence)`

**Honest failure:** If no paint is detected on either side → `confidence=0.0` → system falls through to Module 1c/1b.

---

### Module 1b — Classical Road Detection
**File:** [`perception/road_detection.py`](perception/road_detection.py)

Fallback for unmarked roads (hills, rural areas). Uses colour/texture heuristics to identify the road surface in the lower ROI when no paint markings exist. Less accurate than the learned model but requires no trained weights.

---

### Module 1c — Learned Road Detection
**File:** [`perception/learned_road_detection.py`](perception/learned_road_detection.py)

**Primary** source for unmarked-road guidance. Uses a trained LRASPP MobileNetV3-Large segmentation model.

**Architecture:** LRASPP (Lite R-ASPP) with MobileNetV3-Large backbone
- Input: `768×432` RGB frame, ImageNet-normalized
- Output: 3-class segmentation mask (`0`=background, `1`=drivable, `2`=alt-drivable)
- Inference backends (auto-selected, fastest first):
  1. **OpenVINO INT8** — ~2.9× faster than ONNX on Intel iGPU
  2. **OpenVINO FP16** — slightly slower than INT8
  3. **ONNX Runtime** — ~1.5–3× faster than PyTorch on CPU
  4. **PyTorch** — always available, slowest

**Model files:**
| File | Size | Notes |
|---|---|---|
| `drivable_idd_full_best.pth` | 13 MB | PyTorch checkpoint |
| `drivable_idd_lraspp_768x432.onnx` | ~13 MB | ONNX export |
| `drivable_idd_lraspp_768x432_ov_fp16.*` | ~6.6 MB | OpenVINO FP16 IR |
| `drivable_idd_lraspp_768x432_ov_int8.*` | ~3.8 MB | OpenVINO INT8 IR |

---

### Module 2 — Object Detection
**File:** [`perception/object_detection.py`](perception/object_detection.py)

Runs YOLOv8n for 80-class COCO detection. Detects vehicles, pedestrians, cyclists, animals, and traffic elements.

**Key features:**
- In-path filtering: only objects whose bounding box overlaps the ego lane are flagged as hazards
- Confidence threshold: `OBJ_CONF_THRESHOLD` (default 0.4)
- Risk classification: distance estimate from bounding-box height + calibrated focal length
- Works synchronously (default) or on a background thread (Module 14)

**Model:** `yolov8n.pt` / `yolov8n.onnx` (~6–12 MB)

---

### Module 3 — Decision Engine
**File:** [`core/decision_engine.py`](core/decision_engine.py)

The arbitration brain. Fuses lane + object data into one driving decision per frame.

**Decision pipeline (6 stages):**

```
1. Trust check      → is lane / object data reliable?
2. Hazard selection → nearest confirmed in-path track
3. Policy table     → R1–R7 first-match-wins rules
4. Temporal ratchet → escalate fast, release slow (no flip-flopping)
5. Lateral          → steer toward lane center (safety-clamped)
6. Output           → DrivingDecision dataclass
```

**Longitudinal levels:**
| Level | Name | Meaning |
|---|---|---|
| 0 | `PROCEED` | Clear road, maintain speed |
| 1 | `CAUTION` | Object detected, watch |
| 2 | `SLOW` | Close object, reduce speed |
| 3 | `BRAKE` | Imminent hazard, brake |
| 4 | `EMERGENCY_STOP` | Collision imminent |

**Safety invariants:**
- Collision rules (R1–R3) always outrank lane rules
- Lateral actions can never raise the longitudinal threat level
- Only *confirmed* tracks (seen N-of-M frames) influence decisions
- Emergency latch: once `EMERGENCY_STOP`, must count down before release

---

### Module 4 — Multi-Object Tracker
**File:** [`perception/tracker.py`](perception/tracker.py)

Assigns stable IDs to detected objects across frames using IoU-based matching. Maintains per-track smoothed distance and closing speed estimates for the decision engine.

- Track confirmation: N consecutive detections before a track is "confirmed"
- Track deletion: M consecutive misses before removal
- Outputs smoothed kinematics: `distance_m`, `closing_speed_mps`, `TTC_s`

---

### Module 5–9 — Advisory Simulation Stack
**Files:** `perception_bev.py`, `sensor_fusion.py`, `prediction_planning.py`, `control_sim.py`, `driver_monitoring.py`

> ⚠️ **All advisory/simulated — never wired to any vehicle control.**

| Module | Name | Function |
|---|---|---|
| 5 | BEV Projector | Projects tracks to bird's-eye view using calibration |
| 6 | Sensor Fusion | Simulates radar fusion with camera detections |
| 7 | Prediction/Planning | Trajectory prediction, path planning advisory |
| 8 | Control Sim | Simulated throttle/brake/steer commands (display only) |
| 9 | Driver Monitoring | Drowsiness/attention detection via driver-facing camera |

---

### Module 10 — Forward Collision Warning
**File:** [`perception/`](perception/) (FCW module)

Staged human-facing alert system consuming the decision engine's hazard output.

**Alert levels:**
| Level | Name | Visual | Audio |
|---|---|---|---|
| 0 | Clear | None | None |
| 1 | Advisory | Yellow border | None |
| 2 | Warning | Orange flash | Beep |
| 3 | Critical | Red fullscreen | Alarm |

---

### Module 11 — Scene Understanding
**Files:** [`perception/traffic_light_state.py`](perception/traffic_light_state.py), [`perception/traffic_sign_recognition.py`](perception/traffic_sign_recognition.py)

- **Traffic light:** HSV-based colour detector on YOLO-detected traffic-light bounding boxes → `RED`/`AMBER`/`GREEN`
- **Traffic sign:** CNN classifier (GTSRB-trained `gtsrb_sign_cnn.pth`) → speed limit extraction → feeds advisory planner

---

### Module 12 — Drive Logger
**File:** [`core/drive_logger.py`](core/drive_logger.py)

Black-box `.jsonl` logger. Records per-frame: decision level, lateral action, detected tracks, FCW state, traffic light state, speed limit, planner output, simulated control commands.

Replay with: `python replay_log.py <logfile.jsonl>`

---

### Module 14 — Async Object Detection
**File:** [`perception/async_detector.py`](perception/async_detector.py)

Runs YOLO on a dedicated background thread (targeting the Intel iGPU via ONNX Runtime's DirectML / OpenVINO EP). The main thread continues lane detection and HUD rendering at full FPS while YOLO updates asynchronously. The tracker's motion model coasts object positions between detections.

Enable with: `python main.py --video ... --async`

---

## 4. Data Pipeline & Training

### Datasets

| Dataset | Use | Size | Notes |
|---|---|---|---|
| **IDD** (Indian Driving Dataset) | Drivable area segmentation | ~10k images | Indian roads, 34 classes, IIIT Hyderabad |
| **BDD100K** | Supplementary drivable area | 70k images | Berkeley, global roads, pre-made masks |
| **GTSRB** | Traffic sign classification | 50k images | German Traffic Sign Recognition Benchmark |

### Training Scripts

| Script | Purpose |
|---|---|
| [`training/train_local.py`](training/train_local.py) | CPU fine-tuning from pretrained weights (~3 hrs) |
| [`training/train_bdd.py`](training/train_bdd.py) | Full GPU training on BDD100K |
| [`training/train_signs.py`](training/train_signs.py) | Train GTSRB sign classifier |
| [`training/adverse_aug.py`](training/adverse_aug.py) | Adverse-condition augmentation library |
| [`training/prepare_bdd_drivable.py`](training/prepare_bdd_drivable.py) | BDD100K manifest builder |

### Adverse Augmentation (Phase 12)

The model is fine-tuned with synthetic adverse conditions to improve robustness on Indian roads:

| Augmentation | Simulates |
|---|---|
| `RandomBrightnessContrast(brightness_limit=(-0.5, 0.1))` | Night driving |
| `RandomFog(fog_coef_range=(0.2, 0.6))` | Fog / haze |
| `MotionBlur(blur_limit=(7, 21))` | Camera motion / vibration |
| `RandomShadow` | Tree shadows / overpasses |
| `GaussNoise` | Low-light sensor noise |

### Kaggle Training Notebook

The Kaggle notebook [`kaggle/notebooks/sih_adas_train.ipynb`](kaggle/notebooks/sih_adas_train.ipynb) runs the full training pipeline on free T4 GPU:
- Mounts `krishna22b/adas-weights` (source + pretrained weights)
- Mounts `krishna22b/idd-segmentation` (IDD full dataset)
- Fine-tunes LRASPP on IDD with adverse augmentation
- Exports `.pth` + `.onnx` to `/kaggle/working/models/`

---

## 5. Model Export & Inference Backends

### Export Pipeline

```
train_local.py / train_bdd.py
        │
        ▼
drivable_idd_full_best.pth   (PyTorch checkpoint)
        │
        ├──► export/export_onnx.py
        │           └──► drivable_idd_lraspp_768x432.onnx
        │
        └──► export/export_openvino.py
                    ├──► *_ov_fp16.xml / .bin
                    └──► *_ov_int8.xml / .bin   (via NNCF quantization)
```

### Backend Selection Logic (learned_road_detection.py)

```python
# Auto-selected at runtime, fastest → slowest:
if openvino_int8_available:   use OpenVINO INT8
elif openvino_fp16_available: use OpenVINO FP16
elif onnxruntime_available:   use ONNX Runtime
else:                         use PyTorch
```

### Benchmark (Intel i7-8650U + UHD 620)

| Backend | Latency (ms) | Speedup vs PyTorch |
|---|---|---|
| PyTorch (CPU) | ~185 ms | 1× |
| ONNX Runtime | ~95 ms | ~1.9× |
| OpenVINO FP16 | ~63 ms | ~2.9× |
| OpenVINO INT8 | ~32 ms | ~5.8× |

---

## 6. Configuration Reference

All tunable parameters live in [`core/config.py`](core/config.py).

### Camera / Video
| Parameter | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `TARGET_FPS` | `60` | Target frame rate |
| `FRAME_WIDTH/HEIGHT` | `1280×720` | Processing resolution |

### Lane Detection
| Parameter | Default | Description |
|---|---|---|
| `LANE_PROC_SCALE` | `0.5` | Downsample factor before Hough (1.0 = off) |
| `LANE_REQUIRE_MARKINGS` | `True` | Reject edges not on white/yellow paint |
| `ROI_TOP_Y` | `0.60` | Top of trapezoid ROI (fraction of height) |
| `MIN_SLOPE / MAX_SLOPE` | `0.3 / 2.5` | Lane line slope filter |

### Object Detection
| Parameter | Default | Description |
|---|---|---|
| `OBJ_CONF_THRESHOLD` | `0.4` | YOLO confidence threshold |
| `ENABLE_ASYNC_DETECTION` | `False` | Background-thread YOLO |

### Learned Road Model
| Parameter | Default | Description |
|---|---|---|
| `ENABLE_LEARNED_ROAD` | `True` | Enable Module 1c |
| `LEARNED_MODEL_PATH` | `drivable_idd_full_best.pth` | Weights path |
| `LEARNED_BACKEND` | `auto` | `auto`/`onnx`/`openvino`/`torch` |

### Feature Flags
| Flag | Module |
|---|---|
| `ENABLE_ROAD_FALLBACK` | Module 1b |
| `ENABLE_FCW` | Module 10 |
| `ENABLE_TRAFFIC_LIGHT` | Module 11 |
| `ENABLE_SIGN_RECOGNITION` | Module 11 |
| `ENABLE_BEV` | Module 5 |
| `ENABLE_DRIVER_MON` | Module 9 |

---

## 7. Running the System

### Prerequisites

```bash
pip install -r requirements.txt
```

Minimal (CPU-only, no ONNX/OpenVINO):
```bash
pip install opencv-python numpy ultralytics torchvision
```

### Basic Usage

```bash
# Webcam (both modules)
python core/main.py --camera

# Video file
python core/main.py --video dashcam.mp4

# Lanes only (no YOLO)
python core/main.py --video dashcam.mp4 --no-objects

# Debug overlay + save output
python core/main.py --video dashcam.mp4 --debug --save output.mp4

# With drive logging
python core/main.py --video dashcam.mp4 --log drive_log.jsonl

# With driver monitoring (second camera)
python core/main.py --camera --driver-cam 1

# Async YOLO (background thread, helps on Intel iGPU)
python core/main.py --video dashcam.mp4 --async
```

### Keyboard Controls (while running)
| Key | Action |
|---|---|
| `Q` | Quit |
| `D` | Toggle debug overlay (ROI + raw Hough lines) |
| `P` | Pause / resume |
| `S` | Save screenshot |

### Web Demo

```bash
python web_demo.py
# Opens Gradio UI at http://localhost:7860
```

---

## 8. Performance Benchmarks

Measured on **Intel Core i7-8650U + UHD 620** (laptop, no discrete GPU):

| Configuration | FPS |
|---|---|
| Lanes only (classical) | ~45 FPS |
| Lanes + YOLO (sync) | ~14 FPS |
| Lanes + YOLO (async, Module 14) | ~22 FPS |
| Lanes + learned road (ONNX) | ~18 FPS |
| Lanes + learned road (OpenVINO INT8) | ~28 FPS |
| Full pipeline (all modules) | ~12 FPS |

---

## 9. Project File Structure

```
adas-vision/
├── core/
│   ├── main.py                  # Entry point, main loop
│   ├── config.py                # All tunable parameters
│   ├── decision_engine.py       # Module 3: arbitration brain
│   ├── drive_logger.py          # Module 12: black-box logger
│   ├── frame_guard.py           # Malformed frame detector
│   └── replay_log.py            # Replay drive logs
│
├── perception/
│   ├── lane_detection.py        # Module 1: paint-based lanes
│   ├── road_detection.py        # Module 1b: classical road fallback
│   ├── learned_road_detection.py# Module 1c: LRASPP segmentation
│   ├── object_detection.py      # Module 2: YOLOv8n detection
│   ├── tracker.py               # Module 4: multi-object tracker
│   ├── async_detector.py        # Module 14: background YOLO thread
│   ├── traffic_light_state.py   # Module 11a: light colour reader
│   ├── traffic_sign_recognition.py # Module 11b: sign CNN
│   └── driver_monitoring.py     # Module 9: drowsiness detection
│
├── training/
│   ├── train_local.py           # CPU fine-tuning script
│   ├── train_bdd.py             # BDD100K GPU training
│   ├── train_signs.py           # GTSRB sign classifier training
│   ├── adverse_aug.py           # Augmentation library
│   └── prepare_bdd_drivable.py  # BDD manifest builder
│
├── kaggle/
│   └── notebooks/
│       └── sih_adas_train.ipynb # Kaggle training notebook
│
├── export/                      # ONNX + OpenVINO export scripts
├── evaluation/                  # Benchmark scripts
├── simulation/                  # Advisory simulation modules
│
├── drivable_idd_full_best.pth          # Trained PyTorch weights
├── drivable_idd_lraspp_768x432.onnx   # ONNX export
├── drivable_idd_lraspp_768x432_ov_fp16.* # OpenVINO FP16
├── drivable_idd_lraspp_768x432_ov_int8.* # OpenVINO INT8
├── yolov8n.pt / yolov8n.onnx          # YOLO model
│
├── requirements.txt
├── README.md
└── DOCUMENTATION.md             # ← this file
```

---

## 10. SIH Context & Contributions

This project was developed for **Smart India Hackathon (SIH)** as a research prototype demonstrating a complete ADAS pipeline suitable for Indian road conditions.

### Key Technical Contributions

| Contribution | Detail |
|---|---|
| **Indian-road segmentation** | Fine-tuned LRASPP on IDD with adverse augmentation; val mIoU ~0.92 |
| **3-tier perception fallback** | Paint lanes → learned road → classical road; always has a best-effort answer |
| **Temporal ratchet safety** | Decision smoothing prevents single noisy frames from triggering false alerts |
| **OpenVINO INT8 inference** | ~5.8× speedup over PyTorch on laptop iGPU — enables real-time on embedded hardware |
| **Adverse augmentation** | Night / fog / blur / shadow / noise augmentation for robustness on real Indian roads |
| **Complete pipeline** | 14 modules from raw pixel → driving decision → collision warning |

### Team

- **Repository:** [github.com/krishna2077-s/adas-vision](https://github.com/krishna2077-s/adas-vision)
- **Kaggle training:** [kaggle.com/code/krishna22b/adas-vision](https://kaggle.com/code/krishna22b/adas-vision)

---

*ADAS Vision — Advisory / research prototype. Never wire to a vehicle's controls.*
