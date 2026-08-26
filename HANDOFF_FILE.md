# ADAS Vision: Developer Handoff Guide

> **Last Updated:** 2026-08-26  
> **Updated By:** Akshit (akshit-mat / mathurakshit.30@gmail.com)  
> **Repo:** https://github.com/krishna2077-s/adas-vision  
> **Target Hardware:** Any consumer laptop/PC (Optimized for CPU, OpenVINO, or NVIDIA CUDA GPUs)

---

## ⚡ Current Project Status

> Read this section first if you are picking up this project mid-way.

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Environment setup (venv, CUDA PyTorch, dependencies) | ✅ COMPLETE |
| **Phase 2** | Traffic sign model training (`train_signs.py` → `gtsrb_sign_cnn.pth`) | ✅ COMPLETE — **99.9% validation accuracy** |
| **Phase 3** | Lane detection bug fix (line 252 `TypeError` in `lane_detection.py`) | ✅ COMPLETE |
| **Phase 4** | Full pipeline test on `dashcam.mp4` (main.py --async) | ✅ COMPLETE — **~26+ FPS on mid-range GPU**, 4,246 frames processed |
| **Phase 5** | UDP bridge built (`udp_bridge.py`, `test_udp_bridge.py`) | ✅ COMPLETE — Fixed `track_id` → `id` and `distance_m` → `smoothed_distance_m` attribute bugs |
| **Phase 6** | MATLAB co-simulation scripts generated | ✅ FILES READY — **Awaiting MATLAB installation to execute** |
| **Phase 7** | Driving scenarios generated (village road, urban intersection, cattle crossing) | ✅ FILES READY — **Awaiting MATLAB installation to execute** |
| **Phase 8** | Metrics collector (`collect_metrics.m`) | ✅ FILES READY — **Awaiting MATLAB to run** |
| **Phase 9** | TensorRT export for YOLO (max FPS on NVIDIA GPUs) | 🔄 IN PROGRESS — `tensorrt` pip package installation was started, not yet confirmed complete |

---

## 🚧 What Needs to Be Done Next

1. **Verify TensorRT installation** — Run the following to check if the install we started completed:
   ```powershell
   cd a:\PROJECTS\adas-vision\adas-vision
   .\venv\Scripts\python.exe -c "import tensorrt; print(tensorrt.__version__)"
   ```
   If it throws an error, install it:
   ```powershell
   .\venv\Scripts\python.exe -m pip install tensorrt
   ```

2. **Export YOLO to TensorRT** — Once `tensorrt` is confirmed installed:
   ```powershell
   .\venv\Scripts\yolo.exe export model=yolov8n.pt format=engine quantize=half imgsz=640
   ```
   Then open `config.py` and set `YOLO_TRT_MODEL = "yolov8n.engine"` to activate it.

3. **Run the MATLAB Co-simulation** — MATLAB is **NOT installed** on this laptop (checked the registry — no MathWorks entry found). Options:
   - Install MATLAB (requires MathWorks license / university account). Custom install path works (e.g. `A:\Programs\MATLAB`).
   - Use **MATLAB Online** at [matlab.mathworks.com](https://matlab.mathworks.com) — upload the 6 `.m` files from `matlab/` and run the standalone scenarios (they work without Python). **The live UDP co-sim (`run_cosimulation.m`) will NOT work online** because UDP is blocked to remote hosts.
   - Ask teammate **krishna2077-s** to run the MATLAB side (they have the original setup).

4. **Commit & Push to GitHub** — All local changes from this session are staged and ready. See Section 6 below.

---

## 🛠️ Environment Setup (From Scratch)

The virtual environment is already built at `a:\PROJECTS\adas-vision\adas-vision\venv\`.  
If you need to rebuild it on a new machine:

```powershell
# 1. Create venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install CUDA-enabled PyTorch (if using an NVIDIA GPU — e.g. CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install all other dependencies
pip install -r requirements.txt
pip install ultralytics   # YOLOv8
```

### Verify GPU is active
```powershell
.\venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```
Expected output: `CUDA: True | <Your NVIDIA GPU Name>`

---

## ▶️ How to Run the Application

```powershell
cd a:\PROJECTS\adas-vision\adas-vision

# Standard run (recommended — async YOLO on GPU thread)
.\venv\Scripts\python.exe main.py --video dashcam.mp4 --async

# Press Q inside the video window to quit cleanly
```

**Observed performance:** ~26+ FPS average over 4,246 frames on a mid-range dedicated GPU (e.g. RTX 4050).

---

## 📡 UDP Bridge (Python ↔ MATLAB)

The bridge streams live perception data (lanes, tracks, decisions, signs) over UDP to MATLAB and receives ego-vehicle position back.

```powershell
# Start the bridge (replaces main.py — includes full pipeline + UDP output)
.\venv\Scripts\python.exe udp_bridge.py --video dashcam.mp4 --async

# In a separate terminal — verify the bridge is transmitting:
.\venv\Scripts\python.exe test_udp_bridge.py
```

**Ports:**
- Python → MATLAB: `127.0.0.1:5005` (JSON perception packets)  
- MATLAB → Python: `0.0.0.0:5006` (ego position feedback)

**Known bugs fixed in this session:**
- `AttributeError: 'Track' object has no attribute 'track_id'` → Fixed: use `t.id`
- `AttributeError: 'Track' object has no attribute 'distance_m'` → Fixed: use `t.smoothed_distance_m`

---

## 🗂️ MATLAB Co-Simulation Files

All files are in `matlab/`. They were generated during this session since they were missing from the original zip.

| File | Purpose | Can run standalone? |
|---|---|---|
| `setup_vehicle_model.m` | Loads vehicle physics, PID/Pure-Pursuit gains, UDP ports into workspace | Yes (run first) |
| `run_cosimulation.m` | Live co-sim: connects to Python UDP, animates vehicle | **No** — needs Python bridge running on same machine |
| `scenario_village_road.m` | Animated scenario: narrow road, pedestrian, motorcycle | **Yes** |
| `scenario_urban_intersection.m` | Animated scenario: 4-way junction, cross-traffic, cyclists | **Yes** |
| `scenario_cattle_crossing.m` | Animated scenario: cattle enter road at t=3s → emergency brake | **Yes** |
| `collect_metrics.m` | Computes performance metrics → CSV + bar chart | **Yes** (needs `scenario_result` in workspace) |

### To run MATLAB standalone scenarios (no Python needed):
```matlab
% In MATLAB Command Window, with CWD set to matlab/ folder:
setup_vehicle_model
scenario_village_road       % Watch animated sim
collect_metrics(scenario_result, 'village_road')   % Get metrics CSV
```

---

## 🧠 AI Models — Current State

| Model | File | Size | Status | Notes |
|---|---|---|---|---|
| Traffic Sign CNN | `gtsrb_sign_cnn.pth` | 3.2 MB | ✅ **Trained this session** | 99.9% val accuracy, GTSRB 43-class |
| Road Segmentation | `drivable_idd_lraspp_adv_best.pth` | 12.5 MB | ✅ Pre-trained | IDD + BDD100K adverse fine-tune, IoU 0.92 |
| YOLOv8n | `yolov8n.pt` | 6.2 MB | ✅ Ready | Downloaded automatically |
| YOLO ONNX | `yolov8n.onnx` | 12.3 MB | ✅ Ready | For non-CUDA deployment |
| YOLO TensorRT | `yolov8n.engine` | — | ⏳ Not yet generated | Requires Phase 9 completion |
| Road ONNX | `drivable_idd_lraspp_768x432.onnx` | 12.3 MB | ✅ Ready | For CPU/ONNX Runtime |
| Road OpenVINO FP16 | `*_ov_fp16.xml/.bin` | 6.4 MB | ✅ Ready | For Intel iGPU |
| Road OpenVINO INT8 | `*_ov_int8.xml/.bin` | 3.6 MB | ✅ Ready | For VNNI CPUs |

> ⚠️ **Model weights are excluded from git** (`.gitignore` has `*.pth`, `*.onnx`, etc.).  
> Share weights via GitHub Releases or a shared drive link separately.

---

## 🔄 Git Status — Changes from This Session

```
Modified:   .gitignore          (added *.zip, GTSRB_extracted/, IDE folders)
Modified:   lane_detection.py   (bug fix: TypeError on line 252)
Modified:   train_signs.py      (sign training updates)
New file:   udp_bridge.py       (written from scratch this session)
New file:   test_udp_bridge.py  (written from scratch this session)
New file:   matlab/             (6 scripts — all generated this session)
New file:   web_demo.py         (Gradio web demo)
```

To commit and push:
```powershell
cd a:\PROJECTS\adas-vision\adas-vision
git add .gitignore lane_detection.py train_signs.py udp_bridge.py test_udp_bridge.py matlab/ web_demo.py
git commit -m "Session: sign model training, UDP bridge, MATLAB co-sim, bug fixes, .gitignore update"
git push origin main
```

---

## 2. How the Engine Works: The Core Pipeline

The system processes frames in a strictly ordered pipeline, dropping malformed frames early to prevent crashes.

1. **Frame Health Guard:** `frame_guard.py` ensures the camera buffer is valid.
2. **Lane & Road Perception:** Finds drivable space.
   * *Tier 1:* Painted lines (`lane_detection.py` - Canny/Hough).
   * *Tier 2:* Learned Segmentation CNN (`learned_road_detection.py` - LRASPP) → Finds road when lines are missing.
   * *Tier 3:* Classical CV fallback (`road_detection.py`).
3. **Object Perception:** `object_detection.py` (or `async_detector.py`) uses YOLOv8n to find cars, pedestrians, traffic lights. It calculates distance using monocular pinhole camera geometry.
4. **Tracking:** `tracker.py` associates YOLO bounding boxes across frames using IoU (Intersection over Union) and estimates velocity/Time-to-Collision (TTC).
5. **Decision Engine:** `decision_engine.py` ingests lanes + tracked objects and applies rules R1-R7 (see below).
6. **Alerts & UI:** `forward_collision_warning.py` flashes the screen if TTC is critical. The HUD is drawn on the frame and displayed.

### The Decision Rules (R1 - R7)
In `decision_engine.py`, the system evaluates the closest in-path object against a priority table:
* **R1 (EMERGENCY_STOP):** Object is < 5m away OR TTC < 1.2s.
* **R2 (BRAKE):** TTC is < 2.5s (extra margin given for pedestrians/cyclists).
* **R3 (BRAKE):** Static high-risk obstacle within 8m.
* **R4 (SLOW):** Stop sign or red light within 25m.
* **R5 (SLOW):** Medium-risk obstacle within 20m.
* **R6 (CAUTION):** Gentle closing speed (TTC < 4.0s).
* **R7 (PROCEED):** Path is clear.

---

## 3. Codebase Directory: What Each File Does

### Core Pipeline
* `main.py`: The entry point. Initializes modules, reads frames in a loop, calls processors, and handles UI rendering.
* `config.py`: The central nervous system. Contains all thresholds, model paths, hardware toggles (`PREFER_CUDA`), and UI colors.
* `decision_engine.py`: The deterministic brain. Applies R1-R7 policies, maintains a "temporal ratchet" (escalates alerts quickly, de-escalates slowly), and clamps steering based on hazard proximity.
* `frame_guard.py`: Failsafe to drop corrupted/empty frames before they break inference.

### Perception Modules
* `lane_detection.py`: Classical CV (Gaussian Blur → Canny → Hough Transform) for painted lane lines. **Bug fixed on line 252 this session.**
* `learned_road_detection.py`: Neural Net (LRASPP MobileNetV3) for segmenting drivable surfaces (dirt roads, unmarked roads).
* `road_detection.py`: Classical fallback for unmarked roads using color (CIELAB) and texture flood-fill.
* `object_detection.py`: Synchronous YOLOv8 wrapper. Filters COCO classes to only road-relevant ones (cars, bikes, cows, etc.). Estimates distance based on pixel height.
* `async_detector.py`: Threaded YOLOv8 wrapper. Lets YOLO run at max GPU speed in the background without blocking the main rendering loop.
* `tracker.py`: SORT-style multi-object tracker. Extrapolates bounding boxes using constant velocity to maintain IDs even if YOLO drops a frame.
* `traffic_light_state.py`: Analyzes bounding boxes of traffic lights in HSV space to determine Red/Amber/Green.
* `traffic_sign_recognition.py`: 43-class CNN trained on GTSRB to read speed limits and stop signs.
* `driver_monitoring.py`: Uses Haar Cascades to check if the driver's face/eyes are looking forward or distracted.

### Bridge & Co-Simulation (New This Session)
* `udp_bridge.py`: Streams full perception telemetry (JSON over UDP port 5005) to MATLAB. Receives ego-position feedback on port 5006.
* `test_udp_bridge.py`: Debug tool — listens on port 5005 and prints live telemetry frames to verify the bridge is working.
* `matlab/setup_vehicle_model.m`: Loads vehicle physics and controller parameters into the MATLAB workspace.
* `matlab/run_cosimulation.m`: Master co-simulation loop — connects to Python UDP, runs path planning and vehicle control, displays live animation.
* `matlab/scenario_village_road.m`: Animated village road scenario (standalone).
* `matlab/scenario_urban_intersection.m`: Animated urban intersection scenario (standalone).
* `matlab/scenario_cattle_crossing.m`: Emergency braking scenario — two cows enter road at t=3s (standalone).
* `matlab/collect_metrics.m`: Computes and exports full performance metrics to CSV + bar chart.

### Simulation & Advisory Overlays
*(These files simulate an autonomous vehicle's internal state for debugging/display, but never control real hardware)*
* `perception_bev.py`: Translates 2D camera pixels into a top-down Bird's Eye View (BEV) map.
* `sensor_fusion.py`: Simulates fusing camera depth with a fake radar signal using Bayesian variance.
* `prediction_planning.py`: Projects where tracked objects will be in 3 seconds.
* `control_sim.py`: Simulates lateral (steering) and longitudinal (throttle/brake PID) controls.

### Logging & Evaluation
* `test_adas.py`: 31 PyTest unit tests validating the decision engine logic.
* `safety_audit.py`: Runs a full clip head-less to ensure the system *never* outputs "PROCEED" when an obstacle is dangerously close.
* `evaluate.py` & `bench_speed.py`: Tools for measuring millisecond latency of individual modules.
* `drive_logger.py` & `replay_log.py`: Logs decisions to a `.jsonl` file and replays them offline.
* `web_demo.py`: A Gradio web app for scrubbing video and analyzing perception masks.

---

## 4. Training Roadmap (How to Train/Retrain Models)

### A. Road Segmentation Model (LRASPP) — Pre-trained, no action needed
The road segmentation model identifies drivable areas. It was trained on IDD (~20,000 images) and fine-tuned on BDD100K adverse weather subset. The weights are at `drivable_idd_lraspp_adv_best.pth`.

If you need to retrain:
```bash
python train_bdd.py --epochs 50 --batch-size 16
```

### B. Traffic Sign Classifier — ✅ Already trained this session
The model has been trained and saved as `gtsrb_sign_cnn.pth` (99.9% accuracy).  
To retrain from scratch:
```bash
python train_signs.py
```
Training takes ~30 minutes on a mid-range GPU. Requires GTSRB dataset — extract `GTSRB_Final_Training_Images.zip` into `GTSRB_extracted/` first.

### C. Exporting Models (TensorRT for max FPS on NVIDIA GPUs)
```bash
# Export YOLO to TensorRT (hardware-optimised for NVIDIA cards)
# NOTE: tensorrt pip package must be installed first
yolo export model=yolov8n.pt format=engine quantize=half imgsz=640
```
Then in `config.py`, set: `YOLO_TRT_MODEL = "yolov8n.engine"`  
**Status: In progress — `tensorrt` package was being installed when last checked.**

---

## 5. Next Steps / Future Work

* **Complete TensorRT export** — See Section "What Needs to Be Done Next" above.
* **Run MATLAB scenarios** — Install MATLAB or use MATLAB Online to execute the 6 scripts in `matlab/`.
* **Commit & push to GitHub** — See git commands in Section "Git Status" above.
* **To adjust detection sensitivity:** Tweak `YOLO_CONF_THRESHOLD` in `config.py`.
* **To adjust braking distances:** Edit the `d <= 5.0` and `TTC <= 2.5` thresholds in `decision_engine.py` under the `_rule_1` and `_rule_2` functions.
* **To add a new object class:** Add it to the YOLO model, add its standard height to `CLASS_REAL_HEIGHTS` in `config.py`, and update the `RELEVANT_CLASSES` set.
