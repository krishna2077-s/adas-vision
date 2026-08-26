# ADAS Vision: Developer Handoff Guide (RTX 4050 Edition)

Welcome to the **ADAS Vision** project! This guide is specifically tailored for developers running an NVIDIA RTX 4050 (or similar CUDA-capable GPU). 

The goal of this system is to provide a real-time, lightweight Advanced Driver Assistance System (ADAS) that processes dashcam/webcam footage to provide driver warnings. 

> **Important Architecture Note:** This system uses Neural Networks for **perception only**. All actual driving decisions (when to brake, when to warn) are handled by a strictly **deterministic, rule-based engine**. It is designed to be explainable and fail safely.

---

## 1. Quickstart: Unleashing the RTX 4050 (60-100+ FPS)

The codebase is already configured to automatically detect and use your RTX 4050. `PREFER_CUDA = True` is set in `config.py`. 

### Setup Environment
First, you need a Python environment with CUDA-enabled PyTorch so Python can talk to your GPU:

```bash
# 1. Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest of the project dependencies
pip install -r requirements.txt
pip install ultralytics  # For YOLOv8
```

### Verify & Run
Check that CUDA is active, then run the pipeline in asynchronous mode:

```bash
# Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Run the main pipeline on the dashcam video
python main.py --video dashcam.mp4 --async
```
*Note: Using `--async` puts YOLO object detection on a background worker thread. Combined with your RTX 4050, you should easily hit a smooth 60+ FPS.*

---

## 2. How the Engine Works: The Core Pipeline

The system processes frames in a strictly ordered pipeline, dropping malformed frames early to prevent crashes.

1. **Frame Health Guard:** `frame_guard.py` ensures the camera buffer is valid.
2. **Lane & Road Perception:** Finds drivable space.
   * *Tier 1:* Painted lines (`lane_detection.py` - Canny/Hough).
   * *Tier 2:* Learned Segmentation CNN (`learned_road_detection.py` - LRASPP) -> Finds road when lines are missing.
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
* `lane_detection.py`: Classical CV (Gaussian Blur -> Canny -> Hough Transform) for painted lane lines.
* `learned_road_detection.py`: Neural Net (LRASPP MobileNetV3) for segmenting drivable surfaces (dirt roads, unmarked roads).
* `road_detection.py`: Classical fallback for unmarked roads using color (CIELAB) and texture flood-fill.
* `object_detection.py`: Synchronous YOLOv8 wrapper. Filters COCO classes to only road-relevant ones (cars, bikes, cows, etc.). Estimates distance based on pixel height.
* `async_detector.py`: Threaded YOLOv8 wrapper. Lets YOLO run at max GPU speed in the background without blocking the main rendering loop.
* `tracker.py`: SORT-style multi-object tracker. Extrapolates bounding boxes using constant velocity to maintain IDs even if YOLO drops a frame.
* `traffic_light_state.py`: Analyzes bounding boxes of traffic lights in HSV space to determine Red/Amber/Green.
* `traffic_sign_recognition.py`: 43-class CNN trained on GTSRB to read speed limits and stop signs.
* `driver_monitoring.py`: Uses Haar Cascades to check if the driver's face/eyes are looking forward or distracted.

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

Your RTX 4050 is perfect for retraining these models locally. 

### A. Road Segmentation Model (LRASPP)
The road segmentation model identifies drivable areas. It was originally trained on IDD (Indian Driving Dataset) and BDD100K.

1. **Augmentations:** `adverse_aug.py` applies heavy photometric augmentations on-the-fly (fog, night-darkening, glare, rain, sensor noise) so the model learns to drive at night and in bad weather.
2. **Prepare Data:** Use `prepare_bdd_drivable.py` to index the BDD100K dataset, separate out night/weather footage, and create JSON manifests.
3. **Train:** 
   * `train_local.py`: Used for base training. Uses Cross-Entropy + Soft Dice loss.
   * `train_bdd.py`: Used for fine-tuning on adverse weather. Implements a `WeightedRandomSampler` to oversample night and rain images.
   ```bash
   # Example: run training
   python train_bdd.py --epochs 50 --batch-size 16
   ```

### B. Traffic Sign Classifier
This is a lightweight custom CNN trained on the GTSRB dataset (German Traffic Sign Recognition Benchmark, 43 classes).
1. **Train:** Run `train_signs.py`. Because the model is so small (~0.5M params), this will complete in minutes on your 4050.
   ```bash
   python train_signs.py
   ```
   *Output:* `gtsrb_sign_cnn.pth`.

### C. Exporting Models (For production / OpenVINO / TensorRT)
To strip PyTorch overhead and run models bare-metal:

1. **ONNX:** `export_onnx.py` converts the PyTorch road model (`.pth`) to an ONNX graph (`.onnx`), validating numerical parity.
2. **OpenVINO (Intel):** `export_openvino.py` converts ONNX to FP16/INT8 formats optimized for Intel iGPUs. (You won't need this on the 4050, but it's used for laptop deployments).
3. **TensorRT (NVIDIA/RTX):** To maximize your 4050's performance, export YOLO to a TensorRT engine:
   ```bash
   yolo export model=yolov8n.pt format=engine half=True imgsz=640
   ```
   Then set `YOLO_TRT_MODEL = "yolov8n.engine"` in `config.py`.

---

## 5. Next Steps / Modifications

* **To adjust detection sensitivity:** Tweak `YOLO_CONF_THRESHOLD` in `config.py`.
* **To adjust braking distances:** Edit the `d <= 5.0` and `TTC <= 2.5` thresholds in `decision_engine.py` under the `_rule_1` and `_rule_2` functions.
* **To add a new object class:** Add it to the YOLO model, add its standard height to `CLASS_REAL_HEIGHTS` in `config.py`, and update the `RELEVANT_CLASSES` set.
