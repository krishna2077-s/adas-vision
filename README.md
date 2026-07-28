# ADAS Vision

An open, low-cost Advanced Driver Assistance System built to run on hardware people already own — a dashcam or webcam and a standard laptop CPU. No GPU, no cloud, no dedicated hardware.

**This release: painted-lane detection, learned drivable-area segmentation for unmarked roads (a compact CNN trained on the full Indian Driving Dataset), object detection with collision warnings, multi-object tracking with stable IDs, and a decision engine that fuses it all into a single arbitrated driving action every frame.**

> ## ⚠️ Safety notice — read this first
>
> **This is an advisory / research and learning project. It must NEVER be
> connected to a vehicle's steering, throttle, or brakes, and must NEVER be
> relied on while actually driving.** No configuration, tuning, or amount of
> extra training changes this. It is not a life-safety system, and here is
> honestly why it cannot be one:
>
> - **One camera, no redundancy.** Real ADAS fuse camera **+ radar** (often
>   + lidar). A single camera *cannot* reliably measure distance — this system
>   estimates it from bounding-box size, which is approximate by design. Glare,
>   rain, night, or a dirty lens can blind it with nothing to cross-check.
> - **No guaranteed timing.** Life-safety systems run at 30–60+ fps with hard
>   real-time deadlines on automotive-grade, redundant hardware. A shared laptop
>   CPU running best-effort has no such guarantee — the one moment it stalls
>   could be the moment that matters.
> - **Not certified or validated.** Production ADAS are built to functional-
>   safety standards (ISO 26262 / ASIL) and validated over millions of miles.
>   This is not.
> - **A CPU vision model is sometimes confidently wrong** — and in a real car,
>   being confidently wrong about a pedestrian is exactly the failure that costs
>   a life.
>
> **The safe design is precisely that it stays out of the control loop.** Use it
> as a screen overlay, a dashcam-analysis / driver-awareness tool you *review*,
> and a way to learn computer vision — never as something that drives or that you
> trust to keep you safe on the road.

## What it does

**Lane detection (Module 1) — painted markings**
- Detects lane paint (white/yellow-gated edges) in real time — kerbs, walls and embankments can't masquerade as lanes
- Geometry sanity check rejects implausible pairs (crossed "tent" lines, absurd widths) instead of trusting them
- Calculates drift from lane centre and outputs a steering suggestion (`STRAIGHT`, `SLIGHT LEFT`, ...)
- **Fails honestly**: no usable paint → confidence 0 → hands over to Module 1b

**Learned road segmentation (Module 1c) — unmarked roads, primary**
- A compact CNN (**LRASPP MobileNetV3-Large**) trained on the **full Indian Driving Dataset** (~20k images; validation drivable-IoU **0.92**) segments the drivable road surface directly
- Handles the scenes classical CV can't — curving, hilly, unmarked forest roads — building the same **curved centreline** for steering
- Runs on **CPU** (~4 fps at 768×432 on a laptop, the same order as YOLOv8n) — no GPU needed at run time; the GPU is only used once, for training
- Guidance is capped (drivable-area, not paint-precise) and tagged `LEARNED ROAD (IDD)`
- If the trained weights are absent (or PyTorch isn't installed) it steps aside to Module 1b, so the system still runs on a clean checkout

**Road-surface following (Module 1b) — classical fallback**
- The original hand-tuned detector: **drivable road surface** by texture (smooth) + colour (grey) + connectivity to the patch ahead of the car
- Kept as the fallback when the learned model is unavailable or finds no road
- Builds a **curved centreline** per frame; guidance capped at confidence 0.5, labelled `ROAD-FOLLOW (estimated)`
- Reports *no guidance* (degraded) when not enough road is visible, rather than guessing

**Object detection (Module 2)**
- Detects road-relevant objects with YOLOv8n — cars, trucks, buses, pedestrians, two-wheelers, animals
- Estimates each object's distance from a monocular bounding-box heuristic
- Determines which objects are in the vehicle's forward path (perspective-aware corridor)
- Assigns LOW / MEDIUM / HIGH risk and raises a Forward Collision Warning banner

**Object tracking (Module 4)**
- Gives each detected object a **stable ID** across frames (greedy-IoU association, no extra dependencies)
- Maintains *per-object* smoothed distance, closing speed, and time-to-collision — so a hazard's kinematics stay stable even in dense traffic where the closest object keeps changing
- Confirms a track only after it's seen several frames (one-frame false detections never influence the car) and coasts it through brief dropouts

**Decision engine (Module 3) — the brain**
- Fuses the lane result + confirmed tracks (never pixels) into one arbitrated action per frame
- **Longitudinal**: `PROCEED → CAUTION → SLOW → BRAKE → EMERGENCY_STOP`, driven by the nearest in-path hazard, its estimated distance, and a smoothed closing speed / time-to-collision
- **Lateral**: `KEEP_LANE / CORRECT_LEFT / CORRECT_RIGHT / HOLD`, taken from the lane offset but *safety-clamped* — it never steers toward a hazard and never overrides braking
- **Temporal debouncing** so a single noisy frame (one spurious box, one distance spike, one dropped lane) can't flip the action: escalation is fast, release is slow, and an emergency stop latches
- **Degraded mode**: when lanes are lost it says so and behaves conservatively instead of pretending certainty
- Emits one plain-English **reason** per frame, e.g. `[R2] BRAKE: car closing, 7.1m TTC 2.1s`

All three modules run together at interactive frame rates on a standard laptop CPU — no GPU.

## How it works

**Module 1** — classical computer vision, no neural network:

```
Frame → Grayscale + Gaussian blur
      → Canny edge detection
      → Trapezoidal region-of-interest mask   (ignore sky / bonnet / roadside)
      → Probabilistic Hough transform          (find line segments)
      → Slope filtering + left/right split     (reject non-lane lines)
      → Per-side averaging + extrapolation     (one clean line per lane)
      → Exponential moving average             (stable lanes across frames)
      → Lane-centre offset → steering decision
```

**Module 2** — YOLOv8n inference plus geometry:

```
Frame → YOLOv8n detection (COCO classes, filtered to road-relevant)
      → Monocular distance estimate (pinhole model, per-class heights)
      → In-path test against Module 1's lane centre
      → Distance + path → risk level → collision warning
```

Module 2 uses the lane centre from Module 1 to decide what counts as "in front of us," so the two modules genuinely cooperate rather than just sharing a window.

**Module 4** — multi-object tracking (a stripped-down SORT, no dependencies):

```
Detections → greedy IoU association to existing tracks (nearest overlap wins)
           → per-track EMA distance + closing speed + TTC (spike-rejected)
           → lifecycle: confirm after N hits, coast up to MAX_AGE frames
           → stable Track objects (each keeps its own ID + kinematics)
```

**Module 3** — sensor fusion + arbitration, an explainable rule engine (no neural network):

```
Lane result + confirmed tracks
      → pick nearest actionable track  (non-advisory, in-path; its kinematics
                                        already smoothed by Module 4)
      → policy table R1..R7  (ordered, first-match-wins → raw action 0-4;
                             collision rules on top so safety dominates)
      → temporal ratchet     (fast N-of-M escalation, slow release, emergency
                             latch → committed action; one bad frame can't flip it)
      → lateral arbitration  (steer from lanes, but only ever reduce authority)
      → one DrivingDecision + a plain-English reason
```

Because tracking owns the per-object smoothing, the engine is just a handful of scalar operations over the tracks, so it adds negligible CPU on top of lane + YOLO.

> **Note on distance:** a single camera cannot measure true distance. Estimates come from apparent object size and are meant for relative "is this getting closer" logic, not survey-grade measurement. The decision engine smooths these estimates and reasons about *trends* (closing vs. receding) rather than trusting any single reading.

## Project layout

```
adas-vision/
├── config.py            ← All tunable parameters (all modules)
├── lane_detection.py    ← Module 1: LaneDetector          (painted markings)
├── learned_road_detection.py ← Module 1c: LearnedRoadDetector (CNN drivable-area, primary on unmarked roads)
├── road_detection.py    ← Module 1b: RoadDetector         (classical surface following, fallback)
├── object_detection.py  ← Module 2: ObjectDetector        (YOLOv8n)
├── tracker.py           ← Module 4: MultiObjectTracker    (stable IDs + kinematics)
├── decision_engine.py   ← Module 3: DecisionEngine        (fusion + arbitration)
├── perception_bev.py    ← Module 5: BEVProjector          (bird's-eye ego frame — sim)
├── sensor_fusion.py     ← Module 6: SimRadarFusion        (simulated radar + fusion)
├── prediction_planning.py ← Module 7: Planner             (prediction + advisory plan)
├── control_sim.py       ← Module 8: SimController         (simulated steering/PID — never wired)
├── driver_monitoring.py ← Module 9: DriverMonitor         (attention: driver-facing cam)
├── main.py              ← CLI entry point: runs the whole pipeline together
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

This pulls in OpenCV, NumPy, Ultralytics (which brings a CPU build of PyTorch for Module 2), and torchvision (for Module 1c). On first run, YOLOv8n weights (~6 MB) download automatically. Lane detection alone needs only OpenCV + NumPy — run with `--no-objects` if you haven't installed Ultralytics yet.

**Learned road model (Module 1c):** place the trained weights `drivable_idd_full_best.pth` in the repo root. They are produced by [`colab/phase6c_full_idd.ipynb`](colab/phase6c_full_idd.ipynb) (free Colab GPU) and shared via a GitHub Release rather than committed (the file is large). Without it, the system automatically falls back to the classical road detector (Module 1b) — nothing breaks, unmarked-road guidance is just less accurate.

## Real-time performance

The learned road model (Module 1c) has two stackable, retraining-free speed levers, both on by default in [config.py](config.py):

- **ONNX Runtime backend** (`LEARNED_BACKEND = "onnx"`) — runs the *same* trained weights ~2.8× faster than PyTorch on CPU, with byte-identical segmentation. Export once after training:
  ```bash
  python export_onnx.py
  ```
- **Frame-skip** (`LEARNED_INFER_EVERY = 3`) — runs the CNN every Nth frame and reuses the mask between (the road barely moves frame-to-frame); the centreline still updates every frame. ~3× effective throughput.

Together they take the model from ~2 fps to **~17 effective fps** on a laptop CPU — real-time-smooth. Drop `LEARNED_INPUT_W/H` to 512×288 for roughly 2× more. Benchmark on your own machine:

```bash
python bench_speed.py
```

If the `.onnx` file or `onnxruntime` is absent, the system falls back to the PyTorch backend automatically — nothing breaks.

**Object detection (Module 2) is deliberately *not* sped up the same way.** Two honest reasons:

- **ONNX doesn't help YOLOv8n on CPU.** It's a tiny, overhead-bound model (preprocessing + NMS dominate), so ONNX Runtime measured no faster than PyTorch on this hardware (unlike the road model's 2.8×). The ONNX path (`export_yolo_onnx.py`, `YOLO_BACKEND`) exists for edge/INT8 experiments but defaults to `torch`. The one CPU lever that *does* help is input resolution (`YOLO_IMGSZ`: 512 ≈ 1.3×), left at 640 by default.
- **No frame-skip on hazards — on purpose.** The road barely moves between frames, so skipping its inference is safe. A pedestrian or vehicle can appear between *any* two frames, so object detection runs **every frame** — we never trade hazard-detection latency for speed. See the safety notice above.

## Advisory simulation layers (Phase 9)

Beyond perception, the repo demonstrates the *rest* of the ADAS software chain — **all simulation / advisory only, never wired to a vehicle** (see the safety notice above):

- **Bird's-eye perception (Module 5, `perception_bev.py`)** — projects tracked objects into a top-down ego frame (metres), the shared space the other layers reason in.
- **Simulated sensor fusion (Module 6, `sensor_fusion.py`)** — synthesises a noisy "radar" return and fuses it with the camera's noisy depth via an inverse-variance combine. Shows the fusion *architecture* — the radar is **simulated**, not real hardware, and adds no real-world information.
- **Prediction & planning (Module 7, `prediction_planning.py`)** — rolls each object forward (constant-velocity) and suggests a cruise speed (time-gap logic) plus an advisory path.
- **Simulated control (Module 8, `control_sim.py`)** — pure-pursuit steering + PID speed driving a **simulated** ego, drawn as a steering/pedal readout. **There is no hardware interface anywhere in this code, by design.**
- **Driver monitoring (Module 9, `driver_monitoring.py`)** — a driver-facing camera + OpenCV Haar cascades estimate `ATTENTIVE / DISTRACTED / DROWSY / NO DRIVER`.

```bash
# full stack (all overlays on):
python main.py --video dashcam.mp4
# add driver monitoring from a second, driver-facing camera:
python main.py --video dashcam.mp4 --driver-cam 1
# driver monitor on its own:
python driver_monitor_demo.py
```

> These layers exist to show the *shape* of a full ADAS in software. They do **not** make the system safe to drive with — a single camera, a best-effort CPU, simulated sensors, and no certification are exactly why it stays advisory. Re-read the safety notice above.

## Usage

```bash
# Webcam — both modules
python main.py --camera

# Dashcam video — both modules
python main.py --video dashcam.mp4

# Lanes only (skip YOLO)
python main.py --video dashcam.mp4 --no-objects

# Objects only
python main.py --video dashcam.mp4 --no-lanes

# Debug overlay (ROI + raw Hough lines) and save annotated output
python main.py --video dashcam.mp4 --debug --save output.mp4
```

Controls while running:

| Key | Action |
|---|---|
| `Q` | Quit |
| `D` | Toggle debug overlay |
| `P` | Pause / resume |
| `S` | Save screenshot |

## On-screen display

- Green lines — detected left/right lanes, with translucent lane-area fill
- Yellow marker — computed lane centre vs. frame centre
- Coloured boxes — detected objects (green/amber/red by risk); thicker = in your path
- `#id` tags — the stable tracker ID on each confirmed object (watch it stay on the same object)
- **Top-left HUD** — lane FPS, pixel offset, confidence, per-lane status, steering
- **Top-right panel** — the fused decision: longitudinal state (colour-coded, flashes on EMERGENCY), brake bar, lateral action, nearest-in-path object (with its `#id`, distance + TTC), live tracked count, and a `DEGRADED` chip when inputs are untrusted
- **Bottom reason strip** — the one-line, rule-tagged explanation for the current action
- Bottom bar — drift indicator (fills red on hard drift)

## Tuning

Everything is in [config.py](config.py). The three settings that matter most:

| Setting | When to change |
|---|---|
| `ROI_*` fractions | Lanes cut off, or too much roadside noise detected |
| `CANNY_LOW` / `CANNY_HIGH` | Faded markings missed (lower them) or too many edges (raise them) |
| `SMOOTHING_ALPHA` | Lanes jittery (lower it) or slow to react in curves (raise it) |

For the decision engine (Module 3), the settings that matter most:

| Setting | When to change |
|---|---|
| `TTC_BRAKE_S` / `TTC_EMERGENCY_S` | Braking feels late (raise) or too twitchy (lower) |
| `DIST_EMERGENCY_M` / `RISK_DISTANCE_HIGH` | Distance at which it panics vs. brakes firmly |
| `HOLD_FRAMES` / `ESC_*` | Actions flicker (raise the debounce) or react too slowly (lower it) |
| `VULNERABLE_CLASSES` | Which objects (pedestrians, two-wheelers, cattle) earn earlier braking |

## Roadmap

- [x] **Phase 1** — Lane detection + steering suggestion
- [x] **Phase 2** — Object detection + distance + collision warnings (YOLOv8n on CPU)
- [x] **Phase 3** — Decision engine fusing lanes + objects into a single arbitrated action, with temporal debouncing and a degraded mode
- [x] **Phase 4** — Multi-object tracking: stable IDs + per-object kinematics for steadier decisions in dense traffic
- [x] **Phase 5** — Unmarked-road guidance: hybrid paint + drivable-surface following (validated on hill, city and highway footage)
- [x] **Phase 6** — Learned drivable-area segmentation (LRASPP MobileNetV3 trained on the full IDD, val IoU 0.92) — trained free on Colab, **runs on CPU** at run time, integrated as Module 1c
- [ ] **Phase 7** (in progress) — Stronger model: DeepLabV3-MobileNetV3 (ASPP multi-scale head) + night/fog/blur/shadow augmentation + Dice loss, targeting the haze / night / hill weak spots. `colab/phase7_deeplab_aug.ipynb`; loadable via `LEARNED_ARCH = "deeplabv3"`
- [~] **Phase 8** — Real-time inference: **ONNX Runtime CPU backend (~2.8× over PyTorch, identical output) + frame-skip (~3×)** → real-time drivable-area on a plain laptop CPU (done); next: OpenVINO on the Intel iGPU + INT8, more data (BDD100K night + weather)
- [x] **Phase 9** — Advisory *simulation* of the full ADAS chain: bird's-eye perception, simulated sensor fusion, prediction + planning, a simulated pure-pursuit/PID controller, and driver monitoring — all overlay/simulated, **never wired to a vehicle**

## Design principles

1. **Runs on what you have** — CPU-only, standard Python, two dependencies
2. **Fails safely** — the system reports low confidence instead of guessing
3. **Explainable** — every decision can be traced through the pipeline

## License

MIT
