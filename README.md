# ADAS Vision

An open, low-cost Advanced Driver Assistance System built to run on hardware people already own — a dashcam or webcam and a standard laptop CPU. No GPU, no cloud, no dedicated hardware.

**This release: painted-lane detection, learned drivable-area segmentation for unmarked roads (a compact CNN trained on the full Indian Driving Dataset), object detection, multi-object tracking with stable IDs, a decision engine that fuses it all into a single arbitrated driving action every frame, a staged Forward Collision Warning, traffic-light-state + traffic-sign recognition, OpenVINO acceleration on the Intel iGPU, and an evaluation harness + black-box drive logger to keep it all honest.**

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
- Assigns LOW / MEDIUM / HIGH risk (the driver-facing collision alert is now Module 10)

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

**Forward Collision Warning (Module 10) — the driver-facing alert**
- A staged, escalating collision banner: `CAUTION → WARNING → IMMINENT`, driven by the hazard's live time-to-collision (and a hard distance floor for the imminent case)
- **Consumes the same hazard the decision engine already arbitrated** — a single source of truth, so the banner and the brain can never disagree about which object matters
- Escalates instantly, releases one level at a time (anti-flicker), draws a live TTC countdown, a corner bracket on the threat, a flashing frame border when imminent, and an optional (opt-in) chime

**Scene understanding (Module 11) — signs & lights**
- **Traffic-light state** — reads `RED / AMBER / GREEN` from YOLO's traffic-light boxes by HSV on the lit bulb (with a geometric prior + temporal vote); says `UNKNOWN` in the dark rather than guessing. Works out of the box, no training.
- **Traffic-sign recognition** — colour/shape region proposals → a small CNN (GTSRB, 43 classes) → temporal vote; speed-limit signs latch as the active limit. Gated on its weights file exactly like the road model — inert until you train it (`train_signs.py`, ~5 min on CPU), so a clean checkout is unaffected.
- Both feed the advisory planner: a posted limit caps the cruise target; a red/amber light eases toward a stop.

All of these run together at interactive frame rates on a standard laptop CPU — no GPU (and with OpenVINO, the road model moves onto the Intel iGPU; see *Real-time performance*).

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
├── forward_collision_warning.py ← Module 10: ForwardCollisionWarning (staged TTC alert)
├── traffic_light_state.py       ← Module 11a: TrafficLightReader (RED/AMBER/GREEN)
├── traffic_sign_recognition.py  ← Module 11b: SignRecognizer  (GTSRB CNN + proposals)
├── drive_logger.py      ← Module 12: DriveLogger          (black-box .jsonl recorder)
├── replay_log.py        ← reconstruct a drive from the log alone (no models)
├── evaluate.py          ← eval harness: latency budget + drivable-area IoU
├── test_adas.py         ← regression tests (invariants that must not break)
├── train_local.py       ← CPU trainer for the road model (Module 1c; --adverse = Phase 12)
├── adverse_aug.py       ← Phase 12: rain/snow/glare/low-light/night/fog augmentation
├── prepare_bdd_drivable.py ← Phase 12: index BDD100K drivable data into a manifest
├── train_bdd.py         ← Phase 12: fine-tune on real BDD100K night/weather data
├── train_signs.py       ← CPU trainer for the sign classifier (Module 11b, GTSRB)
├── export_onnx.py / export_openvino.py / export_yolo_openvino.py ← ONNX + OpenVINO IR exporters
├── export_yolo_tflite.py ← Phase 14: YOLOv8n -> per-tensor INT8 TFLite (Axis ARTPEC-8 edge)
├── bench_speed.py / bench_openvino.py  ← per-backend speed benchmarks (--yolo covers YOLO)
├── main.py              ← CLI entry point: runs the whole pipeline together
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

This pulls in OpenCV, NumPy, Ultralytics (which brings a CPU build of PyTorch for Module 2), and torchvision (for Module 1c). On first run, YOLOv8n weights (~6 MB) download automatically. Lane detection alone needs only OpenCV + NumPy — run with `--no-objects` if you haven't installed Ultralytics yet.

**Learned road model (Module 1c):** the default weights are `drivable_idd_lraspp_adv_best.pth` (the Phase 12 adverse fine-tune from [`train_local.py --adverse`](train_local.py)). On a like-for-like A/B it beats the previous fine-tune on **both** clean (0.921 vs 0.919) and the tough night+fog+rain metric (**0.907 vs 0.826**, +8 pts) — more robust at night/in rain with no daytime cost. The base weights `drivable_idd_full_best.pth` come from [`colab/phase6c_full_idd.ipynb`](colab/phase6c_full_idd.ipynb) (val IoU 0.92). Place the `.pth` in the repo root (large files are shared via a GitHub Release, not committed) and run `python export_onnx.py` (and `python export_openvino.py` for the iGPU) once to build the fast backends. Without any weights, the system automatically falls back to the classical road detector (Module 1b) — nothing breaks, unmarked-road guidance is just less accurate.

**Optional extras (all graceful — the system runs without them):**
- **OpenVINO iGPU acceleration** — `pip install openvino nncf`, then `python export_openvino.py`. On Intel hardware this is the fastest road-model backend (see *Real-time performance*); without it, `LEARNED_BACKEND="openvino"` falls back to ONNX then PyTorch.
- **Traffic-sign recognition (Module 11b)** — needs `gtsrb_sign_cnn.pth`, trained in ~5 min on CPU with `python train_signs.py` once you've downloaded GTSRB (Kaggle: *GTSRB — German Traffic Sign*). Until then the sign recogniser is inert; traffic-**light** state works with no extra setup.

## Adverse-condition robustness (Phase 12)

Two independent ways to make drivable-area detection hold up at night and in bad weather — the model's known weak spot. Both are measured against a deliberately **tougher** hard metric than before (night **+ fog + rain**, not just night+fog); on it the Phase 6c model scores ~0.78 vs ~0.92 on the old metric, so there's genuine headroom.

**1. Richer synthetic augmentation (no download).** [`adverse_aug.py`](adverse_aug.py) adds rain streaks, snow, sun/headlight glare, and low-light sensor noise to the existing night/fog/blur/shadow. A 28% share of frames stay near-clean so daytime skill isn't forgotten. Retrain on the IDD data you already have:

```bash
python train_local.py --adverse --epochs 12 --subset 2500
```

It writes to `drivable_idd_lraspp_adv_best.pth` and its own checkpoint — the adopted model is never touched, so it's a clean A/B.

**2. Real BDD100K night/weather data.** Synthetic effects approximate; BDD100K has ~70k real driving frames tagged by time-of-day and weather. Download (free [BDD account](https://bdd-data.berkeley.edu) required) the **100K Images** and **Drivable Area** labels (optionally the **Detection** labels JSON for the weather/time attributes), unzip under one folder, then:

```bash
python prepare_bdd_drivable.py --bdd-root "C:\path\to\bdd100k"
python train_bdd.py --epochs 10 --subset 4000
```

`prepare_bdd_drivable.py` auto-detects the mask's background value and tags each frame night/adverse; `train_bdd.py` fine-tunes on top of the IDD model with a **weighted sampler** (night ×3, rain/snow/fog ×2) and selects on IoU over BDD val's **real** night/adverse frames. Both scripts self-test with `--smoke` (no data needed). Result weights: `drivable_bdd_lraspp_best.pth`.

## Real-time performance

The learned road model (Module 1c) has three stackable, retraining-free speed levers:

- **OpenVINO on the Intel iGPU** (`LEARNED_BACKEND = "openvino"`, the default) — runs the *same* network on the integrated GPU that otherwise sits idle, and in doing so frees the CPU for YOLO (the real bottleneck — see below). Export the IR once, then benchmark:
  ```bash
  python export_openvino.py     # ONNX -> FP16 IR + NNCF INT8 IR (needs: pip install openvino nncf)
  python bench_openvino.py      # times every backend on your machine
  ```
- **ONNX Runtime backend** (`LEARNED_BACKEND = "onnx"`) — the portable fast path: runs the same weights ~2.8× faster than PyTorch on CPU, byte-identical segmentation, no Intel hardware needed. Export with `python export_onnx.py`.
- **Frame-skip** (`LEARNED_INFER_EVERY = 3`) — runs the CNN every Nth frame and reuses the mask between (the road barely moves frame-to-frame); the centreline still updates every frame. ~3× effective throughput.

Measured on an **i7-8650U + Intel UHD 620 iGPU** (`bench_openvino.py`, 768×432):

| backend | device | ms/frame | raw fps | eff. fps (skip 3) |
|---|---|---:|---:|---:|
| PyTorch | CPU | 264 | 3.8 | 11 |
| ONNX Runtime | CPU | 124 | 8.0 | 24 |
| **OpenVINO FP16** | **iGPU** | **43** | **23** | **69** |
| OpenVINO INT8 | iGPU | 46 | 22 | 66 |
| OpenVINO FP16/INT8 | CPU | ~365 | 2.7 | 8 |

The iGPU path is **2.9× over ONNX-CPU and 6.1× over PyTorch**, at 100% argmax parity with the FP32 reference (and 0.91 drivable-IoU on held-out IDD val — no accuracy regression). Two honest findings recorded alongside: **INT8 gives no benefit on this iGPU** (the UHD 620 lacks strong INT8 acceleration; FP16 ties or beats it — INT8 is still exported because it helps on VNNI CPUs / NPUs), and **OpenVINO's *CPU* plugin is slower than ONNX Runtime here**, so CPU-only machines should stay on `"onnx"`.

The whole backend chain is graceful: `openvino` → `onnx` → `torch`. If OpenVINO isn't installed, or no IR / `.onnx` is present, it silently falls to the next tier, so a clean checkout still runs.

**Object detection (Module 2) — the biggest cost, so it moved to the iGPU too (Phase 11).**

- **On CPU, YOLOv8n is stubborn.** ONNX Runtime gives it no speedup (it's tiny and overhead-bound — preprocess + NMS dominate), so that path (`export_yolo_onnx.py`) exists only for edge/INT8 experiments. But **OpenVINO on the Intel iGPU is a real 2.9× win** (torch-CPU 183 ms → iGPU 63 ms, 95% detection parity) — the forward pass dominates even here, so offloading it to the iGPU pays off. Export once with `export_yolo_openvino.py`; `YOLO_BACKEND="openvino"` is the default on Intel hardware and falls back to `torch` if the IR or openvino is absent.
- **Both models share one iGPU — measure, don't assume.** With YOLO *and* the road model on the UHD 620, the iGPU move alone took the honest end-to-end (warmed up, road at its real frame-skip cadence) from **~3.1 → ~7.2 fps**; halving the lane detector's resolution then unsaturated the CPU and pushed it to **~14 fps**. (Naively timing both models on the GPU *every* frame looks far worse; that's iGPU context-thrash, an artefact — `bench_openvino.py` and the corrected latency budget capture the real number.)
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

## Edge deployment feasibility (Axis ARTPEC-8)

A detection model doesn't have to run on a laptop — an Axis camera SoC (**ARTPEC-8**) can run YOLO on its **DLPU** via the `larod` runtime, which wants a **per-tensor INT8 TFLite** model. [`export_yolo_tflite.py`](export_yolo_tflite.py) produces exactly that:

```bash
pip install tensorflow onnx2tf tf_keras   # optional edge extras (not in requirements.txt)
python export_yolo_tflite.py              # YOLOv8n -> per-tensor INT8 TFLite, dashcam-calibrated
```

What's **confirmed**: the conversion runs cross-platform (via `onnx2tf`, sidestepping Ultralytics' Linux-only TFLite gate) and yields the ARTPEC-8 artifact — `yolov8n_full_integer_quant.tflite`, **3.3 MB, ~3.9× smaller** than FP32, calibrated on real frames. The FP32 TFLite matches the PyTorch model **100%** on detections, so the graph translation is faithful.

What's **not yet measured**, honestly: the INT8 *accuracy number*. Desktop TFLite CPU kernels can't execute the quantised graph (the reference sigmoid demands output scale `1/256`; XNNPACK won't delegate onnx2tf's transposes) — both are host-runtime limits that **don't apply to the DLPU**. Verify INT8 accuracy on Linux/Colab (`YOLO(tflite).val(...)`) or on the device itself. Prompted by a reader's suggestion on the project's LinkedIn — a genuine edge path, scoped honestly.

## Evaluation & recording

Measurement and reconstruction, so the system is defensible rather than just demo-able.

**Evaluation harness** — `python evaluate.py`:
- **Latency budget** — times every module over real frames so the bottleneck is *measured, not guessed*. It drove all of Phase 11. YOLO started at **77% of the frame** (~249 ms on CPU); moving it to the iGPU cut it to ~57 ms. That promoted the **lane detector** to the top CPU cost, so it moved to half-resolution (~48 → ~9 ms) — which, by unsaturating the CPU, also sped up YOLO's CPU-side work. Net whole-pipeline: **3.1 → ~14 fps**. (Chasing this also caught two bugs in the harness itself — missing warmup and timing the road model every frame instead of at its frame-skip cadence — both fixed.)
- **Drivable-area IoU** — scores the *currently deployed* road backend on held-out IDD val, so a model swap or a quantisation step can be checked for a silent accuracy regression. The OpenVINO/FP16 path scores mean **0.906** / pooled **0.911** IoU — matching training.

**Black-box drive logger** — `python main.py --video dashcam.mp4 --log drive.jsonl` records one compact JSON line per frame (what it saw + decided). Reconstruct the whole drive from the log *alone* — no models re-run:

```bash
python replay_log.py drive.jsonl                                   # drive summary
python replay_log.py drive.jsonl --video dashcam.mp4 --save replay.mp4   # rebuild the annotated video
python replay_log.py drive.jsonl --video dashcam.mp4 --dump 300 f.jpg    # one reconstructed frame
```

**Regression tests** — `python test_adas.py` (also `pytest`-discoverable) guards the safety-relevant invariants: threshold ordering, decision-engine escalation + single-frame-spurious rejection, FCW monotonic escalation + hysteretic release, tracker ID stability + ghost rejection, traffic-light state, and live OpenVINO-vs-ONNX mask parity.

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

# Record a black-box drive log (replay it later with replay_log.py)
python main.py --video dashcam.mp4 --log drive.jsonl
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
- **Top-centre banner** — the Forward Collision Warning (Module 10): amber `COLLISION RISK` → orange `COLLISION WARNING` → flashing red `BRAKE`, with a live TTC countdown and a bracket on the threat
- **Top-left chips** — traffic-light state (`LIGHT: RED/AMBER/GREEN`) and, when a limit sign is recognised, a speed-limit roundel
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
- [x] **Phase 8** — Real-time inference: **ONNX Runtime CPU backend (~2.8× over PyTorch) + frame-skip (~3×)**, then **OpenVINO on the Intel iGPU (2.9× over ONNX-CPU, 6.1× over PyTorch, 100% parity)** with an INT8 export path — real-time drivable-area that also frees the CPU for YOLO. Remaining: more data (BDD100K night + weather)
- [x] **Phase 9** — Advisory *simulation* of the full ADAS chain: bird's-eye perception, simulated sensor fusion, prediction + planning, a simulated pure-pursuit/PID controller, and driver monitoring — all overlay/simulated, **never wired to a vehicle**
- [x] **Phase 10** — Driver-facing & rigour: staged **Forward Collision Warning** (Module 10), **traffic-light-state + traffic-sign recognition** (Module 11), an **evaluation harness** (latency budget + drivable-area IoU), a **black-box drive logger + replay** (Module 12), and a **regression-test suite**
- [x] **Phase 11** — Speed up the bottlenecks the latency budget exposed: **YOLO on the Intel iGPU** (OpenVINO, 2.9× over torch-CPU, 95% detection parity), then the **lane detector at half-resolution** (per-pixel HSV/Canny/Hough, 96% agreement, ~1.9× — and freeing the CPU sped up YOLO too). Honest end-to-end **3.1 → ~14 fps**. Next: BDD100K night + weather data
- [x] **Phase 12** — Adverse-condition robustness. (a) **Adopted**: a **rich synthetic augmentation suite** (`adverse_aug.py` — rain, snow, glare, low-light sensor noise on top of night/fog) drove an IDD retrain (`train_local.py --adverse`) that lifted the tough **night+fog+rain** IoU from **0.78 → 0.91** and beat the prior fine-tune on both clean and hard — now the default road model. (b) **Ready**: a **BDD100K pipeline** (`prepare_bdd_drivable.py` + `train_bdd.py`) that fine-tunes on **real** night/rain/snow driving data with a night/weather-weighted sampler and a real-adverse validation metric (needs the gated BDD download)
- [x] **Phase 13** — Overnight hardening. Safety regression tests expanded **9 → 19** (emergency latch, gradual release, degraded floor, advisory-never-brakes, lateral-inhibited-toward-hazard, vulnerable-earlier-brake, occlusion survival, frame-guard). A **full-clip weakness sweep** (17,982 frames) and two fixes: **defensive frame-health guards** (malformed frames degrade, never crash) and **corridor mask cleanup** (open/close + connected-component filter — cuts drivable-mask fragmentation, e.g. 6→2 blobs on noisy frames, IoU held at 0.92). *(The spine figure claimed here was later found to be measured off `main.py`'s real decision path — see Phase 15 for the honest re-audit and the fix it surfaced.)*
- [~] **Phase 14** (feasibility) — On-camera edge inference: YOLOv8n → **per-tensor INT8 TFLite** for the Axis **ARTPEC-8** DLPU (`export_yolo_tflite.py`). Conversion + format + graph fidelity confirmed (3.3 MB, ~3.9× smaller, FP32 graph 100% faithful); INT8 accuracy verification is an on-device / Linux step. Next: BDD100K real night/weather data
- [x] **Phase 15** — Honest safety re-audit + real decision-path fix. Phase 13's sweep fed the *raw* lane result to the engine and never wired in the road-model fallback `main.py` actually runs — so it both inflated the degraded rate (90% vs a true **~39%**) and tested "never PROCEED into a hazard" on an artificially conservative path (degraded mode suppresses PROCEED). A committed auditor (`safety_audit.py`) now mirrors the real chain (lanes → learned road → classical → objects → tracker → engine) and re-ran all 17,982 frames: it caught **one** frame that read PROCEED for a single frame as a motorcyclist crossed into the corridor at 11 m — the temporal ratchet's confirmation lag. Fixed with a **vulnerable-proximity floor** (a confirmed pedestrian/rider/animal in-path within 15 m immediately floors to CAUTION; one-directional, can only raise caution). A deterministic dual-engine A/B (floor on vs off, identical inputs) confirmed: spine **1 → 0**, **0** frames made less cautious, cost **exactly 1 frame** (0.01%) eased PROCEED→CAUTION. Regression tests **19 → 21**
- [x] **Phase 16** — Constant-velocity **tracker motion model** (real-time foundation). Each track now carries an EMA per-corner pixel velocity: a coasting (unmatched) track is associated against its *predicted* box, and a *known closing* hazard keeps a live, **bounded** distance/TTC across coast frames instead of freezing — so a hazard doesn't go stale when perception runs below the frame rate. One-directional (only keeps caution live), `TRACK_PREDICT_ON_COAST` toggles the exact prior behaviour. Full-clip `safety_audit.py`: spine holds, churn unchanged. Tests **21 → 25**
- [x] **Phase 17** — Real-time safety net (**cadence + fail-safe degrade**). A full-clip **cadence audit** measured the safe envelope: detection every ≤2 frames of the 60 fps clip (≥30/s, ≤33 ms) keeps the spine clean; slower leaves one vehicle-boundary gap (a bus at 14.6 m). Rather than a fragile looming reflex (tried — a naive optical-flow τ cue can't separate forward egomotion from a real obstacle, so it was dropped), the engine now degrades on **detection staleness** and, in **reduced-cadence mode**, falls back to **raw proximity** — it won't PROCEED past any confirmed in-path object within 15 m when the coasted kinematics are unreliable. Validated across cadences: spine **closes at every cadence (1 → 0)**, **zero** cost at the safe rate, extra CAUTION only when detection is behind. Tests **25 → 27**
- [x] **Phase 18** — Async object detection (Module 14, `async_detector.py`) + a YOLO speed study. First, the obvious speedups were **ruled out by benchmark** (all 6 variants, imgsz {640,512,416} × {FP16,INT8}): **INT8 is *slower* on this iGPU** (its FP16 path is already optimal) and **imgsz < 640 goes blind** — safety-class recall 640 **98.6%** → 512 **77.5%** → 416 **67.6%**. So FP16@640 stays. But it's only **28 ms / 35 fps alone** — already inside the safe budget; the sync pipeline just never *runs* it that often (all stages serial on one thread). So the lever is **parallelism**: a background-thread, newest-frame-wins detector that feeds the engine the real detection age, leaving the Phase-17 net in charge of safety. Validated full-clip: **spine holds (0)**, ~**2× throughput when the road model is idle** (marked roads). **Honest ceiling** — on unmarked roads the LRASPP road model and YOLO contend for the single iGPU (no CUDA), so there's no throughput gain there; true real-time dual-net perception needs more compute (Jetson, or the Phase-14 Axis edge camera). Opt-in via `--async`; off by default. Tests **27 → 28**
- [x] **Phase 19** — Hardware portability for real dual-net real-time. Phase 18 established that fast, confident dual-net perception needs hardware that can run both nets at once — which this laptop's single iGPU (no CUDA) cannot. Added **CUDA-first device selection** (`config.PREFER_CUDA`): when a GPU is present, both YOLO and the road model run on **CUDA** (with an optional TensorRT `.engine`); on a machine without CUDA it's a no-op and the OpenVINO-iGPU / CPU paths run unchanged (28 tests + full-clip audit confirm the laptop fallback is untouched). Plus **[DEPLOYMENT.md](DEPLOYMENT.md)** — target hardware (Jetson Orin / NVIDIA GPU), setup, TensorRT path, and an **on-device validation protocol** to confirm spine + throughput on the real board. *Honest scope: the CUDA path is written correct-by-construction and validated only in its CPU/iGPU fallback here — the dev machine has no GPU, so on-target numbers are the user's to confirm.*

## Design principles

1. **Runs on what you have** — CPU-only, standard Python, two dependencies
2. **Fails safely** — the system reports low confidence instead of guessing
3. **Explainable** — every decision can be traced through the pipeline

## License

MIT
