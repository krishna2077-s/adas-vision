# ADAS Vision

An open, low-cost Advanced Driver Assistance System built to run on hardware people already own — a dashcam or webcam and a standard laptop CPU. No GPU, no cloud, no dedicated hardware.

**Phases 1 & 2 (this release): Real-time lane detection with steering suggestions, plus object detection with collision warnings.**

## What it does

**Lane detection (Module 1)**
- Detects lane markings from dashcam footage or a live webcam in real time
- Calculates the vehicle's drift from lane centre every frame
- Outputs a live steering suggestion (`STRAIGHT`, `SLIGHT LEFT`, `MODERATE RIGHT`, `HARD LEFT`, ...)
- Reports a confidence score and explicitly says so when it cannot see lanes

**Object detection (Module 2)**
- Detects road-relevant objects with YOLOv8n — cars, trucks, buses, pedestrians, two-wheelers, animals
- Estimates each object's distance from a monocular bounding-box heuristic
- Determines which objects are in the vehicle's forward path (perspective-aware corridor)
- Assigns LOW / MEDIUM / HIGH risk and raises a Forward Collision Warning banner

Both modules run together at interactive frame rates on a standard laptop CPU — no GPU.

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

> **Note on distance:** a single camera cannot measure true distance. Estimates come from apparent object size and are meant for relative "is this getting closer" logic, not survey-grade measurement.

## Project layout

```
adas-vision/
├── config.py            ← All tunable parameters (both modules)
├── lane_detection.py    ← Module 1: LaneDetector
├── object_detection.py  ← Module 2: ObjectDetector (YOLOv8n)
├── main.py              ← CLI entry point: runs both modules together
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

That's it — only OpenCV and NumPy.

On first run, YOLOv8n weights (~6 MB) download automatically.

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
- Bottom bar — drift indicator (fills red on hard drift)
- HUD — FPS, pixel offset, confidence, per-lane detection status, steering decision

## Tuning

Everything is in [config.py](config.py). The three settings that matter most:

| Setting | When to change |
|---|---|
| `ROI_*` fractions | Lanes cut off, or too much roadside noise detected |
| `CANNY_LOW` / `CANNY_HIGH` | Faded markings missed (lower them) or too many edges (raise them) |
| `SMOOTHING_ALPHA` | Lanes jittery (lower it) or slow to react in curves (raise it) |

## Roadmap

- [x] **Phase 1** — Lane detection + steering suggestion
- [x] **Phase 2** — Object detection + distance + collision warnings (YOLOv8n on CPU)
- [ ] **Phase 3** — Decision engine fusing lanes + objects into a single driving action
- [ ] **Phase 4** — Indian-road robustness: unmarked lanes, mixed traffic, night driving

## Design principles

1. **Runs on what you have** — CPU-only, standard Python, two dependencies
2. **Fails safely** — the system reports low confidence instead of guessing
3. **Explainable** — every decision can be traced through the pipeline

## License

MIT
