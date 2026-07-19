# ADAS Vision

An open, low-cost Advanced Driver Assistance System built to run on hardware people already own — a dashcam or webcam and a standard laptop CPU. No GPU, no cloud, no dedicated hardware.

**Phase 1 (this release): Real-time lane detection with live steering suggestions.**

## What it does

- Detects lane markings from dashcam footage or a live webcam in real time
- Calculates the vehicle's drift from lane centre every frame
- Outputs a live steering suggestion (`STRAIGHT`, `SLIGHT LEFT`, `MODERATE RIGHT`, `HARD LEFT`, ...)
- Reports a confidence score and explicitly says so when it cannot see lanes
- Runs at ~30 FPS on a standard laptop CPU

## How it works

Classical computer vision — no neural network in this phase:

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

Well-tuned classical CV runs faster than deep learning on CPU and is fully explainable when it fails — the right starting point for a safety-critical system.

## Project layout

```
adas-vision/
├── config.py           ← All tunable parameters (ROI, Canny, Hough, smoothing)
├── lane_detection.py   ← LaneDetector class — the full pipeline
├── main.py             ← CLI entry point: webcam or video file
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

That's it — only OpenCV and NumPy.

## Usage

```bash
# Webcam
python main.py --camera

# Dashcam video file
python main.py --video dashcam.mp4

# Debug overlay (ROI trapezoid + raw Hough lines) and save annotated output
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

- [x] **Phase 1** — Lane detection + steering suggestion (this release)
- [ ] **Phase 2** — Object detection: vehicles, pedestrians, two-wheelers (YOLOv8n on CPU)
- [ ] **Phase 3** — Decision engine combining lanes + objects
- [ ] **Phase 4** — Indian-road robustness: unmarked lanes, mixed traffic, night driving

## Design principles

1. **Runs on what you have** — CPU-only, standard Python, two dependencies
2. **Fails safely** — the system reports low confidence instead of guessing
3. **Explainable** — every decision can be traced through the pipeline

## License

MIT
