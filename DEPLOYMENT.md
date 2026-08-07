# Deployment — running both nets in real time on capable hardware

This project runs two neural nets per frame: **YOLOv8n** (object detection) and
an **LRASPP-MobileNetV3** learned drivable-area model. On the development laptop
(Intel i7-8650U + UHD 620 iGPU, **no CUDA**) they must share the one integrated
GPU, and a full-clip audit showed the hard limit:

- YOLO fp16@640 alone is **~28 ms / 35 fps** — already within the safe budget.
- But on unmarked roads (~68% of the test clip) the road model and YOLO **contend
  for the single iGPU**, so end-to-end drops to ~12 fps and detection latency
  balloons (p90 ~124 ms). Two nets, one iGPU, no CUDA — a hardware wall, not a
  software bug. (See README Phases 17–18.)

The system stays **safe** on the laptop — the Phase 15–17 machinery degrades to
CAUTION when detection falls behind, so the safety spine holds at every cadence —
but it is *conservative* (frequent CAUTION near objects). To get **fast AND
confident** real-time dual-net perception you need hardware that can run both nets
at once. This document is how to take it there.

> ⚠️ **Advisory only, always.** On any hardware, this system warns a human. It is
> never wired to steering, throttle, or brakes. More compute makes it faster and
> less conservative — it does **not** make it safe to hand a CPU/GPU model control
> of a moving vehicle.

> **Honesty note.** The performance figures below are **estimates** from published
> YOLOv8n/Jetson benchmarks, not measured on this project. The development machine
> has no CUDA, so the CUDA path here is written to be correct-by-construction and
> is exercised only in its CPU/iGPU fallback. **Confirm every number on the target
> with the validation protocol at the bottom before trusting it.**

---

## Why a real GPU removes the wall

A discrete NVIDIA GPU or a Jetson SoC has far more compute than the UHD 620 **and**
supports genuine concurrent inference (CUDA streams; TensorRT contexts). Both nets
run on the GPU at once instead of time-slicing one iGPU, so:

- the road model no longer steals cycles from YOLO,
- YOLO's detection interval stays near its native ~28 ms (under the safe budget),
- Phase-17 reduced-cadence mode rarely engages → confident **PROCEED** is restored,
- and the async loop (`--async`) delivers its throughput instead of being masked
  by contention.

## Recommended targets

| Target | Class | Est. dual-net throughput* | Notes |
|---|---|---|---|
| **Jetson Orin Nano (8 GB)** | Embedded, in-vehicle | ~30–60 fps | Best "in a car" fit; JetPack + TensorRT; ~7–15 W |
| **Jetson Orin NX (16 GB)** | Embedded, headroom | ~60–100+ fps | More margin for higher-res / bigger road model |
| **Desktop/mini-PC + RTX GPU** | Dev / bench | 100+ fps | Fastest way to *see and validate* the dual-net at speed |

\* *Estimates for YOLOv8n@640 + LRASPP@768×432 with TensorRT FP16, from published
per-model Jetson numbers. Verify on-device — do not quote these as measured.*

## How the code targets a GPU

Device selection is **CUDA-first with automatic fallback** — no code edits needed:

- `config.PREFER_CUDA = True` (default). When `torch.cuda.is_available()`, both the
  object detector (`object_detection.py`) and the road model
  (`learned_road_detection.py`) run on **CUDA**; the Intel-only OpenVINO paths are
  skipped. On a machine without CUDA (the laptop) it is a no-op — OpenVINO-iGPU /
  CPU run exactly as before.
- The 28-test suite and the full-clip safety audit pass unchanged on the laptop,
  confirming the fallback is untouched.

### 1. Baseline (torch-CUDA) — simplest, works immediately

Install a CUDA build of PyTorch for the target (on Jetson, use the NVIDIA-provided
JetPack wheels; on desktop, the CUDA wheel from pytorch.org), plus `ultralytics`.
Then just run — CUDA is auto-detected:

```bash
python main.py --video dashcam.mp4 --async
```

`PREFER_CUDA` routes YOLO (its `.pt` on the GPU) and the road model (torch on the
GPU) to CUDA. Set `LEARNED_BACKEND = "torch"` in `config.py` so the road model
uses the torch-CUDA path rather than probing for an Intel iGPU.

### 2. Optimised (TensorRT) — for maximum headroom

TensorRT gives the largest speedup on Jetson. Export on the target (TensorRT
engines are hardware-specific and must be built on the device):

```bash
# YOLO -> TensorRT engine (run ON the target)
yolo export model=yolov8n.pt format=engine half=True imgsz=640
```

Point the config at it and keep `PREFER_CUDA = True`:

```python
YOLO_TRT_MODEL = "yolov8n.engine"   # used instead of the .pt when CUDA is present
```

For the road model, export its ONNX (`export_onnx.py`) and build a TensorRT engine
with `trtexec` on the target, or run the torch-CUDA path (already fast enough on
Orin for a MobileNet-class net).

---

## On-device validation protocol (do this before trusting it)

Run these on the target hardware, in order. This is the same discipline used
throughout the project — measure, don't assume.

1. **Unit safety tests** — the decision logic must pass everywhere:
   ```bash
   python -m pytest test_adas.py -q          # expect: 28 passed
   ```
2. **Confirm CUDA is actually in use** (not silent CPU fallback):
   ```bash
   python -c "from object_detection import ObjectDetector as O; from learned_road_detection import LearnedRoadDetector as R; import logging; logging.disable(30); print('YOLO', O(1280,720).backend); r=R(1280,720); print('road', r.backend, r._torch_device)"
   # expect: YOLO cuda  /  road torch cuda
   ```
3. **Safety spine (the load-bearing check)** — must be 0 on the real decision path:
   ```bash
   python safety_audit.py --video dashcam.mp4          # expect: VERDICT spine HOLDS, 0 violations
   ```
4. **Throughput + latency** — confirm the dual-net actually runs fast and that
   detection latency sits under the safe budget (~33 ms), which is what removes the
   Phase-17 reduced-cadence conservatism:
   ```bash
   python main.py --video dashcam.mp4 --async          # watch the fps / DEGRADED chip in the HUD
   ```
   A clean result: high, steady fps with the `DEGRADED` chip rarely showing near
   objects (unlike the laptop, where it shows often on unmarked roads).

If steps 3–4 don't come out clean on the target, treat it as a finding to
investigate — not a number to report — exactly as on the laptop.

---

## Alternative: on-camera edge (Axis ARTPEC-8)

A different architecture, scoped in Phase 14 (`export_yolo_tflite.py`): run
detection **on the camera's DLPU** (per-tensor INT8 TFLite via the larod runtime),
so detection never competes for a host GPU at all. This offloads the object net to
dedicated silicon rather than running two nets on one host — a good fit if the
deployment is camera-centric. The road model and decision logic would run on the
host consuming the camera's detections. INT8 accuracy must still be validated
on-device (see that script's docstring).
