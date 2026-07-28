"""
bench_speed.py — measure drivable-area inference speed on YOUR CPU.

Compares PyTorch vs ONNX Runtime at the configured resolution, then shows the
EFFECTIVE throughput once frame-skip (LEARNED_INFER_EVERY) is applied — that's
the number the live app actually delivers.

    raw fps        = how fast one CNN inference runs
    effective fps  = raw x LEARNED_INFER_EVERY  (frame-skip reuses the mask)

Run:
    python bench_speed.py                         # 40 frames from dashcam.mp4
    python bench_speed.py --frames 60

NOTE: run this when the CPU is otherwise idle. If a training job is running,
absolute fps will be depressed — but the torch-vs-onnx RATIO stays meaningful.

Advisory / research only.
"""

import argparse
import time

import cv2

import config as cfg
from learned_road_detection import LearnedRoadDetector


def grab_frames(path, n, step=200):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"[bench] cannot open {path}")
    frames, fi = [], 0
    while len(frames) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        fi += step
    cap.release()
    return frames


def bench(backend, frames, w, h):
    cfg.LEARNED_BACKEND = backend
    det = LearnedRoadDetector(frame_width=w, frame_height=h)
    if not det.available or det.backend != backend:
        return None, det.backend if det.available else "unavailable"
    det._infer_mask(frames[0])                 # warm-up (graph opt / cache)
    t0 = time.time()
    for f in frames:
        det._infer_mask(f)
    dt = time.time() - t0
    return len(frames) / dt, backend


def bench_yolo(frames):
    """Time YOLO (Module 2) with the PyTorch .pt vs the exported .onnx."""
    import os
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed — skipping YOLO")
        return
    for name, path in (("torch", cfg.YOLO_MODEL),
                       ("onnx", getattr(cfg, "YOLO_ONNX_MODEL", "yolov8n.onnx"))):
        if not os.path.exists(path):
            print(f"  {name:6s}: {path} not found (run export_yolo_onnx.py)")
            continue
        m = YOLO(path)
        m(frames[0], conf=cfg.YOLO_CONF_THRESHOLD, iou=cfg.YOLO_IOU_THRESHOLD, verbose=False)
        t0 = time.time()
        for f in frames:
            m(f, conf=cfg.YOLO_CONF_THRESHOLD, iou=cfg.YOLO_IOU_THRESHOLD, verbose=False)
        dt = time.time() - t0
        print(f"  {name:6s}: {len(frames)/dt:5.1f} fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dashcam.mp4")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    frames = grab_frames(args.video, args.frames)
    if not frames:
        raise SystemExit("[bench] no frames read")
    h, w = frames[0].shape[:2]
    every = cfg.LEARNED_INFER_EVERY
    print(f"[bench] {len(frames)} frames | source {w}x{h}")

    print(f"[bench] road segmentation (Module 1c) @ {cfg.LEARNED_INPUT_W}x{cfg.LEARNED_INPUT_H}, "
          f"frame-skip every {every}:")
    for backend in ("torch", "onnx"):
        fps, used = bench(backend, frames, w, h)
        if fps is None:
            print(f"  {backend:6s}: {used}")
            continue
        print(f"  {backend:6s}: {fps:5.1f} fps raw   ->  {fps*every:5.1f} fps effective")

    print("[bench] YOLO object detection (Module 2), every frame:")
    bench_yolo(frames)

    print("[bench] 'effective' is what the live pipeline delivers.")
    print("[bench] 15+ fps = real-time-smooth for video.")


if __name__ == "__main__":
    main()
