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
    print(f"[bench] {len(frames)} frames | source {w}x{h} | model input "
          f"{cfg.LEARNED_INPUT_W}x{cfg.LEARNED_INPUT_H} | frame-skip every {every}")

    for backend in ("torch", "onnx"):
        fps, used = bench(backend, frames, w, h)
        if fps is None:
            print(f"  {backend:6s}: {used}")
            continue
        print(f"  {backend:6s}: {fps:5.1f} fps raw   ->  {fps*every:5.1f} fps effective")

    print("[bench] 'effective' is what the live pipeline delivers for road guidance.")
    print("[bench] 15+ effective fps = real-time-smooth for video.")


if __name__ == "__main__":
    main()
