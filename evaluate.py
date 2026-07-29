"""
evaluate.py — the evaluation harness: where does the time go, and is the model
still good? Two reports that turn the demo into something you can defend.

1. LATENCY BUDGET  (always runnable, needs only dashcam.mp4)
   Runs the real modules over N frames and times each one, so you can see the
   per-module cost, its share of the frame, and the end-to-end fps — and know
   which module to optimise next instead of guessing. The learned road model is
   timed as a full inference AND at its frame-skipped amortised cost.

2. DRIVABLE-AREA IoU  (runs if the IDD val set is present)
   Scores the CURRENTLY DEPLOYED road backend (OpenVINO / ONNX / PyTorch —
   whatever config selects) on held-out IDD val images, so a model swap or a
   quantisation step can be checked for a silent accuracy regression, not just a
   speed change. Reports mean per-image IoU and global (pooled) IoU.

    python evaluate.py                    # latency (60 frames) + IoU (120 val imgs)
    python evaluate.py --frames 100 --iou-samples 200
    python evaluate.py --no-iou           # latency only
"""

import argparse
import os
import time

import cv2
import numpy as np

import config as cfg


def _t():
    return time.perf_counter()


# ---------------------------------------------------------------------------
# 1. Latency budget
# ---------------------------------------------------------------------------

def latency_budget(frames: int) -> None:
    from lane_detection import LaneDetector
    from tracker import MultiObjectTracker
    from decision_engine import DecisionEngine

    cap = cv2.VideoCapture("dashcam.mp4")
    if not cap.isOpened():
        print("[latency] dashcam.mp4 not found — skipping latency budget.")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lane = LaneDetector(frame_width=w, frame_height=h)
    trk = MultiObjectTracker()
    eng = DecisionEngine(frame_width=w, frame_height=h)

    obj = None
    try:
        from object_detection import ObjectDetector
        obj = ObjectDetector(frame_width=w, frame_height=h)
    except Exception as exc:
        print(f"[latency] object detector unavailable ({exc}); timing without it.")

    road = None
    try:
        from learned_road_detection import LearnedRoadDetector
        r = LearnedRoadDetector(frame_width=w, frame_height=h)
        road = r if r.available else None
    except Exception:
        pass

    from perception_bev import BEVProjector
    from sensor_fusion import SimRadarFusion
    from prediction_planning import Planner
    from control_sim import SimController
    from forward_collision_warning import ForwardCollisionWarning
    from traffic_light_state import TrafficLightReader
    bev, fus, pln = BEVProjector(w, h), SimRadarFusion(), Planner(w, h)
    ctl, fcw, tlr = SimController(w, h), ForwardCollisionWarning(w, h), TrafficLightReader()

    acc = {}
    def timed(name, fn):
        t0 = _t(); out = fn(); acc[name] = acc.get(name, 0.0) + (_t() - t0) * 1000; return out

    n = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 8000)
    while n < frames:
        ok, frame = cap.read()
        if not ok:
            break
        ann = frame.copy()
        lr = timed("lane", lambda: lane.process(ann))[0]
        if road is not None:
            timed("road_infer(full)", lambda: road._infer_mask(frame))
        orr = None
        if obj is not None:
            orr = timed("object(YOLO)", lambda: obj.process(frame.copy(), lr.lane_center_x))[0]
        dets = orr.detections if orr is not None else []
        tracks = timed("tracker", lambda: trk.update(dets))
        timed("lights", lambda: tlr.read(frame, dets))
        dec = timed("decision", lambda: eng.process(lr, tracks))
        ego = timed("bev", lambda: bev.project(tracks))
        timed("fusion", lambda: fus.process(ego))
        plan = timed("planner", lambda: pln.process(ego, lr, ctl.sim_speed))
        timed("control", lambda: ctl.process(plan, lr))
        timed("fcw", lambda: fcw.process(dec, tracks))
        n += 1
    cap.release()

    if n == 0:
        print("[latency] no frames read.")
        return

    every = max(1, cfg.LEARNED_INFER_EVERY)
    # amortise the frame-skipped road inference
    if "road_infer(full)" in acc:
        acc["road_infer(skip%d)" % every] = acc["road_infer(full)"] / every

    print(f"\n== latency budget ({n} frames, {w}x{h}, backend={cfg.LEARNED_BACKEND}"
          f"/{cfg.LEARNED_OV_DEVICE}) ==")
    print(f"{'module':22s} {'ms/frame':>9s} {'share':>7s}")
    print("-" * 42)
    # the pipeline total uses the amortised road cost, not the full one
    live = {k: v for k, v in acc.items() if k != "road_infer(full)"}
    total = sum(v for k, v in live.items()) / n
    for name, ms in sorted(live.items(), key=lambda kv: -kv[1]):
        per = ms / n
        print(f"{name:22s} {per:9.1f} {per/total*100:6.1f}%")
    print("-" * 42)
    print(f"{'PIPELINE TOTAL':22s} {total:9.1f} {'':>7s}")
    print(f"{'-> end-to-end':22s} {1000.0/total:8.1f} fps")
    if "road_infer(full)" in acc:
        print(f"(note: road_infer full = {acc['road_infer(full)']/n:.1f} ms, run 1/{every} frames)")


# ---------------------------------------------------------------------------
# 2. Drivable-area IoU on IDD val (deployed backend)
# ---------------------------------------------------------------------------

def drivable_iou(samples: int) -> None:
    try:
        import train_local as T
        from learned_road_detection import LearnedRoadDetector
    except Exception as exc:
        print(f"\n[IoU] cannot import eval deps ({exc}) — skipping.")
        return

    try:
        root = T.ensure_extracted()
        val = T.pairs(root, "val")
    except Exception as exc:
        print(f"\n[IoU] IDD val set not available ({exc}) — skipping IoU.")
        return
    if not val:
        print("\n[IoU] no val image/mask pairs found — skipping IoU.")
        return

    val = val[:samples] if samples else val
    det = LearnedRoadDetector(frame_width=T.IN_W, frame_height=T.IN_H)
    if not det.available:
        print("\n[IoU] road model unavailable — skipping IoU.")
        return

    inter_sum = union_sum = 0
    per_image = []
    used = 0
    for ip, mp in val:
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        pred = det._infer_mask(img) > 0                          # (IN_H, IN_W) bool
        gtb = cv2.resize(gt, (T.IN_W, T.IN_H), interpolation=cv2.INTER_NEAREST) > 0
        inter = int(np.logical_and(pred, gtb).sum())
        union = int(np.logical_or(pred, gtb).sum())
        inter_sum += inter
        union_sum += union
        if union > 0:
            per_image.append(inter / union)
        used += 1

    if used == 0:
        print("\n[IoU] could not read any val pairs — skipping.")
        return

    backend = f"{det.backend}" + (f"/{det.ov_device}" if det.ov_device else "")
    print(f"\n== drivable-area IoU (IDD val, {used} images, backend={backend}) ==")
    print(f"  mean per-image IoU : {np.mean(per_image):.4f}")
    print(f"  global pooled IoU  : {inter_sum / max(1, union_sum):.4f}")
    print(f"  weights            : {cfg.LEARNED_MODEL_PATH}")


def main():
    ap = argparse.ArgumentParser(description="ADAS Vision evaluation harness.")
    ap.add_argument("--frames", type=int, default=60, help="frames for the latency budget.")
    ap.add_argument("--iou-samples", type=int, default=120, help="IDD val images for IoU (0=all).")
    ap.add_argument("--no-iou", action="store_true", help="skip the IoU report.")
    ap.add_argument("--no-latency", action="store_true", help="skip the latency budget.")
    args = ap.parse_args()

    if not args.no_latency:
        latency_budget(args.frames)
    if not args.no_iou:
        drivable_iou(args.iou_samples)


if __name__ == "__main__":
    main()
