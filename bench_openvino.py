"""
bench_openvino.py — Phase 8: measure the road model across every backend.

Times raw single-frame inference of the drivable-area model on:
    PyTorch (CPU)  |  ONNX Runtime (CPU)  |  OpenVINO FP16/INT8 on CPU and iGPU

and prints ms/frame, raw fps, and the EFFECTIVE fps once the frame-skip
(LEARNED_INFER_EVERY) is applied — the number that actually matters live. Use it
to pick LEARNED_BACKEND / LEARNED_OV_DEVICE for this machine from real data.

    python bench_openvino.py --iters 40
"""

import argparse
import os
import time

import cv2
import numpy as np

import config as cfg

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def sample_input():
    cap = cv2.VideoCapture("dashcam.mp4")
    ok, f = cap.read()
    cap.release()
    if not ok:
        f = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    img = cv2.cvtColor(cv2.resize(f, (cfg.LEARNED_INPUT_W, cfg.LEARNED_INPUT_H)), cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


def timeit(fn, x, iters, warmup=5):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0            # ms/frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    x = sample_input()
    every = max(1, cfg.LEARNED_INFER_EVERY)
    rows = []

    # ── PyTorch ────────────────────────────────────────────────────────────
    try:
        import torch
        from torch import nn
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        m = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
        m.classifier.low_classifier = nn.Conv2d(40, 2, 1)
        m.classifier.high_classifier = nn.Conv2d(128, 2, 1)
        m.load_state_dict(torch.load(cfg.LEARNED_MODEL_PATH, map_location="cpu"))
        m.eval()
        xt = torch.from_numpy(x)
        def run_torch(_):
            with torch.no_grad():
                return m(xt)["out"]
        rows.append(("PyTorch", "CPU", timeit(lambda _: run_torch(_), x, args.iters)))
    except Exception as exc:
        print("PyTorch bench skipped:", exc)

    # ── ONNX Runtime ─────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        if os.path.exists(cfg.LEARNED_ONNX_PATH):
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(cfg.LEARNED_ONNX_PATH, so, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0].name
            rows.append(("ONNX RT", "CPU", timeit(lambda z: sess.run(None, {inp: z}), x, args.iters)))
    except Exception as exc:
        print("ONNX bench skipped:", exc)

    # ── OpenVINO: each IR on each available device ──────────────────────────
    try:
        import openvino as ov
        core = ov.Core()
        variants = [("OV FP16", getattr(cfg, "LEARNED_OV_MODEL_FP16", "")),
                    ("OV INT8", getattr(cfg, "LEARNED_OV_MODEL", ""))]
        for tag, xml in variants:
            if not xml or not os.path.exists(xml):
                continue
            model = core.read_model(xml)
            for dev in ("CPU", "GPU"):
                if dev not in core.available_devices:
                    continue
                try:
                    comp = core.compile_model(model, dev)
                    out = comp.output(0)
                    rows.append((tag, dev, timeit(lambda z: comp(z)[out], x, args.iters)))
                except Exception as exc:
                    print(f"{tag} on {dev} skipped:", exc)
    except Exception as exc:
        print("OpenVINO bench skipped:", exc)

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\nRoad model {cfg.LEARNED_INPUT_W}x{cfg.LEARNED_INPUT_H}  "
          f"| iters={args.iters}  | frame-skip={every}\n")
    print(f"{'backend':10s} {'device':6s} {'ms/frame':>9s} {'raw fps':>8s} {'eff. fps':>9s}")
    print("-" * 46)
    base = None
    for tag, dev, ms in rows:
        raw = 1000.0 / ms
        eff = raw * every
        if base is None:
            base = ms
        print(f"{tag:10s} {dev:6s} {ms:9.1f} {raw:8.1f} {eff:9.1f}")
    if rows:
        best = min(rows, key=lambda r: r[2])
        print(f"\nfastest: {best[0]} on {best[1]}  ({1000.0/best[2]:.1f} raw fps, "
              f"{1000.0/best[2]*every:.1f} effective) — {base/best[2]:.2f}x vs first row")


if __name__ == "__main__":
    main()
