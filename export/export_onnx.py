"""
export_onnx.py — convert the trained PyTorch weights to a single ONNX file.

ONNX Runtime runs the SAME weights noticeably faster than PyTorch eager on CPU,
with identical output. Run this ONCE after training, then keep
LEARNED_BACKEND="onnx" in config.py (already the default) and the app uses the
fast path automatically.

Run:
    python export_onnx.py                                   # uses config.py paths
    python export_onnx.py --weights drivable_idd_lraspp_aug_best.pth
    python export_onnx.py --w 512 --h 288 --out fast_512.onnx   # a faster low-res build

Advisory / research only — never wire any of this to a vehicle's controls.
"""

import argparse
import os

import numpy as np
import torch
from torch import nn
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)

import config as cfg


def build_arch(arch: str) -> nn.Module:
    arch = arch.lower()
    if arch == "deeplabv3":
        return deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None,
                                            num_classes=2, aux_loss=True)
    if arch == "lraspp":
        m = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
        m.classifier.low_classifier = nn.Conv2d(40, 2, 1)
        m.classifier.high_classifier = nn.Conv2d(128, 2, 1)
        return m
    raise ValueError(f"unknown arch '{arch}' (use 'lraspp' or 'deeplabv3')")


def main():
    ap = argparse.ArgumentParser(description="Export trained weights to ONNX")
    ap.add_argument("--weights", default=cfg.LEARNED_MODEL_PATH)
    ap.add_argument("--out",     default=cfg.LEARNED_ONNX_PATH)
    ap.add_argument("--arch",    default=cfg.LEARNED_ARCH)
    ap.add_argument("--w", type=int, default=cfg.LEARNED_INPUT_W)
    ap.add_argument("--h", type=int, default=cfg.LEARNED_INPUT_H)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(f"[export] weights not found: {args.weights}")

    print(f"[export] building {args.arch} and loading {args.weights}")
    model = build_arch(args.arch)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    if args.arch.lower() == "deeplabv3" and getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier = None            # training-only head
    model.eval()

    dummy = torch.randn(1, 3, args.h, args.w)
    print(f"[export] exporting @ {args.w}x{args.h} -> {args.out}")
    try:                                       # dynamo=False -> one self-contained .onnx
        torch.onnx.export(model, dummy, args.out, input_names=["image"],
                          output_names=["out"], opset_version=12, dynamo=False)
    except TypeError:                          # older torch without the dynamo kwarg
        torch.onnx.export(model, dummy, args.out, input_names=["image"],
                          output_names=["out"], opset_version=12)

    # ── parity check: torch vs onnxruntime must agree on the segmentation ──
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, args.h, args.w).astype(np.float32)
    with torch.no_grad():
        t_pred = model(torch.from_numpy(x))["out"].numpy().argmax(1)
    o_pred = sess.run(None, {sess.get_inputs()[0].name: x})[0].argmax(1)
    agree = float((t_pred == o_pred).mean()) * 100.0
    size_mb = os.path.getsize(args.out) / 1e6

    print(f"[export] done — {size_mb:.1f} MB | torch/onnx pixel agreement {agree:.2f}%")
    if agree < 99.0:
        print("[export] WARNING: <99% agreement — inspect before trusting the ONNX path.")
    else:
        print("[export] parity OK — set LEARNED_BACKEND='onnx' (default) to use it.")


if __name__ == "__main__":
    main()
