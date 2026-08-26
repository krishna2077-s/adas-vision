"""
export_openvino.py — Phase 8: convert the road model to OpenVINO IR for the iGPU.

The ONNX path (export_onnx.py) already runs the drivable-area model faster than
PyTorch on the CPU. OpenVINO goes further on Intel hardware: it can run the SAME
network on the integrated GPU (the iGPU that otherwise sits idle), and it can
INT8-quantize the weights so even the CPU does less work — both without
retraining.

This script produces two IR models from the existing ONNX:

    <stem>_ov_fp16.xml/.bin   FP16 IR   — smaller, GPU-friendly, ~lossless
    <stem>_ov_int8.xml/.bin   INT8 IR   — post-training-quantised with NNCF,
                                          calibrated on real dashcam frames

It then reports argmax agreement of each IR (on CPU and, if present, GPU)
against the ONNX reference, so any accuracy regression from quantisation is
visible before you deploy it.

Usage:
    python export_openvino.py                 # FP16 + INT8 from the default ONNX
    python export_openvino.py --no-int8        # FP16 only (skip NNCF)
    python export_openvino.py --calib dashcam.mp4 --calib-frames 200

Requires:  pip install openvino nncf   (nncf only for --int8, the default).
Once the IR exists, set LEARNED_BACKEND="openvino" (config.py) to use it.
"""

import argparse
import logging
import os

import cv2
import numpy as np

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(frame) -> np.ndarray:
    """BGR frame -> (1,3,H,W) float32, identical to learned_road_detection._infer_mask."""
    img = cv2.resize(frame, (cfg.LEARNED_INPUT_W, cfg.LEARNED_INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


def calibration_frames(video_path: str, n: int):
    """Evenly sample n preprocessed frames from a video for INT8 calibration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"calibration video not found: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (n * 10)
    idxs = np.linspace(0, max(0, total - 1), num=n, dtype=int)
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if ok:
            frames.append(preprocess(f))
    cap.release()
    logger.info(f"calibration set: {len(frames)} frames from {video_path}")
    return frames


def stem_from_onnx(onnx_path: str) -> str:
    base = os.path.splitext(onnx_path)[0]
    return base[:-len("_768x432")] if base.endswith("_768x432") else base


def agreement(ref_mask, test_mask) -> float:
    return float((ref_mask == test_mask).mean()) * 100.0


def main():
    ap = argparse.ArgumentParser(description="Export the road model to OpenVINO IR (FP16 + INT8).")
    ap.add_argument("--onnx", default=cfg.LEARNED_ONNX_PATH, help="source ONNX model.")
    ap.add_argument("--calib", default="dashcam.mp4", help="video for INT8 calibration.")
    ap.add_argument("--calib-frames", type=int, default=150)
    ap.add_argument("--no-int8", action="store_true", help="export FP16 only (skip NNCF).")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        logger.error(f"ONNX '{args.onnx}' not found — run export_onnx.py first.")
        return

    try:
        import openvino as ov
    except ImportError:
        logger.error("openvino not installed:  pip install openvino")
        return

    core = ov.Core()
    logger.info(f"OpenVINO {ov.__version__}; devices: {core.available_devices}")

    # ── FP32 read + FP16 IR ────────────────────────────────────────────────
    model = core.read_model(args.onnx)
    stem = os.path.splitext(args.onnx)[0]                       # keep the 768x432 in the name
    fp16_xml = f"{stem}_ov_fp16.xml"
    ov.save_model(model, fp16_xml, compress_to_fp16=True)
    logger.info(f"saved FP16 IR -> {fp16_xml}")

    # ── reference outputs (ONNX via OpenVINO CPU, FP32) ────────────────────
    ref = core.compile_model(model, "CPU")
    ref_out_port = ref.output(0)
    sample = preprocess(_first_frame(args.calib))
    ref_mask = ref(sample)[ref_out_port].argmax(1).squeeze(0).astype(np.uint8)

    made = [("FP16", fp16_xml)]

    # ── INT8 IR via NNCF post-training quantisation ────────────────────────
    if not args.no_int8:
        try:
            import nncf
            frames = calibration_frames(args.calib, args.calib_frames)
            calib = nncf.Dataset(frames, lambda x: x)
            logger.info("running NNCF INT8 post-training quantisation ...")
            int8_model = nncf.quantize(model, calib, subset_size=len(frames))
            int8_xml = f"{stem}_ov_int8.xml"
            ov.save_model(int8_model, int8_xml)
            logger.info(f"saved INT8 IR -> {int8_xml}")
            made.append(("INT8", int8_xml))
        except ImportError:
            logger.warning("nncf not installed — skipping INT8 (pip install nncf).")
        except Exception as exc:
            logger.warning(f"INT8 quantisation failed ({exc}) — FP16 IR is still usable.")

    # ── parity report: each IR on each device vs the FP32 ONNX reference ───
    logger.info("── argmax agreement vs FP32 ONNX (higher = closer) ──")
    for tag, xml in made:
        m = core.read_model(xml)
        for dev in ("CPU", "GPU"):
            if dev not in core.available_devices:
                continue
            try:
                comp = core.compile_model(m, dev)
                out = comp(sample)[comp.output(0)].argmax(1).squeeze(0).astype(np.uint8)
                logger.info(f"  {tag:4s} on {dev:3s}: {agreement(ref_mask, out):6.2f}%")
            except Exception as exc:
                logger.warning(f"  {tag:4s} on {dev:3s}: failed ({exc})")

    logger.info("Done. Set LEARNED_BACKEND=\"openvino\" in config.py to use the IR.")


def _first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"could not read a frame from {video_path}")
    return f


if __name__ == "__main__":
    main()
