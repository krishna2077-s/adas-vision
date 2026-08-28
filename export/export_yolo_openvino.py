"""
export_yolo_openvino.py — Phase 11: export YOLOv8n to OpenVINO for the iGPU.

The evaluation harness (evaluate.py) showed object detection is ~77% of the
frame budget — the single biggest cost. Unlike CPU ONNX (which gave YOLOv8n no
speedup, being overhead-bound), running YOLO on the Intel integrated GPU via
OpenVINO is a real win here: measured ~2.9x over PyTorch-CPU on an i7-8650U +
UHD 620 (bench_openvino.py --yolo). This produces the IR ultralytics loads for
that path.

    python export_yolo_openvino.py                 # FP16 IR at the configured imgsz
    python export_yolo_openvino.py --int8 --data coco128.yaml   # optional INT8

Output: yolov8n_openvino_model/ (a directory ultralytics loads directly). Once
present, set YOLO_BACKEND="openvino" (config.py) to use it; it falls back to the
PyTorch model automatically if the IR or openvino is missing.
"""

import argparse
import logging

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Export YOLOv8n to OpenVINO IR.")
    ap.add_argument("--model", default=cfg.YOLO_MODEL, help="source .pt model.")
    ap.add_argument("--imgsz", type=int, default=cfg.YOLO_IMGSZ)
    ap.add_argument("--int8", action="store_true",
                    help="INT8 quantise (needs --data with a calibration set).")
    ap.add_argument("--data", default="coco128.yaml", help="calibration data for --int8.")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed:  pip install ultralytics")
        return

    kwargs = dict(format="openvino", imgsz=args.imgsz, verbose=False)
    if args.int8:
        kwargs.update(int8=True, data=args.data)
        logger.info(f"exporting {args.model} -> OpenVINO INT8 IR (imgsz={args.imgsz}, "
                    f"calib={args.data}) ...")
    else:
        logger.info(f"exporting {args.model} -> OpenVINO FP16 IR (imgsz={args.imgsz}) ...")

    out = YOLO(args.model).export(**kwargs)
    logger.info(f"done -> {out}")
    logger.info('Set YOLO_BACKEND="openvino" in config.py to run detection on the iGPU.')


if __name__ == "__main__":
    main()
