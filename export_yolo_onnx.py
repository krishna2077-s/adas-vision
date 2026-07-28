"""
export_yolo_onnx.py — export YOLOv8n (Module 2) to ONNX for faster CPU inference.

Ultralytics runs an exported .onnx through ONNX Runtime — typically ~1.5-2x
faster than the PyTorch build on CPU, with the same detections. Run this once:

    python export_yolo_onnx.py

Then keep YOLO_BACKEND="onnx" in config.py (the default) and Module 2 uses it
automatically. If the .onnx is ever missing, the app falls back to the .pt.

Advisory / research only.
"""

import os

import config as cfg


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed — run: pip install ultralytics")

    src = cfg.YOLO_MODEL
    print(f"[yolo-export] loading {src}")
    model = YOLO(src)
    print("[yolo-export] exporting to ONNX @ imgsz 640 (this also runs a self-check) ...")
    out = model.export(format="onnx", imgsz=640)      # returns the output path
    size_mb = os.path.getsize(out) / 1e6
    print(f"[yolo-export] done -> {out}  ({size_mb:.1f} MB)")
    print(f"[yolo-export] keep YOLO_BACKEND='onnx' in config.py to use it "
          f"(expected file: {cfg.YOLO_ONNX_MODEL})")
    if os.path.basename(out) != cfg.YOLO_ONNX_MODEL:
        print(f"[yolo-export] NOTE: config expects '{cfg.YOLO_ONNX_MODEL}' — "
              f"rename '{out}' or update YOLO_ONNX_MODEL if they differ.")


if __name__ == "__main__":
    main()
