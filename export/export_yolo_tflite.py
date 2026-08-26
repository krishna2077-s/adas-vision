"""
export_yolo_tflite.py — edge feasibility: YOLOv8n -> per-tensor INT8 TFLite.

Motivation: an Axis camera SoC (ARTPEC-8) can run detection on-device via its
DLPU + the larod runtime, which needs a *per-tensor* INT8 TFLite model. This
script produces exactly that, calibrated on real dashcam frames, so the edge
path is reproducible.

What it does (all runs on Windows/macOS/Linux):
  1. YOLOv8n -> ONNX          (Ultralytics; not platform-gated)
  2. build a calibration set  (N real frames from the dashcam clip, NHWC [0,1])
  3. ONNX -> per-tensor INT8 TFLite via onnx2tf, dashcam-calibrated

Why onnx2tf and not `YOLO.export(format="tflite")`: Ultralytics gates its
TFLite/LiteRT export to Linux x86 / macOS, so it refuses to run on Windows.
onnx2tf does the same conversion cross-platform and lets us force per-tensor
quantisation (the ARTPEC-8 requirement).

HONEST CAVEAT — accuracy verification is a separate, on-target step. The
resulting INT8 model is a valid per-tensor full-INT8 TFLite (verified: the FP32
TFLite matches the torch model 100% on detections, so the graph translation is
faithful). But desktop TFLite CPU kernels can't *execute* the INT8 graph (the
reference LOGISTIC kernel demands output scale == 1/256, which onnx2tf doesn't
set, and XNNPACK won't delegate the inserted transposes). Neither limitation
applies to the ARTPEC-8 DLPU (larod has its own kernels). Measure the INT8
detection accuracy on Linux/Colab (Ultralytics `YOLO(tflite).val(...)`) or on
the device itself — not on a Windows desktop.

Requires the optional edge toolchain (kept OUT of requirements.txt to keep the
core lean):  pip install tensorflow onnx2tf tf_keras

    python export_yolo_tflite.py
    python export_yolo_tflite.py --calib-frames 128 --imgsz 640
"""

import argparse
import glob
import os

import cv2
import numpy as np

import config as cfg


def build_calibration(video: str, n: int, imgsz: int, out_npy: str) -> bool:
    """Sample N frames spread across the clip -> (N, imgsz, imgsz, 3) float32 in [0,1]."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[calib] cannot open {video} — need a clip for representative calibration.")
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (n * 10)
    idxs = np.linspace(max(0, total // 20), max(1, total - total // 20), n).astype(int)
    buf = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if ok:
            img = cv2.cvtColor(cv2.resize(f, (imgsz, imgsz)), cv2.COLOR_BGR2RGB)
            buf.append(img.astype(np.float32) / 255.0)
    cap.release()
    if not buf:
        print("[calib] no frames read.")
        return False
    np.save(out_npy, np.stack(buf).astype(np.float32))
    print(f"[calib] {len(buf)} frames -> {out_npy}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Export YOLOv8n to per-tensor INT8 TFLite for ARTPEC-8.")
    ap.add_argument("--model", default=cfg.YOLO_MODEL)
    ap.add_argument("--imgsz", type=int, default=cfg.YOLO_IMGSZ)
    ap.add_argument("--calib-video", default="dashcam.mp4")
    ap.add_argument("--calib-frames", type=int, default=64)
    ap.add_argument("--out-dir", default="yolov8n_tflite")
    args = ap.parse_args()

    try:
        import onnx2tf  # noqa: F401
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"[deps] edge toolchain missing ({exc}).\n"
                         f"       pip install tensorflow onnx2tf tf_keras")

    # 1. ONNX (not platform-gated)
    onnx_path = os.path.splitext(args.model)[0] + ".onnx"
    if not os.path.exists(onnx_path):
        print(f"[onnx] exporting {args.model} -> {onnx_path} ...")
        YOLO(args.model).export(format="onnx", imgsz=args.imgsz, opset=13, simplify=True, verbose=False)
    print(f"[onnx] ready: {onnx_path}")

    # 2. calibration set from real frames
    calib = "calib_data.npy"
    if not build_calibration(args.calib_video, args.calib_frames, args.imgsz, calib):
        raise SystemExit("[calib] failed — supply --calib-video with a real clip.")

    # 3. ONNX -> per-tensor INT8 TFLite
    import onnx2tf
    print("[onnx2tf] converting -> per-tensor INT8 TFLite (ARTPEC-8 / larod format) ...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=args.out_dir,
        output_integer_quantized_tflite=True,
        quant_type="per-tensor",                 # ARTPEC-8 DLPU requirement
        custom_input_op_name_np_data_path=[[
            "images", calib,
            np.array([[[[0.0, 0.0, 0.0]]]], dtype=np.float32),   # data already [0,1]
            np.array([[[[1.0, 1.0, 1.0]]]], dtype=np.float32),
        ]],
        non_verbose=True,
    )

    print("\n[done] TFLite artifacts:")
    for f in sorted(glob.glob(os.path.join(args.out_dir, "*.tflite"))):
        print(f"  {os.path.getsize(f)/1e6:5.1f} MB  {f}")
    full = os.path.join(args.out_dir, "yolov8n_full_integer_quant.tflite")
    if os.path.exists(full):
        print(f"\n[deploy] ARTPEC-8 target: {full}")
    print("[verify] measure INT8 accuracy on Linux/Colab or on-device (see this file's docstring).")


if __name__ == "__main__":
    main()
