"""
driver_monitor_demo.py — Standalone demo for Module 9v2 (DMS).

Shows the full driver monitoring overlay in a dedicated window:
  • Face box + head-pose axes
  • Gaze zone label
  • EAR value + PERCLOS bar
  • Hands-on-wheel count
  • Phone detection flag
  • DMS state chip

Run:
    python driver_monitor_demo.py                # default webcam (index 0)
    python driver_monitor_demo.py --cam 1        # USB driver cam on index 1
    python driver_monitor_demo.py --video <path> # pre-recorded driver footage

Optional flags:
    --no-hands   Disable hands-on-wheel detection
    --no-phone   Disable phone detection
    --backend haar   Force Haar-cascade fallback (no mediapipe needed)

Press Q to quit.

Advisory / research only.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# ── make sure perception/ and core/ are on sys.path ────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
for _d in ("core", "perception"):
    _p = _ROOT / _d
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ADAS Vision — Driver Monitoring System standalone demo (Module 9v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cam",   type=int, default=0, metavar="INDEX",
                     help="Webcam device index (default: 0).")
    src.add_argument("--video", type=str, default=None, metavar="PATH",
                     help="Path to a pre-recorded driver-facing video file.")
    ap.add_argument("--no-hands", action="store_true", help="Disable hands-on-wheel detection.")
    ap.add_argument("--no-phone", action="store_true", help="Disable phone detection.")
    ap.add_argument("--backend", choices=["mediapipe", "haar"], default=None,
                    help="Force DMS backend (default: auto-select mediapipe, fall back to haar).")
    ap.add_argument("--save", type=str, default=None, metavar="PATH",
                    help="Save annotated demo output to this .mp4 file.")
    return ap.parse_args()


def main() -> None:
    args = _parse()

    # ── Apply optional overrides to config BEFORE importing DriverMonitor ─
    import config as cfg
    if args.no_hands:
        cfg.DMS_ENABLE_HANDS = False
    if args.no_phone:
        cfg.DMS_ENABLE_PHONE = False
    if args.backend:
        cfg.DMS_BACKEND = args.backend

    from driver_monitoring import DriverMonitor

    # ── Open video source ─────────────────────────────────────────────────
    source = args.video if args.video else args.cam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source!r}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    dm = DriverMonitor(frame_width=w, frame_height=h)
    if not dm.available:
        raise SystemExit(
            "No face detector available (Haar cascades missing from this OpenCV build). "
            "Install mediapipe or rebuild OpenCV with Haar support."
        )

    print(
        f"[DMS demo] source={source!r}  backend={dm._backend_name}  "
        f"hands={'on' if dm._hands else 'off'}  phone={'on' if dm._phone else 'off'}\n"
        "Press Q to quit."
    )

    # ── Optional writer ───────────────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_src, (w, h))
        print(f"[DMS demo] Saving to: {args.save}")

    window = "ADAS Vision — DMS Demo  (Q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if w > 960:
        cv2.resizeWindow(window, 960, int(h * 960 / w))

    # ── Rolling FPS ───────────────────────────────────────────────────────
    _times: list = []

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                dm.reset()
                continue
            break

        result, vis = dm.process(frame)

        # ── PERCLOS bar ───────────────────────────────────────────────
        bar_w = int(result.perclos * (w - 20))
        bar_col = (0, 0, 255) if result.perclos > cfg.DMS_PERCLOS_THRESH else (0, 200, 0)
        cv2.rectangle(vis, (10, h - 16), (10 + bar_w, h - 4), bar_col, -1)
        cv2.rectangle(vis, (10, h - 16), (w - 10, h - 4), (120, 120, 120), 1)
        cv2.putText(vis, f"PERCLOS {result.perclos * 100:.0f}%",
                    (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # ── FPS counter ───────────────────────────────────────────────
        _times.append(time.perf_counter())
        if len(_times) > 60:
            _times.pop(0)
        if len(_times) >= 2:
            fps_live = (len(_times) - 1) / (_times[-1] - _times[0])
            cv2.putText(vis, f"FPS {fps_live:.1f}", (w - 110, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        if writer:
            writer.write(vis)

        cv2.imshow(window, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
