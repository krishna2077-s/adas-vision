"""
main.py — Entry point for the ADAS Vision system.

Runs lane detection (Module 1) and object detection (Module 2) together on a
video file or live webcam feed.

Run:
    # Webcam, both modules
    python main.py --camera

    # Video file, both modules
    python main.py --video dashcam.mp4

    # Lanes only (no YOLO — useful if ultralytics isn't installed)
    python main.py --video dashcam.mp4 --no-objects

    # Objects only
    python main.py --video dashcam.mp4 --no-lanes

    # Debug overlay + save annotated output
    python main.py --video dashcam.mp4 --debug --save output.mp4

Controls while running:
    Q  — quit
    D  — toggle debug mode (ROI + raw lines overlay)
    P  — pause / resume
    S  — save a screenshot
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

import config as cfg
from decision_engine import DecisionEngine
from lane_detection import LaneDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(
    source,
    save_path: str = None,
    debug: bool = False,
    enable_lanes: bool = True,
    enable_objects: bool = True,
) -> None:
    """
    Main loop: reads frames, runs enabled modules, displays the combined result.

    Args:
        source:         Camera index (int) or path to video file (str).
        save_path:      If set, writes the annotated video here.
        debug:          Enables the lane debug overlay at startup.
        enable_lanes:   Run Module 1 (lane detection).
        enable_objects: Run Module 2 (object detection).
    """
    cfg.DEBUG_MODE = debug

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or cfg.TARGET_FPS
    logger.info(f"Source opened: {w}x{h} @ {fps_src:.1f} FPS")

    # ── Initialise modules ────────────────────────────────────────────────
    lane_detector = LaneDetector(frame_width=w, frame_height=h) if enable_lanes else None

    object_detector = None
    if enable_objects:
        try:
            from object_detection import ObjectDetector
            object_detector = ObjectDetector(frame_width=w, frame_height=h)
        except ImportError as exc:
            logger.warning(f"Object detection disabled: {exc}")
            logger.warning("Continuing with lane detection only.")
            enable_objects = False

    if not enable_lanes and not enable_objects:
        logger.error("Both modules are disabled — nothing to run.")
        sys.exit(1)

    # ── Module 3: decision engine (fuses whatever modules are enabled) ─────
    engine = DecisionEngine(frame_width=w, frame_height=h)

    # ── Optional writer ────────────────────────────────────────────────────
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))
        logger.info(f"Saving annotated output to: {save_path}")

    paused = False
    screenshot_idx = 0
    last_status = None
    window_name = "ADAS Vision  (Q quit | D debug | P pause | S screenshot)"

    logger.info("Starting. Press Q to quit.")

    annotated = None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
                    engine.reset()                        # fresh baseline on replay
                    last_status = None
                    continue
                logger.error("Camera read failed.")
                break

            annotated = frame
            lane_result = None
            obj_result = None
            lane_center_x = None

            # ── Module 1: lanes ───────────────────────────────────────
            if lane_detector is not None:
                lane_result, annotated = lane_detector.process(annotated)
                lane_center_x = lane_result.lane_center_x

            # ── Module 2: objects ─────────────────────────────────────
            if object_detector is not None:
                obj_result, annotated = object_detector.process(annotated, lane_center_x)

            # ── Module 3: fuse into one decision, then draw its HUD ────
            decision = engine.process(lane_result, obj_result)
            annotated = engine.draw_hud(annotated, decision)

            # ── Terminal log on decision change ───────────────────────
            status = (decision.longitudinal, decision.lateral, decision.rule_id)
            if status != last_status:
                logger.info(decision.reason)
                last_status = status

            if writer:
                writer.write(annotated)

        if annotated is not None:
            cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Quit.")
            break
        elif key == ord("d"):
            cfg.DEBUG_MODE = not cfg.DEBUG_MODE
            logger.info(f"Debug mode: {'ON' if cfg.DEBUG_MODE else 'OFF'}")
        elif key == ord("p"):
            paused = not paused
            logger.info("Paused." if paused else "Resumed.")
        elif key == ord("s"):
            fname = f"screenshot_{screenshot_idx:04d}.jpg"
            cv2.imwrite(fname, annotated)
            logger.info(f"Screenshot saved: {fname}")
            screenshot_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADAS Vision — lane + object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Webcam (both modules):
    python main.py --camera

  Video file (both modules):
    python main.py --video dashcam.mp4

  Lanes only:
    python main.py --video dashcam.mp4 --no-objects

  Debug + save:
    python main.py --video dashcam.mp4 --debug --save output.mp4
        """,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true", help="Use webcam as input.")
    src.add_argument("--video",  type=str, metavar="PATH", help="Path to a video file.")

    p.add_argument("--camera-index", type=int, default=cfg.CAMERA_INDEX,
                   help=f"Webcam device index (default: {cfg.CAMERA_INDEX}).")
    p.add_argument("--debug", action="store_true",
                   help="Show ROI outline and raw Hough lines.")
    p.add_argument("--save", type=str, metavar="PATH",
                   help="Save annotated output to this .mp4 file.")
    p.add_argument("--no-lanes",   action="store_true", help="Disable lane detection.")
    p.add_argument("--no-objects", action="store_true", help="Disable object detection.")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = args.camera_index if args.camera else args.video
    run(
        source=source,
        save_path=args.save,
        debug=args.debug,
        enable_lanes=not args.no_lanes,
        enable_objects=not args.no_objects,
    )
