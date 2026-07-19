"""
main.py — Entry point for the ADAS Vision lane detection system.

Run:
    # Webcam
    python main.py --camera

    # Video file
    python main.py --video path/to/dashcam.mp4

    # Video file with debug overlay (shows ROI + raw Hough lines)
    python main.py --video path/to/dashcam.mp4 --debug

    # Save annotated output to a file
    python main.py --video path/to/dashcam.mp4 --save output.mp4

Controls while running:
    Q  — quit
    D  — toggle debug mode (ROI + raw lines overlay)
    P  — pause / resume
    S  — save a screenshot
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

import config as cfg
from lane_detection import LaneDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(
    source,            # int (camera index) or str (video path)
    save_path: str = None,
    debug: bool = False,
) -> None:
    """
    Main loop: reads frames, runs lane detection, displays result.

    Args:
        source:    Camera index (int) or path to video file (str).
        save_path: If set, writes annotated video to this path.
        debug:     Enables debug overlay at startup.
    """
    cfg.DEBUG_MODE = debug

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    # Read actual frame size from the capture
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or cfg.TARGET_FPS

    logger.info(f"Source opened: {w}x{h} @ {fps_src:.1f} FPS")

    detector = LaneDetector(frame_width=w, frame_height=h)

    # Optional video writer
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))
        logger.info(f"Saving annotated output to: {save_path}")

    paused = False
    screenshot_idx = 0
    window_name = "ADAS Vision — Lane Detection  (Q quit | D debug | P pause | S screenshot)"

    logger.info("Starting. Press Q to quit.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # End of video file — loop back to start
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.error("Camera read failed.")
                    break

            result, annotated = detector.process(frame)

            # Print steering decision to terminal only when it changes
            if not hasattr(run, "_last_steering") or run._last_steering != result.steering:
                logger.info(
                    f"Steering: {result.steering:<25} | "
                    f"Offset: {result.offset_px:+4d}px | "
                    f"Conf: {result.confidence:.0%} | "
                    f"FPS: {result.fps:.1f}"
                )
                run._last_steering = result.steering

            if writer:
                writer.write(annotated)

        cv2.imshow(window_name, annotated if not paused else annotated)

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
        description="ADAS Vision — Real-time lane detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Webcam:
    python main.py --camera

  Video file:
    python main.py --video dashcam.mp4

  Video with debug overlay + save output:
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

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = args.camera_index if args.camera else args.video
    run(source=source, save_path=args.save, debug=args.debug)
