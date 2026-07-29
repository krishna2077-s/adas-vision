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
from forward_collision_warning import ForwardCollisionWarning
from lane_detection import LaneDetector
from road_detection import RoadDetector
from tracker import MultiObjectTracker

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
    driver_cam: int = None,
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

    # Module 1b: classical road-surface fallback for unmarked roads (hills, rural).
    road_detector = (RoadDetector(frame_width=w, frame_height=h)
                     if enable_lanes and cfg.ENABLE_ROAD_FALLBACK else None)

    # Module 1c: learned drivable-area model — the PRIMARY unmarked-road source.
    # Falls back to Module 1b if the trained weights are absent or torch is
    # missing, so the system still runs on a clean checkout.
    learned_road_detector = None
    if enable_lanes and cfg.ENABLE_LEARNED_ROAD:
        try:
            from learned_road_detection import LearnedRoadDetector
            lrd = LearnedRoadDetector(frame_width=w, frame_height=h)
            learned_road_detector = lrd if lrd.available else None
        except ImportError as exc:
            logger.warning(f"Learned road detection unavailable ({exc}) — using classical fallback.")

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

    # ── Module 4: multi-object tracker (stable IDs; only if objects are on) ─
    tracker = MultiObjectTracker() if enable_objects else None

    # ── Module 3: decision engine (fuses whatever modules are enabled) ─────
    engine = DecisionEngine(frame_width=w, frame_height=h)

    # ── Module 10: Forward Collision Warning (driver-facing staged alert) ──
    # Consumes the engine's already-selected hazard; warns a human, brakes nothing.
    fcw = ForwardCollisionWarning(w, h) if cfg.ENABLE_FCW else None
    last_fcw_level = 0

    # ── Module 11: scene understanding (traffic-light state + traffic signs) ─
    # Light state reads RED/AMBER/GREEN out of the box; sign recognition is inert
    # until train_signs.py produces gtsrb_sign_cnn.pth. Both feed the advisory
    # planner (posted limit caps cruise; a red light eases toward a stop).
    traffic_light_reader = sign_recognizer = None
    try:
        if cfg.ENABLE_TRAFFIC_LIGHT:
            from traffic_light_state import TrafficLightReader
            traffic_light_reader = TrafficLightReader()
        if cfg.ENABLE_SIGN_RECOGNITION:
            from traffic_sign_recognition import SignRecognizer
            sr = SignRecognizer(w, h)
            sign_recognizer = sr if sr.available else None
    except Exception as exc:
        logger.warning(f"Scene understanding disabled ({exc}).")

    # ── Modules 5-9: advisory SIMULATION layers (Phase 9) ─────────────────
    # perception BEV -> sim fusion -> prediction/planning -> sim control, plus
    # driver monitoring. ALL advisory/simulated — never wired to a vehicle.
    bev_projector = fusion = planner = controller = driver_monitor = None
    driver_cap = None
    try:
        if cfg.ENABLE_BEV:
            from perception_bev import BEVProjector
            bev_projector = BEVProjector(w, h)
        if cfg.ENABLE_FUSION:
            from sensor_fusion import SimRadarFusion
            fusion = SimRadarFusion()
        if cfg.ENABLE_PLANNING:
            from prediction_planning import Planner
            planner = Planner(w, h)
        if cfg.ENABLE_CONTROL_SIM:
            from control_sim import SimController
            controller = SimController(w, h)
        if cfg.ENABLE_DRIVER_MON and driver_cam is not None:
            from driver_monitoring import DriverMonitor
            dm = DriverMonitor()
            if dm.available:
                cap_d = cv2.VideoCapture(driver_cam)
                if cap_d.isOpened():
                    driver_monitor, driver_cap = dm, cap_d
                else:
                    logger.warning(f"Driver cam {driver_cam} not opened — driver monitoring off.")
    except Exception as exc:
        logger.warning(f"Advisory-sim layers disabled ({exc}).")

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
                    if tracker is not None:
                        tracker.reset()
                    if road_detector is not None:
                        road_detector.reset()
                    if learned_road_detector is not None:
                        learned_road_detector.reset()
                    for m in (bev_projector, fusion, planner, controller, driver_monitor,
                              fcw, traffic_light_reader, sign_recognizer):
                        if m is not None:
                            m.reset()
                    last_status = None
                    last_fcw_level = 0
                    continue
                logger.error("Camera read failed.")
                break

            annotated = frame
            lane_result = None
            obj_result = None
            tracks = None
            lane_center_x = None

            # ── Module 1: lanes (paint) → 1c: learned road → 1b: classical ───
            if lane_detector is not None:
                lane_result, annotated = lane_detector.process(annotated)
                if lane_result.confidence == 0.0:
                    # No usable paint — follow the drivable road surface instead
                    # (estimated guidance for unmarked roads). Detect on the RAW
                    # frame; draw onto the annotated one. Prefer the learned
                    # model; drop to the classical detector if it has no lock.
                    got = False
                    if learned_road_detector is not None:
                        road_result, annotated = learned_road_detector.process(frame, draw_on=annotated)
                        if road_result.confidence > 0.0:
                            lane_result = road_result
                            got = True
                    if not got and road_detector is not None:
                        road_result, annotated = road_detector.process(frame, draw_on=annotated)
                        if road_result.confidence > 0.0:
                            lane_result = road_result
                lane_center_x = lane_result.lane_center_x

            # ── Module 2: objects ─────────────────────────────────────
            if object_detector is not None:
                obj_result, annotated = object_detector.process(annotated, lane_center_x)

            # ── Module 4: track objects across frames (stable IDs) ────
            if tracker is not None:
                tracks = tracker.update(obj_result.detections)
                annotated = tracker.annotate(annotated, tracks)

            # ── Module 11: scene understanding (lights + signs) ───────
            # Read from the RAW frame (clean colours), draw onto annotated,
            # and derive advisory context for the planner.
            speed_limit_mps = None
            red_light = False
            try:
                if traffic_light_reader is not None and obj_result is not None:
                    lights = traffic_light_reader.read(frame, obj_result.detections)
                    annotated = traffic_light_reader.draw(annotated, lights)
                    red_light = traffic_light_reader.controlling_state in ("RED", "AMBER")
                if sign_recognizer is not None:
                    sign_res = sign_recognizer.process(frame)
                    annotated = sign_recognizer.draw(annotated, sign_res)
                    if sign_res.speed_limit_kmh is not None:
                        speed_limit_mps = sign_res.speed_limit_kmh / 3.6
            except Exception as exc:
                logger.debug(f"scene understanding skipped: {exc}")

            # ── Module 3: fuse into one decision, then draw its HUD ────
            decision = engine.process(lane_result, tracks)
            annotated = engine.draw_hud(annotated, decision)

            # ── Modules 5-9: advisory SIMULATION overlays ─────────────
            # Bird's-eye perception -> sim fusion -> prediction/planning ->
            # sim control, + driver monitoring. Nothing here controls a car.
            try:
                if bev_projector is not None:
                    ego_objs = bev_projector.project(tracks or [])
                    plan = None
                    if fusion is not None:
                        fusion.process(ego_objs)
                    if planner is not None:
                        ego_spd = controller.sim_speed if controller is not None else None
                        plan = planner.process(ego_objs, lane_result, ego_spd,
                                               speed_limit_mps=speed_limit_mps,
                                               red_light=red_light)
                        annotated = planner.draw_overlay(annotated, plan)
                    if controller is not None:
                        cmd = controller.process(plan, lane_result)
                        annotated = controller.draw(annotated, cmd)
                    annotated = bev_projector.draw_panel(
                        annotated, ego_objs, plan.predictions if plan is not None else None)
                    if fusion is not None:
                        annotated = fusion.draw_label(annotated)
                if driver_monitor is not None and driver_cap is not None:
                    ok_d, dframe = driver_cap.read()
                    if ok_d:
                        dres, _ = driver_monitor.process(dframe)
                        annotated = driver_monitor.draw_chip(annotated, dres)
            except Exception as exc:
                logger.debug(f"advisory-sim overlay skipped: {exc}")

            # ── Module 10: Forward Collision Warning (drawn LAST — a ──
            # collision alert must never be occluded by another overlay).
            if fcw is not None:
                try:
                    fcw_state = fcw.process(decision, tracks)
                    annotated = fcw.draw(annotated, fcw_state, tracks)
                    if fcw_state.level >= 2 and fcw_state.level != last_fcw_level:
                        logger.warning(
                            f"FCW {fcw_state.name}: {fcw_state.label} #{fcw_state.hazard_id} "
                            f"TTC {fcw_state.ttc_s}s dist {fcw_state.distance_m}m")
                    last_fcw_level = fcw_state.level
                except Exception as exc:
                    logger.debug(f"FCW overlay skipped: {exc}")

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
    if driver_cap is not None:
        driver_cap.release()
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
    p.add_argument("--driver-cam", type=int, default=None, metavar="INDEX",
                   help="Driver-facing camera index for attention monitoring (Module 9).")

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
        driver_cam=args.driver_cam,
    )
