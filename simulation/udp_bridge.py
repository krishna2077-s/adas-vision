"""
udp_bridge.py — ADAS Vision UDP Co-simulation Bridge

Runs the full ADAS perception pipeline on a video source and continuously
streams perception data (detections, lane, road, decisions, FPS) as JSON
packets over UDP port 5005 to a connected MATLAB co-simulation client.

Also listens on UDP port 5006 for ego-vehicle position feedback from MATLAB
and overlays it on the video window.

Usage:
    python udp_bridge.py --video dashcam.mp4 --async
    python udp_bridge.py --camera

Protocol (outbound, port 5005):
    One JSON object per processed frame, newline-terminated:
    {
        "frame_id":    <int>,
        "timestamp_s": <float>,
        "fps":         <float>,
        "decision": {
            "longitudinal": <str>,   # "PROCEED" | "CAUTION" | "SLOW" | "BRAKE" | "EMERGENCY_STOP"
            "lateral":      <str>,   # "STRAIGHT" | "SLIGHT LEFT" | ...
            "rule_id":      <str>,
            "brake_demand": <float>, # 0.0 – 1.0
            "reason":       <str>
        },
        "lane": {
            "offset_px":     <int>,
            "confidence":    <float>,
            "steering":      <str>,
            "left_lane":     <bool>,
            "right_lane":    <bool>
        },
        "tracks": [
            {
                "track_id":   <int>,
                "label":      <str>,
                "confidence": <float>,
                "distance_m": <float | null>,
                "ttc_s":      <float | null>,
                "in_path":    <bool>,
                "risk":       <str>,   # "HIGH" | "MEDIUM" | "LOW"
                "bbox":       [x1, y1, x2, y2]
            }, ...
        ],
        "ego_from_matlab": <[x, y] | null>
    }

Protocol (inbound, port 5006):
    JSON from MATLAB:  {"ego_x": <float>, "ego_y": <float>}
"""

import argparse
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import cv2

import config as cfg
from decision_engine import DecisionEngine
from forward_collision_warning import ForwardCollisionWarning
from frame_guard import is_valid_frame
from lane_detection import LaneDetector
from road_detection import RoadDetector
from tracker import MultiObjectTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UDP constants
# ---------------------------------------------------------------------------
UDP_OUT_HOST = "127.0.0.1"
UDP_OUT_PORT = 5005          # Python -> MATLAB
UDP_IN_PORT  = 5006          # MATLAB -> Python (ego position feedback)
UDP_MAX_PKT  = 65000         # max UDP payload (bytes)


# ---------------------------------------------------------------------------
# Ego-position listener (runs in a daemon thread)
# ---------------------------------------------------------------------------
class EgoListener:
    """Listens for MATLAB ego-vehicle position updates on UDP port 5006."""

    def __init__(self, port: int = UDP_IN_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.05)
        self._pos = None          # latest (x, y) from MATLAB
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"EgoListener started — receiving MATLAB feedback on UDP {port}.")

    def _loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                obj = json.loads(data.decode("utf-8"))
                with self._lock:
                    self._pos = (float(obj.get("ego_x", 0.0)),
                                 float(obj.get("ego_y", 0.0)))
            except socket.timeout:
                pass
            except Exception as exc:
                logger.debug(f"EgoListener error: {exc}")

    def latest(self):
        with self._lock:
            return self._pos

    def stop(self):
        self._running = False
        self._sock.close()


# ---------------------------------------------------------------------------
# Helper: serialise a track list
# ---------------------------------------------------------------------------
def _serialise_tracks(tracks):
    out = []
    if not tracks:
        return out
    for t in tracks:
        out.append({
            "track_id":   int(t.id),
            "label":      str(t.label),
            "confidence": round(float(t.confidence), 3),
            "distance_m": round(float(t.smoothed_distance_m), 2) if t.smoothed_distance_m is not None else None,
            "ttc_s":      round(float(t.ttc_s), 2)      if t.ttc_s      is not None else None,
            "in_path":    bool(t.in_path),
            "risk":       str(getattr(t, "risk", "LOW")),
            "bbox":       [int(t.bbox[0]), int(t.bbox[1]),
                           int(t.bbox[2]), int(t.bbox[3])],
        })
    return out


# ---------------------------------------------------------------------------
# Helper: serialise a lane result
# ---------------------------------------------------------------------------
def _serialise_lane(lane_result):
    if lane_result is None:
        return {"offset_px": 0, "confidence": 0.0,
                "steering": "UNKNOWN", "left_lane": False, "right_lane": False}
    return {
        "offset_px":  int(lane_result.offset_px),
        "confidence": round(float(lane_result.confidence), 2),
        "steering":   str(lane_result.steering),
        "left_lane":  lane_result.left_lane  is not None,
        "right_lane": lane_result.right_lane is not None,
    }


# ---------------------------------------------------------------------------
# Helper: serialise a decision
# ---------------------------------------------------------------------------
def _serialise_decision(decision):
    return {
        "longitudinal": str(decision.longitudinal),
        "lateral":      str(decision.lateral),
        "rule_id":      str(decision.rule_id),
        "brake_demand": round(float(getattr(decision, "brake_demand", 0.0)), 3),
        "reason":       str(decision.reason),
    }


# ---------------------------------------------------------------------------
# Main bridge loop
# ---------------------------------------------------------------------------
def run_bridge(source, async_det: bool = False) -> None:
    if async_det:
        cfg.ENABLE_ASYNC_DETECTION = True

    # ── Open video source ───────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or cfg.TARGET_FPS
    logger.info(f"Source opened: {w}x{h} @ {fps_src:.1f} FPS")

    # ── Perception modules ──────────────────────────────────────────────
    lane_detector = LaneDetector(frame_width=w, frame_height=h)

    road_detector = (RoadDetector(frame_width=w, frame_height=h)
                     if cfg.ENABLE_ROAD_FALLBACK else None)

    learned_road_detector = None
    if cfg.ENABLE_LEARNED_ROAD:
        try:
            from learned_road_detection import LearnedRoadDetector
            lrd = LearnedRoadDetector(frame_width=w, frame_height=h)
            learned_road_detector = lrd if lrd.available else None
        except ImportError as exc:
            logger.warning(f"Learned road detection unavailable ({exc}).")

    object_detector = None
    async_detector  = None
    try:
        if cfg.ENABLE_ASYNC_DETECTION:
            from async_detector import AsyncObjectDetector
            async_detector = AsyncObjectDetector(w, h)
            logger.info("Object detection: ASYNC worker thread.")
        else:
            from object_detection import ObjectDetector
            object_detector = ObjectDetector(frame_width=w, frame_height=h)
    except ImportError as exc:
        logger.warning(f"Object detection disabled: {exc}")

    tracker = MultiObjectTracker()
    engine  = DecisionEngine(frame_width=w, frame_height=h)
    fcw     = ForwardCollisionWarning(w, h) if cfg.ENABLE_FCW else None

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

    # ── UDP sockets ─────────────────────────────────────────────────────
    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ego_listener = EgoListener(UDP_IN_PORT)

    logger.info(f"UDP bridge streaming on {UDP_OUT_HOST}:{UDP_OUT_PORT}  "
                f"| ego feedback on :{UDP_IN_PORT}")
    logger.info("Starting. Press Q to quit.")

    # ── Main loop ───────────────────────────────────────────────────────
    frame_no   = 0
    bad_frames = 0
    last_status = None
    last_fcw_level = 0
    start_time = time.monotonic()
    window_name = "ADAS UDP Bridge  (Q quit)"

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                engine.reset()
                tracker.reset()
                if road_detector:         road_detector.reset()
                if learned_road_detector: learned_road_detector.reset()
                if async_detector:        async_detector.reset()
                frame_no = 0
                start_time = time.monotonic()
                continue
            logger.error("Camera read failed.")
            break

        if not is_valid_frame(frame):
            bad_frames += 1
            if bad_frames > 300:
                logger.error("Too many consecutive malformed frames — stopping.")
                break
            continue
        bad_frames = 0

        annotated = frame.copy()
        lane_result = obj_result = tracks = None
        lane_center_x = None
        detection_age = 0.0
        is_new = True

        # ── Module 1: lane detection ────────────────────────────────────
        lane_result, annotated = lane_detector.process(annotated)
        if lane_result.confidence == 0.0:
            got = False
            if learned_road_detector is not None:
                road_res, annotated = learned_road_detector.process(frame, draw_on=annotated)
                if road_res.confidence > 0.0:
                    lane_result, got = road_res, True
            if not got and road_detector is not None:
                road_res, annotated = road_detector.process(frame, draw_on=annotated)
                if road_res.confidence > 0.0:
                    lane_result = road_res
        lane_center_x = lane_result.lane_center_x

        # ── Module 2: object detection ──────────────────────────────────
        if object_detector is not None:
            obj_result, annotated = object_detector.process(annotated, lane_center_x)
        elif async_detector is not None:
            async_detector.submit(frame, lane_center_x)
            obj_result, det_age, is_new = async_detector.latest()
            detection_age = det_age or 0.0
            if obj_result is not None:
                annotated = async_detector.annotate(annotated, obj_result)

        # ── Module 4: tracker ───────────────────────────────────────────
        if async_detector is not None:
            dets = obj_result.detections if (obj_result and is_new) else []
        else:
            dets = obj_result.detections if obj_result else []
        tracks = tracker.update(dets)
        annotated = tracker.annotate(annotated, tracks)

        # ── Module 11: traffic lights + signs ───────────────────────────
        try:
            if traffic_light_reader and obj_result:
                lights = traffic_light_reader.read(frame, obj_result.detections)
                annotated = traffic_light_reader.draw(annotated, lights)
            if sign_recognizer:
                sign_res = sign_recognizer.process(frame)
                annotated = sign_recognizer.draw(annotated, sign_res)
        except Exception:
            pass

        # ── Module 3: decision engine ───────────────────────────────────
        decision = engine.process(lane_result, tracks, detection_age_s=detection_age)
        annotated = engine.draw_hud(annotated, decision)

        # ── Module 10: FCW ──────────────────────────────────────────────
        if fcw:
            try:
                fcw_state = fcw.process(decision, tracks)
                annotated = fcw.draw(annotated, fcw_state, tracks)
                last_fcw_level = fcw_state.level
            except Exception:
                pass

        # ── Log decision changes ────────────────────────────────────────
        status = (decision.longitudinal, decision.lateral, decision.rule_id)
        if status != last_status:
            logger.info(decision.reason)
            last_status = status

        # ── Compute FPS ─────────────────────────────────────────────────
        elapsed = time.monotonic() - start_time
        fps_live = frame_no / elapsed if elapsed > 0 else 0.0

        # ── Overlay ego vehicle position from MATLAB ────────────────────
        ego_pos = ego_listener.latest()
        if ego_pos is not None:
            cv2.putText(annotated,
                        f"EGO MATLAB: ({ego_pos[0]:.1f}, {ego_pos[1]:.1f}) m",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 180, 0), 2, cv2.LINE_AA)

        # ── Build and send UDP packet ───────────────────────────────────
        packet = {
            "frame_id":    frame_no,
            "timestamp_s": round(elapsed, 4),
            "fps":         round(fps_live, 1),
            "decision":    _serialise_decision(decision),
            "lane":        _serialise_lane(lane_result),
            "tracks":      _serialise_tracks(tracks),
            "ego_from_matlab": list(ego_pos) if ego_pos else None,
        }
        try:
            payload = json.dumps(packet).encode("utf-8")
            if len(payload) <= UDP_MAX_PKT:
                out_sock.sendto(payload, (UDP_OUT_HOST, UDP_OUT_PORT))
            else:
                # Trim tracks list if packet is too large
                packet["tracks"] = packet["tracks"][:10]
                out_sock.sendto(json.dumps(packet).encode("utf-8"),
                                (UDP_OUT_HOST, UDP_OUT_PORT))
        except Exception as exc:
            logger.debug(f"UDP send error: {exc}")

        cv2.imshow(window_name, annotated)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            logger.info("Quit.")
            break

        frame_no += 1

    # ── Cleanup ─────────────────────────────────────────────────────────
    cap.release()
    out_sock.close()
    ego_listener.stop()
    if async_detector:
        async_detector.stop()
    cv2.destroyAllWindows()
    logger.info(f"Bridge stopped. Processed {frame_no} frames.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="ADAS Vision UDP Co-simulation Bridge",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=str, metavar="PATH",
                     help="Path to a video file.")
    src.add_argument("--camera", action="store_true",
                     help="Use webcam as input.")
    p.add_argument("--camera-index", type=int, default=cfg.CAMERA_INDEX,
                   help=f"Webcam device index (default: {cfg.CAMERA_INDEX}).")
    p.add_argument("--async", dest="async_det", action="store_true",
                   help="Run object detection on a background thread.")
    p.add_argument("--out-port", type=int, default=UDP_OUT_PORT,
                   help=f"UDP port to stream perception data (default: {UDP_OUT_PORT}).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = args.camera_index if args.camera else args.video
    run_bridge(source, async_det=args.async_det)
