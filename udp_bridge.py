"""
UDP Bridge for ADAS Vision
==========================

This script acts as a UDP bridge, wrapping the existing ADAS Vision perception pipeline
and streaming its outputs as JSON over UDP to a receiver (e.g., MATLAB/Simulink).
It serves as the critical integration component for hackathon applications.

Usage:
    python udp_bridge.py --video path/to/video.mp4 [options]
    python udp_bridge.py --camera [options]

Options:
    --video PATH        Path to the input video file.
    --camera            Use the primary camera as input.
    --udp-host HOST     Destination UDP host (default: 127.0.0.1).
    --udp-port PORT     Destination UDP port (default: 5005).
    --listen-port PORT  Local UDP port to listen for ego pose updates (default: 5006).
    --no-display        Run in headless mode, disabling cv2.imshow.
    --async             Use the AsyncObjectDetector instead of the standard ObjectDetector.
    --log               Enable drive logging (same as main.py).
"""

import cv2
import json
import time
import socket
import argparse
import threading
import math
from typing import Optional, Dict, Any

# Required imports from existing pipeline
from lane_detection import LaneDetector
from road_detection import RoadDetector
from tracker import MultiObjectTracker
from decision_engine import DecisionEngine

try:
    from forward_collision_warning import ForwardCollisionWarning
except ImportError:
    ForwardCollisionWarning = None

from frame_guard import is_valid_frame

# Conditional imports
try:
    from learned_road_detection import LearnedRoadDetector
except ImportError:
    LearnedRoadDetector = None

try:
    from object_detection import ObjectDetector
except ImportError:
    ObjectDetector = None

try:
    from async_detector import AsyncObjectDetector
except ImportError:
    AsyncObjectDetector = None

try:
    from traffic_light_state import TrafficLightReader
except ImportError:
    TrafficLightReader = None

try:
    from traffic_sign_recognition import SignRecognizer
except ImportError:
    SignRecognizer = None


import config as cfg
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def clean_floats(obj: Any) -> Any:
    """Recursively clean floats (NaN, Inf) and numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif type(obj).__module__ == 'numpy':
        # Convert numpy types to native Python types
        if hasattr(obj, 'tolist'):
            return clean_floats(obj.tolist())
        elif hasattr(obj, 'item'):
            return clean_floats(obj.item())
    return obj


class EgoPoseReceiver:
    """Non-blocking UDP socket on a daemon thread for the ego pose receiver."""
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.setblocking(False)
        self.running = True
        self.latest_pose: Dict[str, float] = {
            "ego_x": 0.0,
            "ego_y": 0.0,
            "ego_yaw_rad": 0.0,
            "ego_speed_mps": 0.0
        }
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                if data:
                    pose = json.loads(data.decode('utf-8'))
                    self.latest_pose.update(pose)
            except BlockingIOError:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error receiving ego pose: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description="ADAS Vision UDP Bridge")
    parser.add_argument('--video', type=str, help="Path to video file")
    parser.add_argument('--camera', action='store_true', help="Use camera")
    parser.add_argument('--camera-index', type=int, default=cfg.CAMERA_INDEX,
                        help="Webcam device index")
    parser.add_argument('--udp-host', type=str, default='127.0.0.1',
                        help="UDP destination host (default: 127.0.0.1)")
    parser.add_argument('--udp-port', type=int, default=5005,
                        help="UDP destination port (default: 5005)")
    parser.add_argument('--listen-port', type=int, default=5006,
                        help="UDP port for ego pose from Simulink (default: 5006)")
    parser.add_argument('--no-display', action='store_true',
                        help="Disable cv2.imshow (headless mode)")
    parser.add_argument('--async', dest='async_mode', action='store_true',
                        help="Run object detection on background thread (Module 14)")
    parser.add_argument('--log', type=str, default=None, metavar='PATH',
                        help="Record drive log (.jsonl)")

    args = parser.parse_args()

    if not args.video and not args.camera:
        print("Error: Must specify --video PATH or --camera")
        return

    source = args.camera_index if args.camera else args.video

    # ── Open video source ──────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or cfg.TARGET_FPS
    logger.info(f"Source opened: {w}x{h} @ {fps_src:.1f} FPS")

    # ── Initialise modules (mirrors main.py exactly) ───────────────────
    lane_detector = LaneDetector(frame_width=w, frame_height=h)

    road_detector = (RoadDetector(frame_width=w, frame_height=h)
                     if cfg.ENABLE_ROAD_FALLBACK else None)

    learned_road_detector = None
    if cfg.ENABLE_LEARNED_ROAD and LearnedRoadDetector is not None:
        try:
            lrd = LearnedRoadDetector(frame_width=w, frame_height=h)
            learned_road_detector = lrd if lrd.available else None
        except Exception as exc:
            logger.warning(f"Learned road detection unavailable ({exc})")

    object_detector = None
    async_detector = None
    if args.async_mode and AsyncObjectDetector is not None:
        async_detector = AsyncObjectDetector(w, h)
        logger.info("Object detection: ASYNC (Module 14)")
    elif ObjectDetector is not None:
        object_detector = ObjectDetector(frame_width=w, frame_height=h)
        logger.info("Object detection: SYNC")

    tracker = MultiObjectTracker()
    engine = DecisionEngine(frame_width=w, frame_height=h)
    fcw = ForwardCollisionWarning(w, h) if ForwardCollisionWarning else None

    traffic_light_reader = None
    sign_recognizer = None
    try:
        if cfg.ENABLE_TRAFFIC_LIGHT and TrafficLightReader is not None:
            traffic_light_reader = TrafficLightReader()
        if cfg.ENABLE_SIGN_RECOGNITION and SignRecognizer is not None:
            sr = SignRecognizer(w, h)
            sign_recognizer = sr if sr.available else None
    except Exception as exc:
        logger.warning(f"Scene understanding disabled ({exc})")

    # ── Drive logger (opt-in) ──────────────────────────────────────────
    drive_logger = None
    if args.log:
        try:
            from drive_logger import DriveLogger
            drive_logger = DriveLogger(args.log, source, w, h, fps_src)
        except Exception as exc:
            logger.warning(f"Drive logger disabled ({exc})")

    # ── UDP setup ──────────────────────────────────────────────────────
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_addr = (args.udp_host, args.udp_port)
    ego_receiver = EgoPoseReceiver(args.listen_port)
    logger.info(f"UDP Bridge: sending to {args.udp_host}:{args.udp_port}, "
                f"listening for ego pose on port {args.listen_port}")

    # ── Main loop ──────────────────────────────────────────────────────
    frame_no = 0
    last_status = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    engine.reset()
                    tracker.reset()
                    if road_detector:
                        road_detector.reset()
                    if learned_road_detector:
                        learned_road_detector.reset()
                    if async_detector:
                        async_detector.reset()
                    frame_no = 0
                    continue
                break

            if not is_valid_frame(frame):
                continue

            t0 = time.time()
            annotated = frame
            lane_result = None
            obj_result = None
            tracks = None
            lane_center_x = None
            detection_age = 0.0
            is_new = True

            # ── Module 1: lanes → 1c: learned road → 1b: classical ────
            lane_result, annotated = lane_detector.process(annotated)
            road_source = "PAINTED_LANE"
            if lane_result.confidence == 0.0:
                got = False
                if learned_road_detector is not None:
                    road_result, annotated = learned_road_detector.process(
                        frame, draw_on=annotated)
                    if road_result.confidence > 0.0:
                        lane_result = road_result
                        road_source = "LEARNED_ROAD"
                        got = True
                if not got and road_detector is not None:
                    road_result, annotated = road_detector.process(
                        frame, draw_on=annotated)
                    if road_result.confidence > 0.0:
                        lane_result = road_result
                        road_source = "CLASSICAL_ROAD"
            lane_center_x = lane_result.lane_center_x

            # ── Module 2: objects (sync or async) ──────────────────────
            if object_detector is not None:
                obj_result, annotated = object_detector.process(
                    annotated, lane_center_x)
            elif async_detector is not None:
                async_detector.submit(frame, lane_center_x)
                obj_result, det_age, is_new = async_detector.latest()
                detection_age = det_age or 0.0
                if obj_result is not None:
                    annotated = async_detector.annotate(annotated, obj_result)

            # ── Module 4: tracker ──────────────────────────────────────
            if async_detector is not None:
                dets = (obj_result.detections
                        if (obj_result is not None and is_new) else [])
            else:
                dets = obj_result.detections if obj_result else []
            tracks = tracker.update(dets)
            annotated = tracker.annotate(annotated, tracks)

            # ── Module 11: traffic lights & signs ──────────────────────
            speed_limit_mps = None
            red_light = False
            try:
                if traffic_light_reader and obj_result:
                    lights = traffic_light_reader.read(
                        frame, obj_result.detections)
                    annotated = traffic_light_reader.draw(annotated, lights)
                    red_light = traffic_light_reader.controlling_state in (
                        "RED", "AMBER")
                if sign_recognizer:
                    sign_res = sign_recognizer.process(frame)
                    annotated = sign_recognizer.draw(annotated, sign_res)
                    if sign_res.speed_limit_kmh is not None:
                        speed_limit_mps = sign_res.speed_limit_kmh / 3.6
            except Exception:
                pass

            # ── Module 3: decision engine ──────────────────────────────
            decision = engine.process(lane_result, tracks,
                                      detection_age_s=detection_age)
            annotated = engine.draw_hud(annotated, decision)

            # ── Module 10: FCW ─────────────────────────────────────────
            fcw_state = None
            if fcw:
                try:
                    fcw_state = fcw.process(decision, tracks)
                    annotated = fcw.draw(annotated, fcw_state, tracks)
                except Exception:
                    pass

            t1 = time.time()
            fps = 1.0 / max(t1 - t0, 1e-6)

            # ── Build & send UDP JSON packet ───────────────────────────
            track_list = []
            for t in (tracks or []):
                if not t.confirmed:
                    continue
                track_list.append({
                    "track_id": t.id,
                    "class": t.label,
                    "bbox": [int(t.x1), int(t.y1), int(t.x2), int(t.y2)],
                    "distance_m": t.smoothed_distance_m,
                    "closing_speed_mps": t.closing_speed_mps,
                    "ttc_s": t.ttc_s,
                    "in_path": t.in_path,
                    "confidence": t.confidence,
                    "risk": t.risk,
                    "confirmed": t.confirmed
                })

            road_dict = {
                "center_x": lane_result.lane_center_x,
                "offset_px": lane_result.offset_px,
                "confidence": lane_result.confidence,
                "steering": lane_result.steering,
                "source": road_source
            }

            decision_dict = {
                "longitudinal": decision.longitudinal,
                "longitudinal_level": decision.longitudinal_level,
                "lateral": decision.lateral,
                "throttle": decision.throttle,
                "brake": decision.brake,
                "rule_id": decision.rule_id,
                "reason": decision.reason,
                "degraded": decision.degraded,
                "hazard_label": decision.hazard_label,
                "hazard_id": decision.hazard_id,
                "hazard_distance_m": decision.smoothed_distance_m,
                "hazard_ttc_s": decision.ttc_s,
                "hazard_side": decision.hazard_side
            }

            packet = {
                "timestamp_ms": int(time.time() * 1000),
                "frame_id": frame_no,
                "tracks": track_list,
                "road_boundary": road_dict,
                "decision": decision_dict,
                "traffic_light": (traffic_light_reader.controlling_state
                                  if traffic_light_reader else None),
                "speed_limit_kmh": (round(speed_limit_mps * 3.6)
                                    if speed_limit_mps else None),
                "fps": round(fps, 1),
                "ego_pose": ego_receiver.latest_pose
            }

            try:
                payload = json.dumps(clean_floats(packet)).encode('utf-8')
                udp_sock.sendto(payload, dest_addr)
            except Exception as e:
                logger.debug(f"UDP send failed: {e}")

            # ── Terminal log on decision change ────────────────────────
            status = (decision.longitudinal, decision.lateral, decision.rule_id)
            if status != last_status:
                logger.info(decision.reason)
                last_status = status

            # ── Drive logger ───────────────────────────────────────────
            if drive_logger:
                try:
                    drive_logger.log(frame_no, decision, tracks, fcw=fcw_state)
                except Exception:
                    pass

            # ── Display ────────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow("ADAS Vision UDP Bridge  (Q quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_no += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if async_detector:
            async_detector.stop()
        if drive_logger:
            drive_logger.close()
        ego_receiver.stop()
        udp_sock.close()
        cv2.destroyAllWindows()
        logger.info("UDP Bridge shutdown complete.")


if __name__ == "__main__":
    main()
