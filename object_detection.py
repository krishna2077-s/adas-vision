"""
object_detection.py — Module 2: Real-time object detection using YOLOv8n.

Detects road-relevant objects (vehicles, pedestrians, two-wheelers, animals),
estimates their distance using a monocular bounding-box heuristic, and flags
collision risk based on distance + position within the driving lane.

The model (YOLOv8n) is the smallest YOLOv8 variant — ~3M parameters, designed
for CPU / edge inference. It downloads automatically on first run (~6 MB).

NOTE on distance: a single camera cannot measure true distance. We estimate it
from the object's apparent size using a pinhole-camera approximation calibrated
with typical real-world object heights. Treat these as relative estimates, not
survey-grade measurements — good enough for "is this getting closer" logic.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection data class
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detected object."""
    label:       str
    confidence:  float
    x1: int
    y1: int
    x2: int
    y2: int
    distance_m:  float = 0.0        # estimated distance in metres
    risk:        str   = "LOW"      # LOW / MEDIUM / HIGH
    in_path:     bool  = False      # is the object in the vehicle's forward path?

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)


@dataclass
class ObjectDetectionResult:
    """All outputs from one frame of object detection."""
    detections:     List[Detection] = field(default_factory=list)
    nearest_in_path: Optional[Detection] = None
    highest_risk:    str = "LOW"
    fps:             float = 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    YOLOv8n-based road object detector with monocular distance estimation.

    Usage::

        detector = ObjectDetector(frame_width=1280, frame_height=720)
        result, annotated = detector.process(frame, lane_center_x=640)
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self._prev_tick = cv2.getTickCount()

        # Lazy import so the rest of the system works even before ultralytics
        # is installed (lane detection has no such dependency).
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for object detection.\n"
                "Install it:  pip install ultralytics"
            ) from exc

        logger.info(f"Loading YOLO model '{cfg.YOLO_MODEL}' (downloads on first run) ...")
        self.model = YOLO(cfg.YOLO_MODEL)
        logger.info("YOLO model loaded.")

        # Pre-compute the focal length used by the distance estimator.
        # focal_px = (known_pixel_height * known_distance) / real_height
        # We calibrate off a car: a ~1.5 m tall car occupying ~220 px at ~15 m.
        self._focal_px = (cfg.CALIB_PIXEL_HEIGHT * cfg.CALIB_DISTANCE_M) / cfg.CALIB_REAL_HEIGHT_M

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        lane_center_x: Optional[int] = None,
    ) -> Tuple[ObjectDetectionResult, np.ndarray]:
        """
        Runs YOLO on one frame, estimates distances, assigns risk levels.

        Args:
            frame:         BGR numpy array.
            lane_center_x: X pixel of the lane centre from Module 1. Used to
                           decide which objects are in the vehicle's path.
                           Falls back to frame centre if None.

        Returns:
            (ObjectDetectionResult, annotated_frame)
        """
        if lane_center_x is None:
            lane_center_x = self.w // 2

        # ── YOLO inference ────────────────────────────────────────────
        yolo_out = self.model(
            frame,
            conf=cfg.YOLO_CONF_THRESHOLD,
            iou=cfg.YOLO_IOU_THRESHOLD,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        for box in yolo_out.boxes:
            cls_id = int(box.cls[0])
            label  = self.model.names[cls_id]

            # Keep only road-relevant classes
            if label not in cfg.RELEVANT_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            det = Detection(label=label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
            det.distance_m = self._estimate_distance(det)
            det.in_path    = self._is_in_path(det, lane_center_x)
            det.risk       = self._assess_risk(det)
            detections.append(det)

        # ── Aggregate ─────────────────────────────────────────────────
        result = self._aggregate(detections)

        # ── Annotate ──────────────────────────────────────────────────
        annotated = self._annotate(frame.copy(), result, lane_center_x)

        # ── FPS ───────────────────────────────────────────────────────
        tick = cv2.getTickCount()
        result.fps = cv2.getTickFrequency() / (tick - self._prev_tick)
        self._prev_tick = tick

        return result, annotated

    # ------------------------------------------------------------------
    # Distance estimation (monocular, approximate)
    # ------------------------------------------------------------------

    def _estimate_distance(self, det: Detection) -> float:
        """
        Estimates distance using the pinhole camera model:
            distance = (real_height * focal_px) / pixel_height

        Uses a per-class real-world height. This is an approximation — it
        assumes the object is upright and fully visible in frame.
        """
        real_height = cfg.CLASS_REAL_HEIGHTS.get(det.label, cfg.CLASS_REAL_HEIGHTS["default"])
        pixel_height = max(det.height, 1)   # avoid division by zero
        distance = (real_height * self._focal_px) / pixel_height
        return round(distance, 1)

    def _is_in_path(self, det: Detection, lane_center_x: int) -> bool:
        """
        An object is 'in path' if its bottom-centre falls within a corridor
        around the lane centre. The corridor widens toward the bottom of the
        frame (closer objects) to approximate perspective.
        """
        bx, by = det.bottom_center

        # Corridor half-width grows as the object sits lower in the frame
        depth_ratio = by / self.h                      # 0 (top) .. 1 (bottom)
        half_width = int(cfg.PATH_CORRIDOR_MIN_PX +
                         depth_ratio * (cfg.PATH_CORRIDOR_MAX_PX - cfg.PATH_CORRIDOR_MIN_PX))

        return abs(bx - lane_center_x) < half_width

    def _assess_risk(self, det: Detection) -> str:
        """
        Risk = function of distance AND whether the object is in the path.
        Objects outside the path get downgraded by one level.
        """
        d = det.distance_m
        if d < cfg.RISK_DISTANCE_HIGH:
            base = "HIGH"
        elif d < cfg.RISK_DISTANCE_MEDIUM:
            base = "MEDIUM"
        else:
            base = "LOW"

        if not det.in_path:
            # Downgrade risk for objects off to the side
            base = {"HIGH": "MEDIUM", "MEDIUM": "LOW", "LOW": "LOW"}[base]

        return base

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, detections: List[Detection]) -> ObjectDetectionResult:
        result = ObjectDetectionResult(detections=detections)

        in_path = [d for d in detections if d.in_path]
        if in_path:
            result.nearest_in_path = min(in_path, key=lambda d: d.distance_m)

        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if detections:
            result.highest_risk = max((d.risk for d in detections), key=lambda r: rank[r])

        return result

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    _RISK_COLORS = {
        "LOW":    (0,   200, 0),    # green
        "MEDIUM": (0,   200, 255),  # amber
        "HIGH":   (0,   0,   255),  # red
    }

    def _annotate(
        self,
        frame: np.ndarray,
        result: ObjectDetectionResult,
        lane_center_x: int,
    ) -> np.ndarray:
        for det in result.detections:
            color = self._RISK_COLORS[det.risk]
            thickness = 3 if det.in_path else 1

            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, thickness)

            # Label background
            label_text = f"{det.label} {det.distance_m:.0f}m"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (det.x1, det.y1 - th - 8), (det.x1 + tw + 6, det.y1), color, -1)
            cv2.putText(frame, label_text, (det.x1 + 3, det.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Highlight in-path objects with a bottom-centre dot
            if det.in_path:
                cv2.circle(frame, det.bottom_center, 5, color, -1)

        # Forward collision warning banner
        if result.nearest_in_path and result.nearest_in_path.risk == "HIGH":
            self._draw_fcw_banner(frame, result.nearest_in_path)

        return frame

    def _draw_fcw_banner(self, frame: np.ndarray, det: Detection) -> None:
        """Draws a Forward Collision Warning banner across the top of the frame."""
        banner_h = 44
        x0 = self.w // 2 - 240
        x1 = self.w // 2 + 240
        cv2.rectangle(frame, (x0, 6), (x1, 6 + banner_h), (0, 0, 200), -1)
        text = f"! COLLISION WARNING  {det.label} {det.distance_m:.0f}m"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.putText(frame, text, (self.w // 2 - tw // 2, 6 + banner_h // 2 + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
