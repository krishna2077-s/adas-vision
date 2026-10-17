"""
driver_monitoring.py — Module 9 v2: Driver Monitoring System (DMS).

Monitors the driver through an internal (cabin-facing) camera and produces a
rich AttentionResult per frame covering:

    State classification
    ─────────────────────────────────────────────────────────────────────
    ATTENTIVE   — face centred, eyes open, gaze forward, hands on wheel
    DISTRACTED  — head turned / gaze off-road for several frames
    DROWSY      — PERCLOS above threshold or head-drop detected
    PHONE       — phone / handheld object detected in frame
    HANDS_OFF   — hands absent from the wheel region for too long
    NO DRIVER   — no face found

    Sub-signals (MediaPipe backend)
    ─────────────────────────────────────────────────────────────────────
    • Gaze zone         — FORWARD / LEFT / RIGHT / DOWN / CLOSED
    • Head pose         — yaw_deg, pitch_deg (solvePnP on 6 face landmarks)
    • Eye Aspect Ratio  — per-eye EAR, blink counter
    • PERCLOS           — % eyes-closed frames in the last N-frame window
    • Hands on wheel    — 0 / 1 / 2 hands detected in the wheel region
    • Phone detected    — YOLOv8n cell-phone class on the driver frame

    Fallback (Haar backend — zero extra deps)
    ─────────────────────────────────────────────────────────────────────
    • Face centred check  (offset_ratio)
    • Eye detection count (no EAR / PERCLOS)

Backend selection
─────────────────────────────────────────────────────────────────────────────
    cfg.DMS_BACKEND = "mediapipe"   # default — full feature set
    cfg.DMS_BACKEND = "haar"        # legacy fallback — no new dependencies

Advisory / research only — this module never controls a vehicle.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmark index constants (FaceMesh 468-pt canonical map)
# ---------------------------------------------------------------------------

# 6-point head-pose model (same as dlib/OpenFace conventions)
_HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]  # nose, chin, R-eye-R, L-eye-L, R-mouth, L-mouth

# Eye landmarks (right eye = 33-side, left eye = 263-side in MediaPipe mirror)
_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # p1..p6 for EAR
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# Iris centre estimates (MediaPipe FaceMesh Attention – indices 468-477 if available;
# fall back to geometric eye-centre for the standard 468-pt model)
_RIGHT_IRIS_CENTRE = 468
_LEFT_IRIS_CENTRE  = 473

# ---------------------------------------------------------------------------
# 3-D face model points (mm, OpenCV convention) — used for solvePnP
# ---------------------------------------------------------------------------
_FACE_3D_MODEL = np.array([
    [0.0,     0.0,     0.0],     # Nose tip       (lm 1)
    [0.0,    330.0,  -65.0],     # Chin            (lm 152)  — below nose, +Y in OpenCV
    [225.0,  -170.0, -135.0],    # L eye outer     (lm 263)  — camera-left = +X
    [-225.0, -170.0, -135.0],    # R eye outer     (lm 33)   — camera-right = -X
    [150.0,   150.0, -125.0],    # L mouth corner  (lm 287)
    [-150.0,  150.0, -125.0],    # R mouth corner  (lm 57)
], dtype=np.float64)

# ---------------------------------------------------------------------------
# HUD colours (BGR)
# ---------------------------------------------------------------------------
_STATE_COLORS = {
    "ATTENTIVE":  (0, 200, 0),
    "DISTRACTED": (0, 200, 255),
    "DROWSY":     (0, 100, 255),
    "PHONE":      (0, 0, 255),
    "HANDS_OFF":  (0, 180, 255),
    "NO DRIVER":  (80, 80, 80),
}
_GAZE_COLORS = {
    "FORWARD": (0, 200, 0),
    "LEFT":    (0, 200, 255),
    "RIGHT":   (0, 200, 255),
    "DOWN":    (0, 100, 255),
    "CLOSED":  (0, 0, 255),
    "UNKNOWN": (120, 120, 120),
}


# ---------------------------------------------------------------------------
# AttentionResult
# ---------------------------------------------------------------------------

@dataclass
class AttentionResult:
    """Full DMS output for one driver frame."""
    # ── Core state ─────────────────────────────────────────────────────
    state:            str   = "NO DRIVER"   # see module docstring
    # ── Face / gaze ────────────────────────────────────────────────────
    face_box:         Optional[Tuple[int, int, int, int]] = None
    eyes_found:       int   = 0
    offset_ratio:     float = 0.0           # horizontal face offset (legacy)
    gaze_zone:        str   = "UNKNOWN"     # FORWARD/LEFT/RIGHT/DOWN/CLOSED
    yaw_deg:          float = 0.0
    pitch_deg:        float = 0.0
    ear:              float = 1.0           # Eye Aspect Ratio (mean)
    perclos:          float = 0.0           # % closed frames in window
    drowsiness_score: float = 0.0          # composite 0-1
    # ── Hands ──────────────────────────────────────────────────────────
    hands_on_wheel:   int   = 0             # 0 / 1 / 2
    # ── Phone & Behavior ───────────────────────────────────────────────
    phone_detected:   bool  = False
    behavior_label:   str   = "Safe driving"
    behavior_conf:    float = 0.0
    # ── Meta ───────────────────────────────────────────────────────────
    confidence:       float = 0.0           # overall DMS confidence 0-1
    backend:          str   = "none"        # "mediapipe" | "haar" | "none"


# ===========================================================================
# Face analyser — MediaPipe FaceMesh
# ===========================================================================

class _FaceAnalyzer:
    """
    Uses MediaPipe FaceMesh to estimate head pose, gaze zone, EAR and PERCLOS.
    Raises ImportError at construction if mediapipe is not installed.
    """

    def __init__(self, frame_w: int, frame_h: int) -> None:
        import os
        import urllib.request
        import mediapipe as mp  # noqa: F401 (ImportError propagates)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        self._mp = mp
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        if not os.path.exists(model_path):
            logger.info("Downloading face_landmarker.task...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", model_path)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._face_mesh = vision.FaceLandmarker.create_from_options(options)
        self._w = frame_w
        self._h = frame_h

        # PERCLOS rolling window
        self._eye_closed_history: deque = deque(maxlen=cfg.DMS_PERCLOS_WINDOW)

        # Distraction / drowsy consecutive-frame counters
        self._distract_frames = 0
        self._drowsy_frames   = 0

        # Camera intrinsics — deferred until first process() call to use
        # the actual driver frame resolution (may differ from dashcam).
        self._cam_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        logger.info("DMS FaceAnalyzer: MediaPipe FaceMesh ready.")

    def reset(self) -> None:
        self._eye_closed_history.clear()
        self._distract_frames = 0
        self._drowsy_frames   = 0

    # ------------------------------------------------------------------

    def process(self, gray_or_bgr) -> dict:
        """
        Returns a dict with keys:
          face_found, face_box, offset_ratio,
          gaze_zone, yaw_deg, pitch_deg,
          ear, perclos, drowsiness_score, eyes_found
        """
        # Lazy-init camera intrinsics from the actual driver frame resolution
        if self._cam_matrix is None:
            fh, fw = gray_or_bgr.shape[:2]
            focal = fw
            self._cam_matrix = np.array([
                [focal,     0, fw / 2],
                [0,     focal, fh / 2],
                [0,         0,      1],
            ], dtype=np.float64)

        rgb = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2RGB) \
              if gray_or_bgr.ndim == 3 and gray_or_bgr.shape[2] == 3 \
              else gray_or_bgr
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._face_mesh.detect(mp_image)

        blank = dict(
            face_found=False, face_box=None, offset_ratio=0.0,
            gaze_zone="UNKNOWN", yaw_deg=0.0, pitch_deg=0.0,
            ear=1.0, perclos=0.0, drowsiness_score=0.0, eyes_found=0,
        )

        if not result.face_landmarks:
            self._eye_closed_history.append(1)  # treat no-face as closed
            self._drowsy_frames = min(self._drowsy_frames + 1, cfg.DMS_DROWSY_FRAMES * 3)
            blank["perclos"] = self._perclos()
            blank["drowsiness_score"] = self._drowsiness_score(blank["perclos"], 0.0, 1.0)
            return blank

        lm = result.face_landmarks[0]
        h, w = gray_or_bgr.shape[:2]

        # ── Face bounding box (pixel coords) ─────────────────────────
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        fx, fy = int(min(xs)), int(min(ys))
        fw, fh = int(max(xs)) - fx, int(max(ys)) - fy
        face_cx    = (min(xs) + max(xs)) / 2.0
        offset_ratio = abs(face_cx - w / 2.0) / w

        # ── Head pose (solvePnP) ──────────────────────────────────────
        face_2d = np.array([
            [lm[idx].x * w, lm[idx].y * h] for idx in _HEAD_POSE_LANDMARKS
        ], dtype=np.float64)

        yaw_deg = pitch_deg = 0.0
        try:
            success, rvec, _ = cv2.solvePnP(
                _FACE_3D_MODEL, face_2d,
                self._cam_matrix, self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                # cv2.RQDecomp3x3 returns (pitch, yaw, roll) in degrees directly,
                # correctly decomposed from the rotation matrix.
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch_deg = angles[0]
                yaw_deg   = angles[1]
        except cv2.error:
            pass

        # ── Eye Aspect Ratio (EAR) ────────────────────────────────────
        ear_r = self._ear(lm, _RIGHT_EYE_IDX, w, h)
        ear_l = self._ear(lm, _LEFT_EYE_IDX,  w, h)
        ear   = (ear_r + ear_l) / 2.0

        eyes_closed = ear < cfg.DMS_EAR_THRESH
        self._eye_closed_history.append(1 if eyes_closed else 0)
        eyes_found = 0 if eyes_closed else 2

        # ── PERCLOS ───────────────────────────────────────────────────
        perclos = self._perclos()

        # ── Gaze zone ─────────────────────────────────────────────────
        if eyes_closed:
            gaze_zone = "CLOSED"
        elif abs(yaw_deg) > cfg.DMS_YAW_THRESH_DEG:
            gaze_zone = "RIGHT" if yaw_deg > 0 else "LEFT"
        elif pitch_deg < -cfg.DMS_PITCH_THRESH_DEG:
            gaze_zone = "DOWN"
        else:
            gaze_zone = "FORWARD"

        # ── Consecutive-frame state tracking ─────────────────────────
        if gaze_zone != "FORWARD":
            self._distract_frames += 1
        else:
            self._distract_frames = max(0, self._distract_frames - 2)

        if perclos > cfg.DMS_PERCLOS_THRESH:
            self._drowsy_frames += 1
        else:
            self._drowsy_frames = max(0, self._drowsy_frames - 1)

        drowsiness_score = self._drowsiness_score(perclos, ear, pitch_deg)

        return dict(
            face_found=True, face_box=(fx, fy, fw, fh),
            offset_ratio=offset_ratio,
            gaze_zone=gaze_zone, yaw_deg=yaw_deg, pitch_deg=pitch_deg,
            ear=ear, perclos=perclos, drowsiness_score=drowsiness_score,
            eyes_found=eyes_found,
            _distract_frames=self._distract_frames,
            _drowsy_frames=self._drowsy_frames,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _ear(lm, indices: List[int], w: int, h: int) -> float:
        """Eye Aspect Ratio from 6 landmark indices (Soukupova & Cech, 2016)."""
        p = [(lm[i].x * w, lm[i].y * h) for i in indices]
        # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        v1 = math.dist(p[1], p[5])
        v2 = math.dist(p[2], p[4])
        h0 = math.dist(p[0], p[3])
        return (v1 + v2) / (2.0 * h0 + 1e-6)

    def _perclos(self) -> float:
        if not self._eye_closed_history:
            return 0.0
        return sum(self._eye_closed_history) / len(self._eye_closed_history)

    @staticmethod
    def _drowsiness_score(perclos: float, ear: float, pitch_deg: float) -> float:
        """Composite drowsiness score in [0, 1]."""
        score = perclos * 0.6
        if ear < cfg.DMS_EAR_THRESH:
            score += 0.2
        if pitch_deg < -cfg.DMS_PITCH_THRESH_DEG:
            score += 0.2
        return min(score, 1.0)


# ===========================================================================
# Hand analyser — MediaPipe Hands
# ===========================================================================

class _HandAnalyzer:
    """
    Detects both hands and checks how many are in the steering-wheel region.
    Raises ImportError at construction if mediapipe is not installed.
    """

    def __init__(self, frame_w: int, frame_h: int) -> None:
        import os
        import urllib.request
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        self._mp = mp
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            logger.info("Downloading hand_landmarker.task...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", model_path)
            
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)
        self._w = frame_w
        self._h = frame_h
        self._no_hands_frames = 0
        logger.info("DMS HandAnalyzer: MediaPipe Hands ready.")

    def reset(self) -> None:
        self._no_hands_frames = 0

    def process(self, bgr_frame) -> dict:
        """Returns dict: hands_on_wheel (int 0-2), no_hands_frames (int)."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._hands.detect(mp_image)

        h, w = bgr_frame.shape[:2]
        # Wheel region: lower portion of the frame, centred horizontally
        wheel_y_min = h * cfg.DMS_WHEEL_Y_FRAC
        wheel_x_min = w * (0.5 - cfg.DMS_WHEEL_X_RANGE)
        wheel_x_max = w * (0.5 + cfg.DMS_WHEEL_X_RANGE)

        hands_on_wheel = 0
        hand_boxes: List[Tuple[int, int, int, int]] = []

        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                wrist = hand_lm[0]
                wx_px = wrist.x * w
                wy_px = wrist.y * h
                in_wheel = (wy_px >= wheel_y_min
                            and wheel_x_min <= wx_px <= wheel_x_max)
                if in_wheel:
                    hands_on_wheel += 1

                # Bounding box for HUD
                xs = [lm.x * w for lm in hand_lm]
                ys = [lm.y * h for lm in hand_lm]
                hand_boxes.append((
                    int(min(xs)), int(min(ys)),
                    int(max(xs) - min(xs)), int(max(ys) - min(ys)),
                ))

        if hands_on_wheel == 0:
            self._no_hands_frames += 1
        else:
            self._no_hands_frames = 0

        return dict(
            hands_on_wheel=hands_on_wheel,
            no_hands_frames=self._no_hands_frames,
            hand_boxes=hand_boxes,
        )


# ===========================================================================
# Phone detector — YOLOv8n on the driver frame
# ===========================================================================

class _PhoneDetector:
    """
    Detects driver distraction behaviors (phone usage, texting, drinking, etc.)
    using the fine-tuned YOLOv8 classification model trained on the State Farm dataset.
    Falls back to YOLOv8 COCO cell-phone detection if custom weights are missing.
    """

    # Human-readable labels for the State Farm dataset classes
    _STATEFARM_LABELS = {
        'c0': 'Safe driving',
        'c1': 'Texting (Right Hand)',
        'c2': 'Talking on Phone (Right Hand)',
        'c3': 'Texting (Left Hand)',
        'c4': 'Talking on Phone (Left Hand)',
        'c5': 'Operating Radio',
        'c6': 'Drinking',
        'c7': 'Reaching Behind',
        'c8': 'Hair & Makeup',
        'c9': 'Talking to Passenger'
    }
    _PHONE_CLASSES = {"c1", "c2", "c3", "c4"}

    # COCO fallback class IDs
    _COCO_PHONE_CLASSES = {"cell phone", "remote", "book", "tv"}
    _CONF = 0.10

    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
            from pathlib import Path
            
            # Check for fine-tuned driver behavior classifier
            root_dir = Path(__file__).resolve().parent.parent
            custom_weights = root_dir / "runs" / "classify" / "driver_behavior_full" / "weights" / "best.pt"
            
            if custom_weights.exists():
                self._model = YOLO(str(custom_weights))
                self._is_classifier = True
                self._available = True
                logger.info(f"DMS PhoneDetector: loaded custom behavior classifier ({custom_weights.name})")
            else:
                self._model = YOLO(cfg.YOLO_MODEL)
                self._is_classifier = False
                self._available = True
                logger.info("DMS PhoneDetector: YOLOv8n ready (COCO fallback).")
        except Exception as exc:
            self._model = None
            self._available = False
            self._is_classifier = False
            logger.warning(f"DMS PhoneDetector unavailable ({exc}) — phone detection off.")

    @property
    def available(self) -> bool:
        return self._available

    def process(self, bgr_frame) -> Tuple[bool, str, float]:
        """
        Returns:
            (phone_detected: bool, behavior_label: str, behavior_conf: float)
        """
        if not self._available or bgr_frame is None:
            return False, "UNKNOWN", 0.0

        try:
            if self._is_classifier:
                # Run classification at 224x224
                results = self._model.predict(bgr_frame, verbose=False, imgsz=224)
                probs = results[0].probs
                top_idx = probs.top1
                top_class_id = results[0].names[top_idx]
                top_conf = float(probs.top1conf.item())

                # Cumulative phone probability across all 4 phone modes (text/talk left/right)
                phone_prob = sum(float(probs.data[idx]) for idx in [1, 2, 3, 4] if idx < len(probs.data))

                if top_class_id in self._PHONE_CLASSES:
                    phone_detected = top_conf >= 0.15
                    behavior_label = self._STATEFARM_LABELS.get(top_class_id, top_class_id)
                elif phone_prob >= 0.20:
                    # Phone usage is strongly suspected (e.g. hand holding phone near face/lap)
                    phone_detected = True
                    behavior_label = "Texting (Right Hand)" if float(probs.data[1]) >= float(probs.data[3]) else "Texting (Left Hand)"
                    top_conf = phone_prob
                else:
                    phone_detected = False
                    behavior_label = self._STATEFARM_LABELS.get(top_class_id, top_class_id)

                return phone_detected, behavior_label, top_conf
            else:
                # COCO object detection fallback
                results = self._model.predict(
                    bgr_frame,
                    conf=self._CONF,
                    verbose=False,
                    imgsz=640,
                )
                for r in results:
                    for cls_id, cls_name in zip(r.boxes.cls.tolist(),
                                                [self._model.names[int(c)]
                                                 for c in r.boxes.cls]):
                        if cls_name in self._COCO_PHONE_CLASSES:
                            return True, "Phone Detected", 0.70
                return False, "Safe driving", 0.90
        except Exception as exc:
            logger.debug(f"PhoneDetector: {exc}")
            return False, "UNKNOWN", 0.0


# ===========================================================================
# Haar fallback (zero extra dependencies)
# ===========================================================================

class _HaarFallback:
    """
    Minimal Haar-cascade face+eye detector — the legacy Module 9 logic.
    Used when mediapipe is not installed.
    """

    def __init__(self) -> None:
        base         = cv2.data.haarcascades
        self._face   = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._eye    = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        self.available = not (self._face.empty() or self._eye.empty())
        self._no_eye_frames = 0
        if not self.available:
            logger.warning("DMS Haar fallback: cascades unavailable.")

    def reset(self) -> None:
        self._no_eye_frames = 0

    def process(self, bgr_frame) -> dict:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        faces = self._face.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        if len(faces) == 0:
            self._no_eye_frames += 1
            return dict(
                face_found=False, face_box=None, offset_ratio=0.0,
                eyes_found=0, gaze_zone="UNKNOWN",
                yaw_deg=0.0, pitch_deg=0.0, ear=1.0,
                perclos=0.0, drowsiness_score=0.0,
                _distract_frames=0,
                _drowsy_frames=self._no_eye_frames,
            )

        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        face_cx       = fx + fw / 2.0
        offset_ratio  = abs(face_cx - w / 2.0) / w

        roi  = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        eyes = self._eye.detectMultiScale(roi, 1.1, 4, minSize=(20, 20))
        if len(eyes) >= 1:
            self._no_eye_frames = 0
        else:
            self._no_eye_frames += 1

        gaze_zone = "FORWARD" if offset_ratio <= cfg.DMS_LOOKAWAY_RATIO else "LEFT"

        return dict(
            face_found=True, face_box=(fx, fy, fw, fh),
            offset_ratio=offset_ratio,
            eyes_found=len(eyes), gaze_zone=gaze_zone,
            yaw_deg=offset_ratio * 40.0, pitch_deg=0.0,
            ear=(1.0 if len(eyes) >= 1 else 0.10),
            perclos=min(self._no_eye_frames / cfg.DMS_PERCLOS_WINDOW, 1.0),
            drowsiness_score=min(self._no_eye_frames / cfg.DMS_EYES_CLOSED_FRAMES, 1.0),
            _distract_frames=int(offset_ratio > cfg.DMS_LOOKAWAY_RATIO) * self._no_eye_frames,
            _drowsy_frames=self._no_eye_frames,
        )


# ===========================================================================
# DriverMonitorV2 — public class
# ===========================================================================

class DriverMonitor:
    """
    Module 9 v2 — Driver Monitoring System.

    Drop-in replacement for the original DriverMonitor. The constructor tries
    to build the MediaPipe backend first; on ImportError it falls back to Haar.

    Usage::

        dm = DriverMonitor(frame_width=640, frame_height=480)
        if dm.available:
            result, annotated_driver_frame = dm.process(driver_frame)
            road_frame = dm.draw_chip(road_frame, result)
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        self._w = frame_width
        self._h = frame_height
        self._backend_name = "none"

        # ── Face analyser ─────────────────────────────────────────────
        self._face: Optional[_FaceAnalyzer] = None
        if cfg.DMS_BACKEND in ("mediapipe", "auto"):
            try:
                self._face = _FaceAnalyzer(frame_width, frame_height)
                self._backend_name = "mediapipe"
            except (ImportError, AttributeError) as exc:
                logger.warning(f"DMS: MediaPipe unavailable ({exc}), falling back to Haar.")

        if self._face is None:
            haar = _HaarFallback()
            if haar.available:
                self._face = haar          # type: ignore[assignment]
                self._backend_name = "haar"
            else:
                self._face = None

        # ── Hand analyser ─────────────────────────────────────────────
        self._hands: Optional[_HandAnalyzer] = None
        if cfg.DMS_ENABLE_HANDS and self._backend_name == "mediapipe":
            try:
                self._hands = _HandAnalyzer(frame_width, frame_height)
            except (ImportError, AttributeError) as exc:
                logger.warning(f"DMS HandAnalyzer unavailable ({exc}).")

        # ── Phone detector ────────────────────────────────────────────
        self._phone: Optional[_PhoneDetector] = None
        if cfg.DMS_ENABLE_PHONE:
            pd = _PhoneDetector()
            if pd.available:
                self._phone = pd

        # ── State smoothing ───────────────────────────────────────────
        self._state_history: deque = deque(maxlen=5)

        self.available = self._face is not None
        if not self.available:
            logger.warning("DMS: no face detector available — driver monitoring disabled.")
        else:
            logger.info(
                f"DriverMonitorV2 ready — backend={self._backend_name}, "
                f"hands={'on' if self._hands else 'off'}, "
                f"phone={'on' if self._phone else 'off'}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear temporal state (called on video loop restart)."""
        if self._face is not None:
            self._face.reset()
        if self._hands is not None:
            self._hands.reset()
        self._state_history.clear()

    def process(self, driver_frame) -> Tuple[AttentionResult, "cv2.Mat"]:
        """
        Analyse one driver-camera frame.

        Returns:
            (AttentionResult, annotated_driver_frame)
        """
        if not self.available or driver_frame is None:
            return AttentionResult(backend=self._backend_name), driver_frame

        h, w = driver_frame.shape[:2]

        # ── Face / gaze ───────────────────────────────────────────────
        face_data = self._face.process(driver_frame)

        # ── Hands ─────────────────────────────────────────────────────
        hand_data = {"hands_on_wheel": 0, "no_hands_frames": 0, "hand_boxes": []}
        if self._hands is not None:
            hand_data = self._hands.process(driver_frame)

        # ── Phone & Behavior ──────────────────────────────────────────
        phone_detected = False
        behavior_label = "Safe driving"
        behavior_conf = 0.0
        if self._phone is not None:
            phone_detected, behavior_label, behavior_conf = self._phone.process(driver_frame)

        # ── State classification ──────────────────────────────────────
        state = self._classify(face_data, hand_data, phone_detected, behavior_label, behavior_conf)
        self._state_history.append(state)
        # Smooth: use the most common state in recent history
        smoothed_state = max(set(self._state_history), key=list(self._state_history).count)

        # ── Confidence ────────────────────────────────────────────────
        confidence = self._confidence(face_data, hand_data)

        result = AttentionResult(
            state            = smoothed_state,
            face_box         = face_data.get("face_box"),
            eyes_found       = face_data.get("eyes_found", 0),
            offset_ratio     = face_data.get("offset_ratio", 0.0),
            gaze_zone        = face_data.get("gaze_zone", "UNKNOWN"),
            yaw_deg          = face_data.get("yaw_deg", 0.0),
            pitch_deg        = face_data.get("pitch_deg", 0.0),
            ear              = face_data.get("ear", 1.0),
            perclos          = face_data.get("perclos", 0.0),
            drowsiness_score = face_data.get("drowsiness_score", 0.0),
            hands_on_wheel   = hand_data["hands_on_wheel"],
            phone_detected   = phone_detected,
            behavior_label   = behavior_label,
            behavior_conf    = behavior_conf,
            confidence       = confidence,
            backend          = self._backend_name,
        )

        annotated = self._annotate_driver_frame(driver_frame.copy(), result, hand_data)
        return result, annotated

    # ------------------------------------------------------------------
    # State classification
    # ------------------------------------------------------------------

    def _classify(self, fd: dict, hd: dict, phone: bool, behavior_label: str = "Safe driving", behavior_conf: float = 0.0) -> str:
        if phone:
            return "PHONE"
        if not fd.get("face_found", False):
            return "NO DRIVER"
        drowsy_frames   = fd.get("_drowsy_frames",   0)
        distract_frames = fd.get("_distract_frames", 0)
        no_hands_frames = hd.get("no_hands_frames",  0)

        # Expedite distraction alert if custom model detects secondary distraction (drinking, reaching behind, radio)
        is_secondary_distraction = (
            behavior_label not in ("Safe driving", "UNKNOWN") and
            behavior_conf >= 0.40 and
            not phone
        )
        if is_secondary_distraction:
            distract_frames += 15

        if drowsy_frames >= cfg.DMS_DROWSY_FRAMES:
            return "DROWSY"
        if distract_frames >= cfg.DMS_DISTRACT_FRAMES:
            return "DISTRACTED"
        if self._hands is not None and no_hands_frames >= cfg.DMS_HANDS_WARN_FRAMES:
            return "HANDS_OFF"
        return "ATTENTIVE"

    @staticmethod
    def _confidence(fd: dict, hd: dict) -> float:
        if not fd.get("face_found", False):
            return 0.0
        c = 0.6      # base for a detected face
        if fd.get("eyes_found", 0) > 0:
            c += 0.2
        if fd.get("gaze_zone", "UNKNOWN") != "UNKNOWN":
            c += 0.2
        return min(c, 1.0)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _annotate_driver_frame(self, frame, res: AttentionResult, hd: dict):
        """Draw overlays on the driver-facing frame itself."""
        col = _STATE_COLORS.get(res.state, (200, 200, 200))
        h, w = frame.shape[:2]

        # Face box
        if res.face_box:
            x, y, bw, bh = res.face_box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)

        # Hand boxes
        for hx, hy, hw2, hh in hd.get("hand_boxes", []):
            cv2.rectangle(frame, (hx, hy), (hx + hw2, hy + hh), (255, 200, 0), 1)

        # State label
        cv2.putText(frame, f"DRIVER: {res.state}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

        # Gaze zone
        gz_col = _GAZE_COLORS.get(res.gaze_zone, (200, 200, 200))
        cv2.putText(frame, f"GAZE: {res.gaze_zone}", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, gz_col, 1, cv2.LINE_AA)

        # Yaw / pitch
        cv2.putText(frame,
                    f"YAW {res.yaw_deg:+.0f}\u00b0  PITCH {res.pitch_deg:+.0f}\u00b0",
                    (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

        # EAR / PERCLOS bar
        ear_col = (0, 0, 255) if res.ear < cfg.DMS_EAR_THRESH else (0, 220, 0)
        cv2.putText(frame, f"EAR {res.ear:.2f}  PERCLOS {res.perclos * 100:.0f}%",
                    (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.50, ear_col, 1, cv2.LINE_AA)

        # Hands
        hw_col = (0, 220, 0) if res.hands_on_wheel >= 2 else \
                 (0, 200, 255) if res.hands_on_wheel == 1 else (0, 0, 255)
        if self._hands is not None:
            cv2.putText(frame, f"HANDS ON WHEEL: {res.hands_on_wheel}",
                        (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, hw_col, 1, cv2.LINE_AA)

        # Behavior
        if res.behavior_label and res.behavior_label != "UNKNOWN":
            beh_col = (0, 0, 255) if res.phone_detected else (0, 200, 255) if res.state == "DISTRACTED" else (0, 220, 0)
            cv2.putText(frame, f"BEHAVIOR: {res.behavior_label} ({res.behavior_conf * 100:.0f}%)",
                        (12, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.50, beh_col, 1, cv2.LINE_AA)

        # Phone
        if res.phone_detected:
            cv2.putText(frame, "PHONE DETECTED!", (12, 178),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # Backend chip
        cv2.putText(frame, f"[{res.backend.upper()}]",
                    (w - 110, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (100, 100, 100), 1, cv2.LINE_AA)

        return frame

    def draw_chip(self, road_frame, res: AttentionResult):
        """
        Draw the compact DMS chip on the main ADAS road frame HUD.
        Provides state, gaze zone, hands count, behavior and phone flag at a glance.
        """
        col  = _STATE_COLORS.get(res.state, (200, 200, 200))
        W    = road_frame.shape[1]
        chip_w = 240
        x0   = W - chip_w - 10
        y0   = 56

        # Background panel
        cv2.rectangle(road_frame, (x0, y0), (x0 + chip_w, y0 + cfg.DMS_CHIP_H * 3 + 4),
                      (20, 20, 20), -1)

        # Row 1 — State
        cv2.circle(road_frame, (x0 + 14, y0 + 10), 7, col, -1)
        cv2.putText(road_frame, f"DRIVER: {res.state}",
                    (x0 + 28, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

        # Row 2 — Gaze + EAR
        gz_col = _GAZE_COLORS.get(res.gaze_zone, (160, 160, 160))
        cv2.putText(road_frame,
                    f"GAZE:{res.gaze_zone}  EAR:{res.ear:.2f}",
                    (x0 + 8, y0 + cfg.DMS_CHIP_H + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, gz_col, 1, cv2.LINE_AA)

        # Row 3 — Hands + phone / behavior
        hw_str = f"HANDS:{res.hands_on_wheel} " if self._hands else ""
        if res.phone_detected:
            ph_str = "[P] PHONE"
        elif res.behavior_label not in ("Safe driving", "UNKNOWN"):
            ph_str = f"[!] {res.behavior_label[:12]}"
        else:
            ph_str = "[OK] NORMAL"

        hw_col = (0, 220, 0) if res.hands_on_wheel >= 2 else \
                 (0, 200, 255) if res.hands_on_wheel == 1 else (0, 0, 255)
        cv2.putText(road_frame,
                    hw_str + ph_str,
                    (x0 + 8, y0 + cfg.DMS_CHIP_H * 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 255) if res.phone_detected else hw_col,
                    1, cv2.LINE_AA)

        return road_frame
