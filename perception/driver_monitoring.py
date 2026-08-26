"""
driver_monitoring.py — Module 9 (Layer 7): driver attention monitoring.

A real ADAS below full autonomy must check the *driver* is paying attention.
This is a lightweight demo of that: a driver-facing camera + OpenCV Haar
cascades (bundled with opencv — no downloads) to estimate an attention state:

    ATTENTIVE   — face centred, eyes visible
    DISTRACTED  — face turned / off to the side
    DROWSY      — eyes not detected for several frames
    NO DRIVER   — no face found

It needs a driver-facing camera (run `driver_monitor_demo.py`, or pass
`--driver-cam INDEX` to main.py). It only advises — it shows a chip on the HUD.
Advisory / research only.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

import config as cfg

logger = logging.getLogger(__name__)

_STATE_COLORS = {
    "ATTENTIVE":  (0, 200, 0),
    "DISTRACTED": (0, 200, 255),
    "DROWSY":     (0, 0, 255),
    "NO DRIVER":  (0, 0, 255),
}


@dataclass
class AttentionResult:
    state:        str = "NO DRIVER"
    face_box:     Optional[Tuple[int, int, int, int]] = None
    eyes_found:   int = 0
    offset_ratio: float = 0.0


class DriverMonitor:
    def __init__(self) -> None:
        base = cv2.data.haarcascades
        self._face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._eye = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        self.available = not (self._face.empty() or self._eye.empty())
        self._no_eye_frames = 0
        if not self.available:
            logger.warning("Driver monitor: Haar cascades unavailable — disabled.")

    def reset(self) -> None:
        self._no_eye_frames = 0

    def process(self, driver_frame) -> Tuple[AttentionResult, "cv2.Mat"]:
        if not self.available:
            return AttentionResult(), driver_frame

        gray = cv2.cvtColor(driver_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        faces = self._face.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        if len(faces) == 0:
            self._no_eye_frames += 1
            res = AttentionResult(state="NO DRIVER")
            return res, self._annotate(driver_frame, res)

        # largest face
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        face_cx = fx + fw / 2.0
        offset_ratio = abs(face_cx - w / 2.0) / w

        roi = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        eyes = self._eye.detectMultiScale(roi, 1.1, 4, minSize=(20, 20))
        if len(eyes) >= 1:
            self._no_eye_frames = 0
        else:
            self._no_eye_frames += 1

        if self._no_eye_frames >= cfg.DMS_EYES_CLOSED_FRAMES:
            state = "DROWSY"
        elif offset_ratio > cfg.DMS_LOOKAWAY_RATIO:
            state = "DISTRACTED"
        else:
            state = "ATTENTIVE"

        res = AttentionResult(state=state, face_box=(fx, fy, fw, fh),
                              eyes_found=len(eyes), offset_ratio=offset_ratio)
        return res, self._annotate(driver_frame, res)

    def _annotate(self, frame, res: AttentionResult):
        col = _STATE_COLORS.get(res.state, (200, 200, 200))
        if res.face_box:
            x, y, w, h = res.face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        cv2.putText(frame, f"DRIVER: {res.state}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        return frame

    def draw_chip(self, road_frame, res: AttentionResult):
        """Small attention chip drawn on the MAIN road frame's HUD."""
        col = _STATE_COLORS.get(res.state, (200, 200, 200))
        W = road_frame.shape[1]
        x0 = W - 210
        cv2.rectangle(road_frame, (x0, 56), (x0 + 196, 82), (20, 20, 20), -1)
        cv2.circle(road_frame, (x0 + 14, 69), 6, col, -1)
        cv2.putText(road_frame, f"DRIVER {res.state}", (x0 + 28, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return road_frame
