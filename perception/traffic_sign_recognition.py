"""
traffic_sign_recognition.py — Module 11b: traffic-SIGN recognition.

YOLO/COCO knows "stop sign" but nothing about speed limits, so this module adds
a classical detect-then-classify sign pipeline that DOES read them:

    1. Region proposal (no training): threshold the frame for sign colours
       (red rims of prohibitory/warning signs, blue of mandatory ones), clean
       the mask, and take blobs that are the right size, roughly square, and in
       the band where signs actually sit (upper-middle of the view, never the
       bonnet). Cheap and runs every frame.
    2. Classification (tiny CNN): each candidate crop is resized to 32x32 and run
       through a small CNN trained on GTSRB (43 classes). Softmax below
       SIGN_MIN_CONF is rejected — we would rather miss a sign than invent one.
    3. Temporal vote: a class must lead in SIGN_VOTE_MIN of the last
       SIGN_VOTE_WINDOW frames before it is 'recognised', killing one-frame
       false positives. The last recognised speed-limit latches as the active
       limit and is offered to the advisory planner.

Gated exactly like the learned road model: without the weights file (or without
torch) the recogniser reports ``available = False`` and does nothing, so a clean
checkout still runs. Train the weights in ~5 min on CPU with train_signs.py.

⚠️ Advisory only. A recognised limit or stop sign is surfaced to the human and
to the advisory planner; nothing here actuates a vehicle. Monocular sign reading
is imperfect (motion blur, look-alikes, non-German sign styles) — a cue, not law.
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)

# GTSRB 43-class labels (id -> human name). Speed-limit ids carry a km/h value.
GTSRB_CLASSES = {
    0: "speed limit 20", 1: "speed limit 30", 2: "speed limit 50",
    3: "speed limit 60", 4: "speed limit 70", 5: "speed limit 80",
    6: "end speed limit 80", 7: "speed limit 100", 8: "speed limit 120",
    9: "no passing", 10: "no passing >3.5t", 11: "right-of-way at next",
    12: "priority road", 13: "yield", 14: "stop", 15: "no vehicles",
    16: "no >3.5t", 17: "no entry", 18: "general caution",
    19: "curve left", 20: "curve right", 21: "double curve",
    22: "bumpy road", 23: "slippery road", 24: "road narrows right",
    25: "road work", 26: "traffic signals", 27: "pedestrians",
    28: "children crossing", 29: "bicycles crossing", 30: "ice/snow",
    31: "wild animals", 32: "end limits", 33: "turn right ahead",
    34: "turn left ahead", 35: "ahead only", 36: "go straight or right",
    37: "go straight or left", 38: "keep right", 39: "keep left",
    40: "roundabout", 41: "end no passing", 42: "end no passing >3.5t",
}
SPEED_LIMIT_KMH = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}
NUM_CLASSES = 43


# ---------------------------------------------------------------------------
# Preprocessing shared by training and inference (must stay identical)
# ---------------------------------------------------------------------------

def sign_crop_to_tensor(crop_bgr: np.ndarray, size: int):
    """Resize a BGR crop to (size,size), RGB, [0,1], CHW float32 -> torch tensor."""
    import torch
    rgb = cv2.cvtColor(cv2.resize(crop_bgr, (size, size)), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    return torch.from_numpy(x)


def build_sign_cnn():
    """Small, fast GTSRB CNN (~0.5M params). Defined here so train + infer share it."""
    import torch.nn as nn

    class SignCNN(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            def block(ci, co):
                return nn.Sequential(
                    nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                    nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))
            self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.4), nn.Linear(256, num_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    return SignCNN()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class SignDetection:
    label: str
    class_id: int
    confidence: float
    box: Tuple[int, int, int, int]     # x1, y1, x2, y2


@dataclass
class SignResult:
    detections: List[SignDetection] = field(default_factory=list)   # this frame (raw, high-conf)
    recognised: List[str] = field(default_factory=list)             # temporally-confirmed labels
    speed_limit_kmh: Optional[int] = None                           # latched active limit


class SignRecognizer:
    """Colour/shape region proposals + tiny GTSRB CNN. Inert without weights."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self.available = False
        self.model = None
        self._recent: "deque[set]" = deque(maxlen=cfg.SIGN_VOTE_WINDOW)
        self.speed_limit_kmh: Optional[int] = None
        self._load()

    def _load(self) -> None:
        import os
        if not os.path.exists(cfg.SIGN_MODEL_PATH):
            logger.info(f"Sign model '{cfg.SIGN_MODEL_PATH}' not found — sign recognition off "
                        f"(train it with train_signs.py to enable).")
            return
        try:
            import torch
            self.model = build_sign_cnn()
            self.model.load_state_dict(torch.load(cfg.SIGN_MODEL_PATH, map_location="cpu"))
            self.model.eval()
            self._torch = torch
            self.available = True
            logger.info(f"SignRecognizer loaded '{cfg.SIGN_MODEL_PATH}'.")
        except Exception as exc:                       # torch missing / bad file
            logger.warning(f"Sign recognition unavailable ({exc}).")
            self.model = None
            self.available = False

    def reset(self) -> None:
        self._recent.clear()
        self.speed_limit_kmh = None

    # ------------------------------------------------------------------
    # Region proposal (no model needed)
    # ------------------------------------------------------------------

    def propose_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Colour+shape candidate sign boxes. Usable standalone (tested without weights)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = (cv2.inRange(hsv, (0, 90, 70), (10, 255, 255))
               | cv2.inRange(hsv, (170, 90, 70), (180, 255, 255)))
        blue = cv2.inRange(hsv, (100, 120, 60), (130, 255, 255))
        mask = cv2.morphologyEx(red | blue, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        y_lo = int(cfg.SIGN_ROI_TOP_FRAC * self.h)
        y_hi = int(cfg.SIGN_ROI_BOT_FRAC * self.h)
        max_area = cfg.SIGN_MAX_AREA_FRAC * self.w * self.h

        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.SIGN_MIN_AREA or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / h
            if not (cfg.SIGN_MIN_ASPECT <= aspect <= cfg.SIGN_MAX_ASPECT):
                continue
            cy = y + h // 2
            if cy < y_lo or cy > y_hi:
                continue
            solidity = area / max(1, w * h)
            if solidity < 0.45:                        # reject stringy, non-sign blobs
                continue
            boxes.append((x, y, x + w, y + h))
        return boxes

    # ------------------------------------------------------------------
    # Full recognition (proposal -> CNN -> temporal vote)
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> SignResult:
        if not self.available:
            return SignResult()

        dets: List[SignDetection] = []
        boxes = self.propose_regions(frame)
        if boxes:
            crops = []
            for (x1, y1, x2, y2) in boxes:
                pad = int(0.12 * (x2 - x1))
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(self.w, x2 + pad), min(self.h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size:
                    crops.append(sign_crop_to_tensor(crop, cfg.SIGN_INPUT_SIZE))
            if crops:
                with self._torch.no_grad():
                    batch = self._torch.stack(crops)
                    probs = self._torch.softmax(self.model(batch), dim=1)
                    conf, cls = probs.max(dim=1)
                for i, (box) in enumerate(boxes[:len(crops)]):
                    c = float(conf[i]); k = int(cls[i])
                    if c >= cfg.SIGN_MIN_CONF:
                        dets.append(SignDetection(GTSRB_CLASSES.get(k, str(k)), k, c, box))

        # temporal vote over the set of high-conf classes each frame
        self._recent.append({d.class_id for d in dets})
        tally = Counter(k for s in self._recent for k in s)
        recognised_ids = [k for k, n in tally.items() if n >= cfg.SIGN_VOTE_MIN]
        recognised = [GTSRB_CLASSES.get(k, str(k)) for k in recognised_ids]

        for k in recognised_ids:                        # latch newest speed limit
            if k in SPEED_LIMIT_KMH:
                self.speed_limit_kmh = SPEED_LIMIT_KMH[k]

        return SignResult(detections=dets, recognised=recognised,
                          speed_limit_kmh=self.speed_limit_kmh)

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, result: SignResult):
        for d in result.detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), cfg.COLOR_SIGN_BOX, 2)
            cv2.putText(frame, f"{d.label} {d.confidence:.0%}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg.COLOR_SIGN_BOX, 1, cv2.LINE_AA)
        if result.speed_limit_kmh is not None:
            self._draw_limit_chip(frame, result.speed_limit_kmh)
        return frame

    def _draw_limit_chip(self, frame: np.ndarray, kmh: int) -> None:
        """A little speed-limit roundel (red ring + number), top-left under the light chip."""
        cx, cy, r = 44, 78, 26
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), r, (0, 0, 220), 4)
        txt = str(kmh)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.putText(frame, txt, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
