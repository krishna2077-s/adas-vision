"""
learned_road_detection.py — Module 1c: learned drivable-area road following.

The classical surface detector (road_detection.py) follows unmarked roads with
hand-tuned texture + colour + connectivity heuristics. Those heuristics break
in hazy vegetation, deep shadow, night, and on unusual surfaces. This module
replaces them with a small CNN — LRASPP MobileNetV3-Large, trained on the full
Indian Driving Dataset (IDD, ~20k images, val drivable-IoU 0.92) — that
segments the drivable road surface directly, then steers along its centreline.

It is the PRIMARY guidance source on unmarked roads. The pipeline is:

    Module 1  (paint lanes)      precise where markings exist
    Module 1c (this, learned)    unmarked roads — the accurate path
    Module 1b (classical road)   fallback when the model file is absent or the
                                 model finds no road

Output is a LaneDetectionResult (same dataclass as Modules 1 and 1b), so the
decision engine consumes it unchanged. Confidence is capped (LEARNED_CONF_CAP)
because monocular drivable-area guidance is honest but less precise than paint.

Runs on CPU via PyTorch — ~4 fps at 768x432 on a laptop i7, the same order as
YOLOv8n. Advisory / research only: never wire this to a vehicle's controls.
"""

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)

import config as cfg
from lane_detection import LaneDetectionResult

logger = logging.getLogger(__name__)

# ImageNet normalisation — must match how the model was trained (phase6c).
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class LearnedRoadDetector:
    """
    Learned drivable-area detector for unmarked roads.

    Usage::

        road = LearnedRoadDetector(frame_width=1280, frame_height=720)
        if road.available:
            result, annotated = road.process(frame, draw_on=annotated)

    If the model file is missing (or fails to load) the detector constructs
    successfully but reports ``available == False`` — the caller then falls
    back to the classical RoadDetector instead of crashing.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        model_path: Optional[str] = None,
    ) -> None:
        self.w = frame_width
        self.h = frame_height
        self.model_path = model_path or cfg.LEARNED_MODEL_PATH
        self._smooth_look_x: Optional[float] = None
        self.model: Optional[nn.Module] = None
        self.session = None                       # ONNX Runtime session (if backend=onnx)
        self._onnx_input = None
        self.backend = "torch"
        self.available = False

        # Frame-skip state: run the CNN every Nth frame, reuse the mask between.
        self._last_mask: Optional[np.ndarray] = None
        self._skip_ctr = 0

        if cfg.LEARNED_NUM_THREADS > 0:
            torch.set_num_threads(cfg.LEARNED_NUM_THREADS)

        self._load_model()

    def _build_arch(self) -> nn.Module:
        """Builds the (untrained) architecture named by cfg.LEARNED_ARCH.

        weights=None / weights_backbone=None: we load our own full state_dict,
        so torchvision's pretrained downloads are skipped (works offline). The
        architecture must match the weights file.
        """
        arch = cfg.LEARNED_ARCH.lower()
        if arch == "deeplabv3":                        # Phase 7 — richer ASPP head
            return deeplabv3_mobilenet_v3_large(
                weights=None, weights_backbone=None, num_classes=2, aux_loss=True)
        if arch == "lraspp":                           # Phase 6c — lightweight
            model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
            model.classifier.low_classifier = nn.Conv2d(40, 2, 1)
            model.classifier.high_classifier = nn.Conv2d(128, 2, 1)
            return model
        raise ValueError(f"unknown LEARNED_ARCH '{cfg.LEARNED_ARCH}' (use 'lraspp' or 'deeplabv3')")

    def _load_model(self) -> None:
        """Loads the requested backend (onnx → torch → unavailable), in order."""
        if getattr(cfg, "LEARNED_BACKEND", "torch").lower() == "onnx" and self._load_onnx():
            return
        self._load_torch()

    def _load_onnx(self) -> bool:
        """Tries to bring up the ONNX Runtime backend. Returns True on success."""
        onnx_path = getattr(cfg, "LEARNED_ONNX_PATH", "")
        if not onnx_path or not os.path.exists(onnx_path):
            logger.info(f"ONNX model '{onnx_path}' not found — trying PyTorch backend.")
            return False
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if cfg.LEARNED_NUM_THREADS > 0:
                so.intra_op_num_threads = cfg.LEARNED_NUM_THREADS
            self.session = ort.InferenceSession(onnx_path, sess_options=so,
                                                providers=["CPUExecutionProvider"])
            self._onnx_input = self.session.get_inputs()[0].name
            self.backend = "onnx"
            self.available = True
            logger.info(
                f"LearnedRoadDetector ready (ONNX) — '{onnx_path}', "
                f"{cfg.LEARNED_INPUT_W}x{cfg.LEARNED_INPUT_H} input, "
                f"infer every {cfg.LEARNED_INFER_EVERY} frame(s)"
            )
            return True
        except Exception as exc:
            logger.warning(f"ONNX backend failed to load ({exc}) — trying PyTorch backend.")
            return False

    def _load_torch(self) -> None:
        try:
            model = self._build_arch()
            state = torch.load(self.model_path, map_location="cpu")
            model.load_state_dict(state)
            # DeepLabV3's aux head is training-only — drop it to save CPU at inference.
            if cfg.LEARNED_ARCH.lower() == "deeplabv3" and getattr(model, "aux_classifier", None) is not None:
                model.aux_classifier = None
            model.eval()
            self.model = model
            self.backend = "torch"
            self.available = True
            logger.info(
                f"LearnedRoadDetector ready (PyTorch) ({self.w}x{self.h}) — arch '{cfg.LEARNED_ARCH}', "
                f"weights '{self.model_path}', {cfg.LEARNED_INPUT_W}x{cfg.LEARNED_INPUT_H} input, "
                f"{torch.get_num_threads()} threads, infer every {cfg.LEARNED_INFER_EVERY} frame(s)"
            )
        except FileNotFoundError:
            logger.warning(
                f"Learned road model '{self.model_path}' not found — "
                f"falling back to the classical road detector."
            )
        except Exception as exc:  # corrupt/mismatched checkpoint, etc.
            logger.warning(f"Learned road model failed to load ({exc}) — using classical fallback.")

    def reset(self) -> None:
        self._smooth_look_x = None
        self._last_mask = None
        self._skip_ctr = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        draw_on: Optional[np.ndarray] = None,
    ) -> Tuple[LaneDetectionResult, np.ndarray]:
        """
        Segments the drivable road surface and returns steering guidance.

        Args:
            frame:   RAW BGR frame — inference runs on clean pixels.
            draw_on: optional frame to draw the overlay onto (e.g. one already
                     annotated by other modules). Defaults to `frame`.

        Returns:
            (LaneDetectionResult, annotated_frame). confidence is > 0 (scaled by
            coverage, capped at LEARNED_CONF_CAP) when the road is locked, 0.0
            when the model is unavailable or no reliable road was found.
        """
        result = LaneDetectionResult(frame_center_x=self.w // 2)
        canvas = draw_on if draw_on is not None else frame

        if not self.available:
            result.confidence = 0.0
            result.steering = "UNKNOWN -- learned model unavailable"
            return result, canvas

        # Frame-skip: only run the CNN every Nth frame; reuse the mask between.
        # The centreline/steering below still recompute every frame, so guidance
        # stays smooth while the expensive inference runs ~1/N as often.
        every = max(1, getattr(cfg, "LEARNED_INFER_EVERY", 1))
        if self._last_mask is None or self._skip_ctr <= 0:
            self._last_mask = self._infer_mask(frame)
            self._skip_ctr = every - 1
        else:
            self._skip_ctr -= 1
        mask = self._last_mask
        centre, coverage = self._centreline(mask)

        if coverage >= cfg.LEARNED_MIN_COVERAGE and len(centre) >= cfg.LEARNED_MIN_ROWS:
            look = centre[min(len(centre) - 1, int(len(centre) * cfg.LEARNED_LOOKAHEAD_FRAC))]
            look_x = float(look[1])

            if self._smooth_look_x is None:
                self._smooth_look_x = look_x
            else:
                a = cfg.LEARNED_SMOOTHING_ALPHA
                self._smooth_look_x = a * look_x + (1 - a) * self._smooth_look_x

            cx = int(self._smooth_look_x)
            offset = cx - result.frame_center_x
            result.lane_center_x = cx
            result.offset_px = offset
            result.confidence = min(cfg.LEARNED_CONF_CAP, 0.5 + 0.4 * coverage)
            result.steering = self._steering(offset)
            annotated = self._annotate(canvas, mask, centre, cx, offset)
        else:
            # Not enough road found — say so; the caller drops to Module 1b.
            self._smooth_look_x = None
            result.confidence = 0.0
            result.steering = "UNKNOWN -- no road surface lock"
            annotated = canvas

        return result, annotated

    # ------------------------------------------------------------------
    # Inference + geometry
    # ------------------------------------------------------------------

    def _infer_mask(self, frame: np.ndarray) -> np.ndarray:
        """Runs the CNN (torch or onnx) and returns a full-res binary mask (0/255)."""
        img = cv2.resize(frame, (cfg.LEARNED_INPUT_W, cfg.LEARNED_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]      # (1, 3, H, W)

        if self.backend == "onnx":
            out = self.session.run(None, {self._onnx_input: x})[0]   # (1, 2, H, W)
            pred = out.argmax(1).squeeze(0).astype(np.uint8)
        else:
            with torch.no_grad():
                out = self.model(torch.from_numpy(x))["out"]
            pred = out.argmax(1).squeeze(0).numpy().astype(np.uint8)

        mask = cv2.resize(pred, (self.w, self.h), interpolation=cv2.INTER_NEAREST) * 255
        return mask

    def _centreline(self, mask: np.ndarray) -> Tuple[List[Tuple[int, int, int]], float]:
        """
        Per-row centre of the drivable mask -> a CURVED centreline, so bends and
        hills are followed naturally. Returns (centreline [(y, x, width)...],
        row-coverage 0-1). Rows are scanned bottom-up within the road band.
        """
        top = int(cfg.LEARNED_HORIZON_FRAC * self.h)
        bottom = int(cfg.LEARNED_BONNET_FRAC * self.h)
        centre: List[Tuple[int, int, int]] = []
        rows = 0
        for y in range(bottom - 1, top, -6):
            rows += 1
            xs = np.where(mask[y] > 0)[0]
            if len(xs) > 16:      # ignore a few stray pixels
                centre.append((y, int((xs.min() + xs.max()) / 2), int(xs.max() - xs.min())))
        coverage = len(centre) / max(1, rows)
        return centre, coverage

    # ------------------------------------------------------------------
    # Steering + drawing
    # ------------------------------------------------------------------

    def _steering(self, offset: int) -> str:
        a = abs(offset)
        direction = "RIGHT" if offset > 0 else "LEFT"
        if a < cfg.STEER_THRESHOLD_SLIGHT:
            return "STRAIGHT (learned)"
        elif a < cfg.STEER_THRESHOLD_MODERATE:
            return f"SLIGHT {direction} (learned)"
        elif a < cfg.STEER_THRESHOLD_HARD:
            return f"MODERATE {direction} (learned)"
        return f"HARD {direction} (learned)"

    def _annotate(self, frame, mask, centre, look_x, offset) -> np.ndarray:
        out = frame.copy()

        # Translucent drivable-surface fill (distinct colour from Module 1b)
        fill = np.zeros_like(out)
        fill[mask > 0] = cfg.LEARNED_FILL_COLOR
        out = cv2.addWeighted(out, 1.0, fill, cfg.LEARNED_FILL_ALPHA, 0)

        # Curved centreline
        pts = [(x, y) for (y, x, _wd) in centre]
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], cfg.COLOR_CENTER_LINE, 3, cv2.LINE_AA)

        # Look-ahead marker + frame-centre reference
        cx = self.w // 2
        cv2.line(out, (cx, self.h), (cx, int(0.5 * self.h)), (255, 255, 255), 1, cv2.LINE_AA)
        y_marker = int(0.62 * self.h)
        cv2.circle(out, (look_x, y_marker), 8, cfg.COLOR_CENTER_LINE, -1)
        cv2.line(out, (cx, y_marker), (look_x, y_marker), cfg.COLOR_CENTER_LINE, 2)

        # Mode tag so the driver knows this is the learned model, not paint
        cv2.putText(out, "LEARNED ROAD (IDD)", (14, self.h - 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cfg.LEARNED_FILL_COLOR, 2, cv2.LINE_AA)
        return out
