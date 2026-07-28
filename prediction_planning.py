"""
prediction_planning.py — Module 7 (Layer 4): prediction + advisory planning.

Prediction: each ego object is rolled forward with a constant-velocity model
over PRED_HORIZON_S, giving a short predicted trajectory in the ego frame
(drawn on the bird's-eye panel).

Planning (ADVISORY): from the predicted scene + the lane centre it produces a
suggested cruise speed (time-gap / following-distance logic) and an advisory
path ribbon along the lane. It COMMANDS nothing — the simulated controller
(Module 8) turns the suggestion into simulated steering/pedal, and even that
never touches a vehicle. Advisory / research only.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from perception_bev import EgoObject

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    target_speed_mps: float = cfg.PLAN_CRUISE_MPS
    predictions: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    path_pts: List[Tuple[int, int]] = field(default_factory=list)
    lead_id:  Optional[int]   = None
    lead_gap_m: Optional[float] = None
    reason:   str = "cruise"


class Planner:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self.cx = frame_width // 2

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    def _predict(self, ego_objs: List[EgoObject]) -> Dict[int, List[Tuple[float, float]]]:
        preds: Dict[int, List[Tuple[float, float]]] = {}
        ts = np.arange(cfg.PRED_STEP_S, cfg.PRED_HORIZON_S + 1e-6, cfg.PRED_STEP_S)
        for o in ego_objs:
            z0 = o.fused_z_m if o.fused_z_m is not None else o.z_m
            vz = o.fused_vz_mps if o.fused_vz_mps is not None else o.vz_mps
            pts = []
            for t in ts:
                z = z0 + vz * t
                if z <= 0:
                    break
                pts.append((o.x_m + o.vx_mps * t, z))
            if pts:
                preds[o.id] = pts
        return preds

    def _lane_target_x(self, lane_result) -> int:
        if lane_result is not None and getattr(lane_result, "confidence", 0.0) > 0.0 \
                and lane_result.lane_center_x is not None:
            return int(lane_result.lane_center_x)
        return self.cx

    def _path_ribbon(self, lane_x: int) -> List[Tuple[int, int]]:
        """A simple advisory path from the car up to the lane target."""
        start = (self.cx, self.h - 1)
        end = (lane_x, int(0.55 * self.h))
        pts = []
        for i in range(9):
            f = i / 8.0
            x = int(start[0] + (end[0] - start[0]) * f)
            y = int(start[1] + (end[1] - start[1]) * f)
            pts.append((x, y))
        return pts

    def process(self, ego_objs: List[EgoObject], lane_result,
                ego_speed: Optional[float] = None) -> PlanResult:
        ego_speed = cfg.PLAN_CRUISE_MPS if ego_speed is None else ego_speed
        preds = self._predict(ego_objs)

        # nearest in-path lead (by fused forward distance)
        leads = [(o.fused_z_m if o.fused_z_m is not None else o.z_m, o)
                 for o in ego_objs if o.in_path]
        target = cfg.PLAN_CRUISE_MPS
        lead_id = lead_gap = None
        reason = "cruise"
        if leads:
            gap, lead = min(leads, key=lambda p: p[0])
            lead_id, lead_gap = lead.id, gap
            desired_gap = max(cfg.PLAN_MIN_GAP_M, cfg.PLAN_TIME_GAP_S * ego_speed)
            if gap <= cfg.PLAN_MIN_GAP_M:
                target, reason = 0.0, f"stop: lead #{lead.id} at {gap:.0f}m"
            elif gap < desired_gap:
                target = cfg.PLAN_CRUISE_MPS * max(0.0, min(1.0, gap / desired_gap))
                reason = f"follow #{lead.id} @ {gap:.0f}m (gap<{desired_gap:.0f}m)"

        return PlanResult(
            target_speed_mps=target, predictions=preds,
            path_pts=self._path_ribbon(self._lane_target_x(lane_result)),
            lead_id=lead_id, lead_gap_m=lead_gap, reason=reason,
        )

    # ------------------------------------------------------------------
    def draw_overlay(self, frame, plan: PlanResult):
        """Advisory path ribbon + a compact plan line on the main image."""
        # path ribbon, coloured by target speed (green fast -> red slow)
        if len(plan.path_pts) >= 2:
            f = max(0.0, min(1.0, plan.target_speed_mps / max(1e-3, cfg.PLAN_CRUISE_MPS)))
            col = (0, int(255 * f), int(255 * (1 - f)))
            cv2.polylines(frame, [np.array(plan.path_pts, np.int32)], False, col, 3, cv2.LINE_AA)
            cv2.circle(frame, plan.path_pts[-1], 5, col, -1)

        kmh = plan.target_speed_mps * 3.6
        txt = f"PLAN  {kmh:4.0f} km/h  |  {plan.reason}"
        cv2.putText(frame, txt, (14, self.h - 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, cfg.COLOR_PLAN_PATH, 2, cv2.LINE_AA)
        return frame
