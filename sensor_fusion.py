"""
sensor_fusion.py — Module 6 (Layer 1, SIMULATED sensor fusion).

Real ADAS fuse a camera (rich classification, noisy monocular depth) with radar
(precise range + range-rate, no class). This module demonstrates that fusion
architecture — but the radar here is SIMULATED: there is no radar hardware, so
we synthesise a radar return from the camera's own ego objects plus noise, then
fuse the two with an inverse-variance (Bayesian) combine.

Why it's still worth building: it shows the *structure* of fusion — how you take
a precise-but-blind sensor and a rich-but-noisy one and get an estimate better
than either alone. Swap the synthetic radar for a real driver and the maths is
unchanged. It does NOT add real-world information (the "radar" comes from the
camera), and it is clearly labelled SIM everywhere. Advisory / research only.
"""

import logging
from typing import List

import numpy as np

import config as cfg
from perception_bev import EgoObject

logger = logging.getLogger(__name__)


class SimRadarFusion:
    """Synthesises a radar measurement per object and fuses it with the camera."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    @staticmethod
    def _fuse(cam_val: float, cam_var: float, radar_val: float, radar_var: float):
        """Inverse-variance combine of two measurements -> (estimate, variance)."""
        wc, wr = 1.0 / cam_var, 1.0 / radar_var
        est = (cam_val * wc + radar_val * wr) / (wc + wr)
        var = 1.0 / (wc + wr)
        return est, var

    def process(self, ego_objs: List[EgoObject]) -> List[EgoObject]:
        """Populates radar_z_m / fused_z_m / fused_vz_mps on each ego object."""
        for o in ego_objs:
            # --- simulated radar return (range + range-rate, with noise) ------
            radar_z = max(0.1, o.z_m + float(self._rng.normal(0.0, cfg.RADAR_RANGE_NOISE_M)))
            radar_vz = o.vz_mps + float(self._rng.normal(0.0, cfg.RADAR_VEL_NOISE_MPS))
            o.radar_z_m = radar_z

            # --- fuse camera depth (noisy) with radar range (precise) --------
            o.fused_z_m, _ = self._fuse(
                o.z_m, cfg.FUSION_CAM_RANGE_VAR, radar_z, cfg.FUSION_RADAR_RANGE_VAR)
            # radar range-rate is far better than differenced camera depth
            o.fused_vz_mps, _ = self._fuse(
                o.vz_mps, cfg.FUSION_CAM_RANGE_VAR, radar_vz, cfg.FUSION_RADAR_RANGE_VAR)
        return ego_objs

    def draw_label(self, frame):
        """Small tag so it's clear the radar/fusion is simulated."""
        import cv2
        H = frame.shape[0]
        cv2.putText(frame, "SIM RADAR + FUSION", (14, H - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg.COLOR_BEV_RADAR, 1, cv2.LINE_AA)
        return frame
