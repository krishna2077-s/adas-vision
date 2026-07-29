"""
drive_logger.py — Module 12: the "black box" drive recorder.

Every real ADAS keeps a flight-recorder log so a drive can be reconstructed and
audited afterwards (openpilot's rlog is the reference). This is that, in a plain,
greppable JSON-Lines file: one header line describing the run, then one compact
record per frame capturing what the system SAW (tracked objects) and DECIDED
(decision engine, FCW, scene, plan, sim control).

The point is reconstruction: replay_log.py re-renders the whole drive from this
file alone — no models, no video decode of the network — so you can answer "what
did it see and decide at 00:42, and why?" long after the run. It also feeds the
evaluation harness (decision-rate stats, ID-switch counts, etc.).

Design:
    - Append-only JSON Lines (.jsonl): survives a crash mid-drive (every line is
      flushed), diffs cleanly, and needs no schema library to read.
    - Compact keys, rounded floats — a full drive stays small.
    - Tolerant: every field is pulled with getattr/defaults, so a disabled
      module simply logs nulls instead of breaking the recorder.

Opt-in via `python main.py --video x.mp4 --log drive.jsonl`. Advisory system —
the log records advice, not vehicle actions (there are none).
"""

import json
import logging
from typing import List, Optional

import cv2

import config as cfg

logger = logging.getLogger(__name__)

SCHEMA = 1


def _round(v, n=1):
    return round(v, n) if isinstance(v, (int, float)) else v


class DriveLogger:
    """Writes one header + one JSON record per frame. Flushes each line."""

    def __init__(self, path: str, source, frame_width: int, frame_height: int,
                 fps: float) -> None:
        self.path = path
        self.frames = 0
        self._f = open(path, "w", encoding="utf-8")
        self._t0 = cv2.getTickCount()
        header = {
            "type": "header", "schema": SCHEMA,
            "source": str(source), "w": frame_width, "h": frame_height, "fps": round(fps, 2),
            "config": {
                "learned_backend": getattr(cfg, "LEARNED_BACKEND", None),
                "ov_device": getattr(cfg, "LEARNED_OV_DEVICE", None),
                "yolo_imgsz": getattr(cfg, "YOLO_IMGSZ", None),
                "infer_every": getattr(cfg, "LEARNED_INFER_EVERY", None),
                "ttc_brake_s": getattr(cfg, "TTC_BRAKE_S", None),
                "fcw_ttc_imminent_s": getattr(cfg, "FCW_TTC_IMMINENT_S", None),
            },
        }
        self._write(header)
        logger.info(f"DriveLogger recording to {path}")

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._f.flush()

    def _t_ms(self) -> int:
        return int((cv2.getTickCount() - self._t0) / cv2.getTickFrequency() * 1000)

    def log(self, frame_index: int, decision, tracks: Optional[List] = None,
            fcw=None, light_state: Optional[str] = None,
            speed_limit_kmh: Optional[int] = None, plan=None, control=None) -> None:
        """Append one frame record. All arguments except frame_index are optional."""
        rec = {"type": "frame", "i": frame_index, "t": self._t_ms()}

        if decision is not None:
            rec["dec"] = {
                "lon": getattr(decision, "longitudinal", None),
                "lvl": getattr(decision, "longitudinal_level", None),
                "lat": getattr(decision, "lateral", None),
                "rule": getattr(decision, "rule_id", None),
                "valid": getattr(decision, "valid", None),
                "degraded": getattr(decision, "degraded", None),
                "hz": getattr(decision, "hazard_label", None),
                "hid": getattr(decision, "hazard_id", None),
                "d": _round(getattr(decision, "smoothed_distance_m", None)),
                "ttc": _round(getattr(decision, "ttc_s", None)),
                "vc": _round(getattr(decision, "closing_speed_mps", None)),
                "fps": _round(getattr(decision, "fps", None)),
                "brake": _round(getattr(decision, "brake", None), 2),
            }

        if fcw is not None:
            rec["fcw"] = {"lvl": getattr(fcw, "level", None),
                          "name": getattr(fcw, "name", None),
                          "ttc": _round(getattr(fcw, "ttc_s", None))}

        if light_state is not None:
            rec["light"] = light_state
        if speed_limit_kmh is not None:
            rec["limit"] = speed_limit_kmh

        if plan is not None:
            rec["plan"] = {"tgt": _round(getattr(plan, "target_speed_mps", None)),
                           "lead": getattr(plan, "lead_id", None),
                           "gap": _round(getattr(plan, "lead_gap_m", None)),
                           "why": getattr(plan, "reason", None)}

        if control is not None:
            rec["ctl"] = {"steer": _round(getattr(control, "steering_deg", None)),
                          "thr": _round(getattr(control, "throttle", None), 2),
                          "brk": _round(getattr(control, "brake", None), 2),
                          "spd": _round(getattr(control, "sim_speed_mps", None))}

        if tracks:
            rec["trk"] = [
                {"id": t.id, "lab": t.label,
                 "box": [int(t.x1), int(t.y1), int(t.x2), int(t.y2)],
                 "d": _round(getattr(t, "smoothed_distance_m", None)),
                 "ip": bool(getattr(t, "in_path", False)),
                 "rk": getattr(t, "risk", None)}
                for t in tracks if getattr(t, "confirmed", False)
            ]

        self._write(rec)
        self.frames += 1

    def close(self) -> None:
        try:
            self._write({"type": "footer", "frames": self.frames})
            self._f.close()
            logger.info(f"DriveLogger closed ({self.frames} frames) -> {self.path}")
        except Exception:
            pass
