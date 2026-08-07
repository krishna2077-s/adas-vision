"""
async_detector.py — Module 14: non-blocking object detection on a worker thread.

YOLOv8n on the iGPU finishes in ~28 ms (~35 fps) — already inside the safe
detection budget — but in the synchronous pipeline it only *runs* every ~50-70 ms
because lane, road, tracker, decision and drawing all execute before it on the
same thread. That stretches the detection interval into the "reduced-cadence"
band where Phase 17 has to ease off (extra CAUTION).

This wrapper runs the detector on a background thread so it churns continuously
on the iGPU (its OpenVINO C++ inference releases the GIL, so it genuinely overlaps
the main thread's CPU work). The main loop submits the newest frame and reads the
latest result plus its AGE — which it hands to the decision engine, so the
Phase-17 staleness fail-safe + reduced-cadence floor stay in charge of safety.
Newest-frame-wins: if the main loop outruns the detector, stale pending frames are
dropped, never queued — so latency can't build up.

Safety model: this changes only WHEN detections arrive, not the decision logic.
Correct detection_age_s + fresh-vs-coast handling is all it must get right; the
already-audited Phase 16/17 machinery does the rest. Advisory only.
"""

import logging
import threading
import time
from typing import Optional, Tuple

import config as cfg

logger = logging.getLogger(__name__)


class AsyncObjectDetector:
    """Runs an ObjectDetector on a worker thread; newest-frame-wins, non-blocking."""

    def __init__(self, frame_width: int, frame_height: int, detector=None) -> None:
        # `detector` is injectable for testing; production constructs the real one.
        if detector is not None:
            self._det = detector
        else:
            from object_detection import ObjectDetector
            self._det = ObjectDetector(frame_width, frame_height)
        self.backend = getattr(self._det, "backend", "?")

        self._cond = threading.Condition()
        self._in_frame = None          # newest submitted frame (clean copy)
        self._in_lane_x: Optional[float] = None
        self._in_seq = 0               # increments on every submit
        self._out = None               # (obj_result, seq, compute_time_monotonic)
        self._out_seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="yolo-async", daemon=True)
        self._thread.start()
        logger.info(f"AsyncObjectDetector started (worker thread, backend={self.backend}).")

    # ------------------------------------------------------------------
    def submit(self, frame, lane_center_x: Optional[float]) -> None:
        """Hand the worker the newest frame (a clean copy). Non-blocking; any
        frame still pending un-processed is overwritten (dropped)."""
        with self._cond:
            self._in_frame = frame.copy()      # worker needs clean pixels (main loop draws on its own)
            self._in_lane_x = lane_center_x
            self._in_seq += 1
            self._cond.notify()

    def latest(self) -> Tuple[Optional[object], Optional[float], bool]:
        """Return (obj_result, age_seconds, is_new_since_last_latest).

        obj_result is None until the first detection completes. age_seconds is the
        real time since that result was computed (what the decision engine uses as
        detection_age_s). is_new flags that a fresh result arrived since the caller
        last read one — so the caller updates the tracker on new results and coasts
        otherwise."""
        with self._cond:
            if self._out is None:
                return None, None, False
            res, seq, t = self._out
            is_new = seq != self._out_seq
            self._out_seq = seq
        return res, time.monotonic() - t, is_new

    def _worker(self) -> None:
        last = 0
        while True:
            with self._cond:
                while self._running and self._in_seq == last:
                    self._cond.wait(timeout=1.0)
                if not self._running:
                    return
                frame = self._in_frame
                lane_x = self._in_lane_x
                seq = self._in_seq
                last = seq
            try:
                res, _annotated = self._det.process(frame, lane_x)   # heavy: runs OFF the lock
                t = time.monotonic()
                with self._cond:
                    self._out = (res, seq, t)
            except Exception as exc:                                 # never let the worker die
                logger.debug(f"async detect skipped a frame ({exc}).")

    def annotate(self, frame, result):
        """Draw the given detection result's boxes onto frame (delegates to the
        wrapped detector). The result is ~one detection old, so boxes lag by the
        worker's latency — acceptable for the advisory overlay."""
        if result is None:
            return frame
        return self._det._annotate(frame, result, 0)

    def reset(self) -> None:
        """Clear pending/last results (e.g. when a video loops)."""
        with self._cond:
            self._in_frame = None
            self._out = None
            self._out_seq = 0

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=2.0)
