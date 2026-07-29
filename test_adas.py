"""
test_adas.py — regression tests: invariants that must never silently break.

These are the safety-relevant contracts the system is built on. A future change
(a threshold tweak, a model swap, a refactor) that violates one of them should
fail loudly here instead of shipping. Runs with plain asserts — no pytest
required — but the test_* functions are pytest-discoverable too:

    python test_adas.py          # runs all, prints PASS/FAIL/SKIP, exits nonzero on fail
    pytest test_adas.py          # same tests under pytest

Covered:
  - config ordering (the thresholds the whole policy depends on)
  - decision engine: clear -> PROCEED; a sustained closing hazard -> emergency;
    a single spurious frame never reaches BRAKE (temporal ratchet)
  - FCW stage machine: monotonic escalation + one-level-at-a-time release
  - tracker: stable IDs + one-frame ghosts never confirm
  - traffic-light state: RED/GREEN read correctly, dark -> UNKNOWN
  - deployed road backend parity (OpenVINO vs ONNX) — guards model/quant swaps
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config as cfg


class Skip(Exception):
    """Raised by a test when its optional dependency/data is absent."""


# ---------------------------------------------------------------------------
# config ordering
# ---------------------------------------------------------------------------

def test_config_ordering():
    # TTC bands must escalate correctly: emergency < brake < caution
    assert cfg.TTC_EMERGENCY_S < cfg.TTC_BRAKE_S < cfg.TTC_CAUTION_S
    # FCW warning bands likewise
    assert cfg.FCW_TTC_IMMINENT_S < cfg.FCW_TTC_WARN_S < cfg.FCW_TTC_CAUTION_S
    # risk-by-distance ordering
    assert cfg.RISK_DISTANCE_HIGH < cfg.RISK_DISTANCE_MEDIUM
    # the degraded-trust gate must sit ABOVE the YOLO keep threshold, or it can
    # never trip (YOLO already filtered everything below its own threshold)
    assert cfg.DET_CONF_MIN > cfg.YOLO_CONF_THRESHOLD
    # evidence window must cover the largest escalation M
    max_m = max(m for _n, m in (cfg.ESC_EMERGENCY, cfg.ESC_BRAKE, cfg.ESC_SLOW, cfg.ESC_CAUTION))
    assert cfg.EVIDENCE_WINDOW >= max_m


# ---------------------------------------------------------------------------
# decision engine
# ---------------------------------------------------------------------------

@dataclass
class FakeTrack:
    id: int = 1
    label: str = "car"
    confirmed: bool = True
    in_path: bool = True
    smoothed_distance_m: Optional[float] = 30.0
    closing_speed_mps: float = 0.0
    ttc_s: Optional[float] = None
    risk: str = "LOW"
    confidence: float = 0.9
    center: tuple = (640, 400)


def _engine():
    from decision_engine import DecisionEngine
    return DecisionEngine(frame_width=1280, frame_height=720)


def test_decision_clear_is_proceed():
    eng = _engine()
    d = None
    for _ in range(3):
        d = eng.process(None, [])           # a clear scene
    assert d.longitudinal == "PROCEED", d.longitudinal


def test_decision_closing_hazard_emergency():
    eng = _engine()
    hz = FakeTrack(id=7, label="car", in_path=True, smoothed_distance_m=4.0,
                   closing_speed_mps=8.0, ttc_s=1.0, risk="HIGH", center=(640, 500))
    d = None
    for _ in range(5):                      # sustained -> ratchet confirms
        d = eng.process(None, [hz])
    assert d.longitudinal_level >= 3, f"expected BRAKE/EMERGENCY, got {d.longitudinal}"
    assert d.hazard_id == 7


def test_decision_single_spurious_frame_ignored():
    eng = _engine()
    clear = []
    hot = [FakeTrack(id=9, smoothed_distance_m=4.0, closing_speed_mps=9.0, ttc_s=0.9,
                     risk="HIGH", center=(640, 520))]
    eng.process(None, clear)
    eng.process(None, clear)
    d = eng.process(None, hot)              # ONE hot frame amid calm
    d = eng.process(None, clear)
    # ESC_BRAKE is 3-of-5: a single hot frame must not have reached BRAKE
    assert d.longitudinal_level < 3, f"one frame reached {d.longitudinal}"


# ---------------------------------------------------------------------------
# FCW stage machine
# ---------------------------------------------------------------------------

@dataclass
class FakeDecision:
    ttc_s: Optional[float] = None
    smoothed_distance_m: Optional[float] = None
    closing_speed_mps: float = 0.0
    hazard_label: Optional[str] = None
    hazard_id: Optional[int] = None


def test_fcw_monotonic_and_hysteretic():
    from forward_collision_warning import ForwardCollisionWarning
    fcw = ForwardCollisionWarning(1280, 720)
    seq, dist, closing = [], 40.0, 10.0
    while dist > 3.0:
        dist -= closing * 0.1
        ttc = dist / closing
        st = fcw.process(FakeDecision(round(ttc, 1), round(dist, 1), closing, "car", 7))
        seq.append(st.level)
    assert set(seq) == {0, 1, 2, 3}, f"did not reach all stages: {sorted(set(seq))}"
    assert seq == sorted(seq), "stage decreased during a monotonic approach"
    # clear -> must step down one level at a time, never jump to 0
    decay = [fcw.process(FakeDecision()).level for _ in range(40)]
    steps = [decay[i - 1] - decay[i] for i in range(1, len(decay)) if decay[i] != decay[i - 1]]
    assert steps and all(s == 1 for s in steps), f"release not one-level-at-a-time: {steps}"
    assert decay[-1] == 0


# ---------------------------------------------------------------------------
# tracker
# ---------------------------------------------------------------------------

def _det(x1, y1, x2, y2, label="car", conf=0.9):
    from object_detection import Detection
    d = Detection(label=label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
    d.distance_m = 25.0
    d.in_path = True
    d.risk = "LOW"
    return d


def test_tracker_stable_id_and_confirmation():
    from tracker import MultiObjectTracker
    trk = MultiObjectTracker()
    ids = []
    for k in range(cfg.TRACK_MIN_HITS + 2):
        tracks = trk.update([_det(600 + k * 3, 400, 700 + k * 3, 520)])  # drifts slowly
        conf = [t for t in tracks if t.confirmed]
        if conf:
            ids.append(conf[0].id)
    assert ids, "track never confirmed"
    assert len(set(ids)) == 1, f"ID switched across frames: {ids}"


def test_tracker_one_frame_ghost_never_confirms():
    from tracker import MultiObjectTracker
    trk = MultiObjectTracker()
    trk.update([_det(100, 100, 160, 200)])   # appears once
    for _ in range(cfg.TRACK_MAX_AGE + 2):    # then never again
        tracks = trk.update([])
    assert all(not t.confirmed for t in tracks), "a one-frame ghost confirmed"


# ---------------------------------------------------------------------------
# traffic-light state
# ---------------------------------------------------------------------------

def test_traffic_light_states():
    from traffic_light_state import TrafficLightReader, RED, GREEN, UNKNOWN
    import cv2
    reader = TrafficLightReader()

    def bulb(state):
        img = np.full((66, 26, 3), 22, np.uint8)
        slot = {"R": 11, "G": 55}[state]
        color = {"R": (40, 40, 255), "G": (40, 230, 40)}[state]
        cv2.circle(img, (13, slot), 8, color, -1)
        return img

    assert reader._classify(bulb("R"))[0] == RED
    assert reader._classify(bulb("G"))[0] == GREEN
    assert reader._classify(np.full((66, 26, 3), 18, np.uint8))[0] == UNKNOWN


# ---------------------------------------------------------------------------
# deployed road backend parity (guards model / quantisation swaps)
# ---------------------------------------------------------------------------

def test_road_backend_parity():
    import os, cv2
    if cfg.LEARNED_BACKEND != "openvino":
        raise Skip("backend is not openvino")
    if not os.path.exists(cfg.LEARNED_ONNX_PATH):
        raise Skip("reference ONNX absent")
    try:
        import onnxruntime as ort
        from learned_road_detection import LearnedRoadDetector
    except Exception as exc:
        raise Skip(str(exc))

    cap = cv2.VideoCapture("dashcam.mp4")
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise Skip("dashcam.mp4 unavailable")

    det = LearnedRoadDetector(frame_width=1280, frame_height=720)
    if not det.available or det.backend != "openvino":
        raise Skip("openvino backend did not load")
    ov_mask = det._infer_mask(frame) > 0

    # ONNX reference at the same input
    img = cv2.resize(frame, (cfg.LEARNED_INPUT_W, cfg.LEARNED_INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = np.ascontiguousarray(((img / 255.0 - mean) / std).transpose(2, 0, 1))[None].astype(np.float32)
    sess = ort.InferenceSession(cfg.LEARNED_ONNX_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    onnx_mask = cv2.resize(out.argmax(1).squeeze(0).astype(np.uint8), (1280, 720),
                           interpolation=cv2.INTER_NEAREST) > 0

    agree = float((ov_mask == onnx_mask).mean())
    assert agree >= 0.99, f"OpenVINO vs ONNX drivable masks agree only {agree:.4f}"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _all_tests():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("test_") and callable(g[n])]


def main() -> int:
    passed = failed = skipped = 0
    for name, fn in _all_tests():
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Skip as s:
            print(f"  SKIP  {name}  ({s})")
            skipped += 1
        except AssertionError as a:
            print(f"  FAIL  {name}  -- {a}")
            failed += 1
        except Exception as e:  # unexpected error is a failure
            print(f"  ERROR {name}  -- {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
