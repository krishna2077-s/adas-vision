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
# decision engine — deeper safety invariants
# ---------------------------------------------------------------------------

@dataclass
class FakeLane:
    """Minimal stand-in for a LaneDetectionResult (only fields the engine reads)."""
    confidence: float = 1.0
    offset_px: int = 0
    lane_center_x: Optional[int] = 640


def _drive_to_emergency(eng):
    hz = FakeTrack(id=3, smoothed_distance_m=4.0, closing_speed_mps=9.0, ttc_s=0.9,
                   risk="HIGH", center=(640, 520))
    d = None
    for _ in range(6):
        d = eng.process(None, [hz])
    return d


def test_decision_emergency_latches():
    """Once EMERGENCY, a few clear frames must NOT release it (safety latch)."""
    eng = _engine()
    d = _drive_to_emergency(eng)
    assert d.longitudinal_level == 4, f"did not reach EMERGENCY: {d.longitudinal}"
    for _ in range(3):                       # hazard vanishes for 3 frames (< latch)
        d = eng.process(None, [])
    assert d.longitudinal_level == 4, f"emergency released after only 3 clear frames ({d.longitudinal})"


def test_decision_release_is_gradual_and_reaches_clear():
    """De-escalation is one level at a time, and the system does eventually clear."""
    eng = _engine()
    _drive_to_emergency(eng)
    levels = [eng.process(None, []).longitudinal_level for _ in range(120)]
    for a, b in zip(levels, levels[1:]):
        assert b >= a - 1, f"committed level dropped {a}->{b} (more than one step in a frame)"
    assert levels[-1] == 0, f"never returned to PROCEED: ended at {levels[-1]}"


def test_decision_degraded_floor_never_proceed_with_hazard():
    """Blind (no lane lock) + a hazard present must never read PROCEED."""
    eng = _engine()
    benign = FakeTrack(id=5, label="car", in_path=True, smoothed_distance_m=30.0,
                       closing_speed_mps=0.0, ttc_s=None, risk="LOW", center=(640, 400))
    d = None
    for _ in range(4):                       # lane_result=None -> degraded/object-only
        d = eng.process(None, [benign])
    assert d.degraded, "expected degraded with lane input absent"
    assert d.longitudinal_level >= 1, "degraded with a hazard present but still PROCEED"


def test_decision_advisory_eases_never_brakes():
    """A stop sign / light in-path eases to SLOW — it must never trigger a hard brake."""
    eng = _engine()
    sign = FakeTrack(id=6, label="stop sign", in_path=True, smoothed_distance_m=4.0,
                     closing_speed_mps=0.0, ttc_s=None, risk="HIGH", center=(640, 300))
    levels = [eng.process(FakeLane(1.0, 0), [sign]).longitudinal_level for _ in range(6)]
    assert max(levels) <= 2, f"an advisory control triggered level {max(levels)} (>= BRAKE)"
    assert levels[-1] == 2, f"advisory did not settle at SLOW: {levels}"


def test_decision_lateral_inhibited_toward_hazard():
    """When braking, a lane correction toward the hazard's side is suppressed to HOLD."""
    eng = _engine()
    hz = FakeTrack(id=8, label="car", in_path=True, smoothed_distance_m=7.0,
                   closing_speed_mps=0.0, ttc_s=None, risk="HIGH", center=(560, 500))  # left of centre
    lane = FakeLane(confidence=1.0, offset_px=60)   # offset>0 -> CORRECT_LEFT, toward the hazard
    d = None
    for _ in range(6):
        d = eng.process(lane, [hz])
    assert d.longitudinal_level == 3, f"expected BRAKE, got {d.longitudinal}"
    assert d.lateral == "HOLD", f"lateral not inhibited toward hazard: {d.lateral}"


def test_decision_vulnerable_brakes_earlier_than_vehicle():
    """At an identical TTC, a pedestrian must brake earlier than a car (vulnerable margin)."""
    def run(label):
        eng = _engine()
        hz = FakeTrack(id=1, label=label, in_path=True, smoothed_distance_m=15.0,
                       closing_speed_mps=5.0, ttc_s=3.0, risk="LOW", center=(640, 460))
        d = None
        for _ in range(6):
            d = eng.process(FakeLane(1.0, 0), [hz])
        return d.longitudinal_level
    ped, car = run("person"), run("car")
    assert ped >= 3 > car, f"vulnerable margin missing: person={ped}, car={car}"


def test_decision_vulnerable_in_path_never_leaks_proceed():
    """A confirmed vulnerable road user close in-path must floor to >=CAUTION on the
    VERY FIRST frame — the escalation ratchet's confirmation delay must not leak a
    single frame of PROCEED past a nearby pedestrian/rider.

    Regression: the faithful full-clip sweep (mirroring main.py's real decision
    path, road-fallback wired in) caught exactly one such frame — a motorcyclist
    at ~11 m crossing into the corridor showed PROCEED for one frame before the
    ratchet confirmed SLOW. The vulnerable-proximity floor closes that window.
    """
    eng = _engine()
    d = None
    for _ in range(4):                        # establish a clean PROCEED baseline
        d = eng.process(FakeLane(1.0, 0), [])
    assert d.longitudinal_level == 0, "expected a clean PROCEED baseline"
    # A pedestrian appears close in-path, NOT closing (ttc None) — the exact case.
    vru = FakeTrack(id=13, label="person", in_path=True, smoothed_distance_m=11.0,
                    closing_speed_mps=0.0, ttc_s=None, risk="MEDIUM", center=(640, 500))
    d = eng.process(FakeLane(1.0, 0), [vru])
    assert not d.degraded, "scene is not degraded — the non-degraded floor must fire"
    assert d.longitudinal_level >= 1, (
        f"leaked PROCEED with a person 11 m in-path (level {d.longitudinal_level})")


def test_decision_vulnerable_floor_is_scoped():
    """The vulnerable floor must NOT fire for an off-path or a far VRU — no spurious
    CAUTION on a roadside bystander or a distant pedestrian."""
    # (a) off-path VRU, close: no floor
    eng = _engine()
    d = None
    for _ in range(4):
        d = eng.process(FakeLane(1.0, 0), [])
    off = FakeTrack(id=1, label="person", in_path=False, smoothed_distance_m=8.0,
                    closing_speed_mps=0.0, ttc_s=None, risk="LOW", center=(200, 500))
    d = eng.process(FakeLane(1.0, 0), [off])
    assert d.longitudinal_level == 0, "off-path VRU wrongly floored to CAUTION"
    # (b) in-path VRU beyond the floor distance: no floor
    eng = _engine()
    for _ in range(4):
        d = eng.process(FakeLane(1.0, 0), [])
    far = FakeTrack(id=2, label="person", in_path=True,
                    smoothed_distance_m=cfg.VULNERABLE_FLOOR_DIST_M + 25.0,
                    closing_speed_mps=0.0, ttc_s=None, risk="LOW", center=(640, 380))
    d = eng.process(FakeLane(1.0, 0), [far])
    assert d.longitudinal_level == 0, "far in-path VRU wrongly floored to CAUTION"


# ---------------------------------------------------------------------------
# tracker — occlusion survival
# ---------------------------------------------------------------------------

def test_tracker_survives_brief_occlusion():
    """A confirmed track keeps its ID through an occlusion shorter than TRACK_MAX_AGE."""
    from tracker import MultiObjectTracker
    trk = MultiObjectTracker()
    box, ident = (600, 400, 700, 520), None
    for _ in range(cfg.TRACK_MIN_HITS + 1):
        c = [t for t in trk.update([_det(*box)]) if t.confirmed]
        if c:
            ident = c[0].id
    assert ident is not None, "track never confirmed"
    for _ in range(cfg.TRACK_MAX_AGE - 1):    # occlude, but not long enough to drop it
        trk.update([])
    reappeared = [t for t in trk.update([_det(*box)]) if t.id == ident]
    assert reappeared, f"track lost its ID through a {cfg.TRACK_MAX_AGE - 1}-frame occlusion"


# ---------------------------------------------------------------------------
# config — safety bounds
# ---------------------------------------------------------------------------

def test_config_safety_bounds():
    # distance bands tighten toward the vehicle in the right order
    assert cfg.DIST_EMERGENCY_M < cfg.RISK_DISTANCE_HIGH < cfg.RISK_DISTANCE_MEDIUM
    # brake scalars are a valid, ordered fraction
    assert 0.0 <= cfg.BRAKE_MIN <= cfg.BRAKE_MAX <= 1.0
    # latches / holds are real (>=1 frame)
    assert cfg.EMERGENCY_LATCH_FRAMES >= 1 and cfg.HOLD_FRAMES >= 1
    # a degraded system must release its caution NO faster than a healthy one
    assert cfg.HOLD_FRAMES_DEGRADED >= cfg.HOLD_FRAMES
    # every safety margin is non-negative (a negative one would brake LATE)
    assert cfg.VULNERABLE_TTC_MARGIN_S >= 0 and cfg.DEGRADED_TTC_MARGIN_S >= 0
    assert cfg.MIN_CLOSING_MPS >= 0


# ---------------------------------------------------------------------------
# frame guard — malformed input must degrade, never crash
# ---------------------------------------------------------------------------

def test_frame_guard_truth_table():
    from frame_guard import is_valid_frame
    assert is_valid_frame(np.zeros((720, 1280, 3), np.uint8))          # good BGR frame
    assert not is_valid_frame(None)                                    # dropped grab
    assert not is_valid_frame("not an array")
    assert not is_valid_frame(np.zeros((720, 1280), np.uint8))         # no channel axis
    assert not is_valid_frame(np.zeros((720, 1280, 4), np.uint8))      # wrong channel count
    assert not is_valid_frame(np.zeros((1, 1, 3), np.uint8))           # degenerate size
    assert not is_valid_frame(np.zeros((720, 1280, 3), np.float32))    # wrong dtype


def test_lane_detector_survives_malformed_frame():
    from lane_detection import LaneDetector
    lane = LaneDetector(frame_width=1280, frame_height=720)
    for bad in (None, np.zeros((10, 10), np.uint8), np.zeros((4, 4, 3), np.float32)):
        res, _ = lane.process(bad)
        assert res.confidence == 0.0, "malformed frame produced a lane lock"
    # a healthy frame still returns a usable result object (no lock on black, but no crash)
    res, _ = lane.process(np.zeros((720, 1280, 3), np.uint8))
    assert res is not None and res.frame_center_x == 640


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
