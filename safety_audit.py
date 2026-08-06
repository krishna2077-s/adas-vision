"""
safety_audit.py — full-clip audit of the SAFETY SPINE on main.py's real decision path.

The unit tests (test_adas.py) prove the decision engine's rules in isolation; this
tool proves the *assembled* system over a whole clip. It mirrors main.py's exact
perception -> fusion chain:

    lanes  ->  (no paint) learned road  ->  (no lock) classical road
           ->  objects(lane_center_x)  ->  tracker  ->  engine.process(...)

and then checks the invariant that matters most for an advisory ADAS:

    THE SPINE:  the system must never output PROCEED while a confirmed in-path
                hazard sits within SPINE_DIST metres — and especially never for a
                vulnerable road user (pedestrian / cyclist / rider / animal).

Why this file exists: an earlier ad-hoc sweep fed the *raw* lane result to the
engine and never wired in the road-model fallback that main.py actually runs.
Because the engine marks itself `degraded` whenever lane confidence is 0, that
sweep both inflated the degraded rate AND tested the spine on an artificially
conservative path (degraded mode suppresses PROCEED). Auditing the REAL path
(engine un-degraded by the road model, so it PROCEEDs far more often) is the only
honest test — and it caught a one-frame PROCEED past a pedestrian entering the
corridor, since fixed by the vulnerable-proximity floor.

The nearest-hazard check here is computed INDEPENDENTLY of the engine's own hazard
pick (straight from the confirmed in-path tracks), so it is a true external audit,
not the engine grading its own homework.

Honest limitation: run this at or above real-time speed. The engine has an fps
trust-gate (fps < cfg.MIN_FPS -> degraded); an offline run slower than MIN_FPS
will over-report degraded frames. This tool prints the achieved fps and warns if
it dipped near MIN_FPS so that artifact is visible, never silent.

Usage:
    python safety_audit.py --video dashcam.mp4
    python safety_audit.py --video dashcam.mp4 --max-frames 4000 --spine-dist 15
    python safety_audit.py --video dashcam.mp4 --json audit.json

Exit code is 0 when the spine holds, 1 when any PROCEED-with-in-path-hazard frame
is found — so it can gate a change in CI.
"""

import argparse
import collections
import logging
import sys
import time

import cv2

import config as cfg
from decision_engine import DecisionEngine
from frame_guard import is_valid_frame
from lane_detection import LaneDetector
from tracker import MultiObjectTracker

LEVEL_NAMES = ["PROCEED", "CAUTION", "SLOW", "BRAKE", "EMERGENCY"]


def _build_road_detectors(w, h):
    """Same fallback chain main.py builds: learned drivable-area model first,
    classical surface detector second. Either may be absent on a clean checkout."""
    learned = classical = None
    if cfg.ENABLE_LEARNED_ROAD:
        try:
            from learned_road_detection import LearnedRoadDetector
            lrd = LearnedRoadDetector(w, h)
            learned = lrd if lrd.available else None
        except Exception as exc:  # torch/model absent -> classical fallback
            logging.getLogger(__name__).info(f"learned road detector off ({exc})")
    if cfg.ENABLE_ROAD_FALLBACK:
        try:
            from road_detection import RoadDetector
            classical = RoadDetector(frame_width=w, frame_height=h)
        except Exception:
            classical = None
    return learned, classical


def audit(video, max_frames=10**9, spine_dist=15.0, quiet=False):
    for n in ("ultralytics", "object_detection", "learned_road_detection",
              "lane_detection", "road_detection", "decision_engine"):
        logging.getLogger(n).setLevel(logging.ERROR)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"[audit] cannot open {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames

    lane = LaneDetector(frame_width=w, frame_height=h)
    trk = MultiObjectTracker()
    eng = DecisionEngine(frame_width=w, frame_height=h)
    learned, classical = _build_road_detectors(w, h)
    try:
        from object_detection import ObjectDetector
        obj = ObjectDetector(w, h)
    except ImportError as exc:
        raise SystemExit(f"[audit] object detector required for a safety audit ({exc})")

    # Non-invasively record WHY the engine degrades, for an honest breakdown.
    _orig_assess = eng._assess_trust
    cause_ref = {"c": ""}

    def _assess_wrap(lane_result, tracks, hazard, fps):
        d, c = _orig_assess(lane_result, tracks, hazard, fps)
        cause_ref["c"] = c if d else ""
        return d, c
    eng._assess_trust = _assess_wrap

    src = collections.Counter()
    lvl = collections.Counter()
    causes = collections.Counter()
    degraded = 0
    proceed = 0
    spine_hits = []          # PROCEED with an in-path hazard within spine_dist
    late = []                # TTC < 1.5 s but not yet BRAKE
    prev_lvl = None
    changes = 0
    min_fps_seen = 1e9
    n = 0
    t0 = time.time()

    while n < min(max_frames, total):
        ok, frame = cap.read()
        if not ok:
            break
        if not is_valid_frame(frame):
            continue

        # ── main.py order: lanes -> learned road -> classical road ──
        lane_result, _ = lane.process(frame.copy())
        used = "paint" if lane_result.confidence > 0 else "none"
        if lane_result.confidence == 0.0:
            got = False
            if learned is not None:
                rr, _ = learned.process(frame)
                if rr.confidence > 0.0:
                    lane_result, used, got = rr, "learned", True
            if not got and classical is not None:
                rr, _ = classical.process(frame)
                if rr.confidence > 0.0:
                    lane_result, used = rr, "classical"

        obj_result, _ = obj.process(frame.copy(), lane_result.lane_center_x)
        tracks = trk.update(obj_result.detections)
        dec = eng.process(lane_result, tracks)

        # ── independent nearest confirmed in-path non-advisory hazard ──
        near = None
        for t in tracks:
            if (t.confirmed and t.in_path and t.label not in cfg.ADVISORY_CLASSES
                    and t.smoothed_distance_m is not None):
                if near is None or t.smoothed_distance_m < near.smoothed_distance_m:
                    near = t

        lv = dec.longitudinal_level
        src[used] += 1
        lvl[lv] += 1
        if dec.degraded:
            degraded += 1
            causes[cause_ref["c"]] += 1
        if lv == 0:
            proceed += 1
            if near is not None and near.smoothed_distance_m < spine_dist:
                spine_hits.append((n, near.label, round(near.smoothed_distance_m, 1),
                                   near.label in cfg.VULNERABLE_CLASSES, used))
        if (near is not None and near.ttc_s is not None
                and near.ttc_s < 1.5 and lv < 3):
            late.append((n, near.label, round(near.smoothed_distance_m, 1),
                         round(near.ttc_s, 1), lv))
        if prev_lvl is not None and lv != prev_lvl:
            changes += 1
        prev_lvl = lv
        if n > 5:          # skip warmup: the engine's first dt reads as fps~0
            min_fps_seen = min(min_fps_seen, dec.fps)

        n += 1
        if not quiet and n % 1000 == 0:
            print(f"  ...{n}/{total}  ({n/(time.time()-t0):.1f} fps)", flush=True)
    cap.release()

    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else 0.0
    return {
        "frames": n, "fps": round(fps, 1), "min_fps_seen": round(min_fps_seen, 1),
        "spine_dist": spine_dist, "src": src, "lvl": lvl, "degraded": degraded,
        "causes": causes, "proceed": proceed, "spine_hits": spine_hits,
        "late": late, "changes": changes,
    }


def report(a):
    N = max(1, a["frames"])
    vuln_hits = [x for x in a["spine_hits"] if x[3]]
    print(f"\n=== SAFETY-SPINE AUDIT — {a['frames']} frames @ {a['fps']} fps "
          f"(spine {a['spine_dist']:.0f} m) ===")

    print("\nguidance source:")
    for k in ("paint", "learned", "classical", "none"):
        v = a["src"].get(k, 0)
        print(f"  {k:10s} {v:6d} ({v/N*100:5.1f}%)")

    print("\ncommitted-level distribution:")
    for k in range(5):
        v = a["lvl"].get(k, 0)
        print(f"  {LEVEL_NAMES[k]:10s} {v:6d} ({v/N*100:5.1f}%)")

    print(f"\ndegraded frames: {a['degraded']} ({a['degraded']/N*100:.1f}%)")
    for c, k in a["causes"].most_common():
        print(f"  {k:6d} ({k/N*100:5.2f}%)  {c or '(unlabelled)'}")

    print(f"\n[SPINE] PROCEED frames               : {a['proceed']} ({a['proceed']/N*100:.1f}%)")
    print(f"[SPINE] PROCEED w/ <{a['spine_dist']:.0f} m in-path hazard : "
          f"{len(a['spine_hits'])}   <-- MUST be 0")
    print(f"[SPINE]   of those, vulnerable        : {len(vuln_hits)}")
    for f, lbl, d, v, used in a["spine_hits"][:10]:
        print(f"    f{f} near {lbl} {d}m {'(VRU)' if v else ''} used={used}")

    print(f"\n[LATE]  TTC<1.5 s but < BRAKE        : {len(a['late'])} "
          f"({len(a['late'])/N*100:.2f}%)")
    for f, lbl, d, ttc, lv in a["late"][:6]:
        print(f"    f{f} near {lbl} {d}m ttc {ttc}s lvl {LEVEL_NAMES[lv]}")

    print(f"\n[CHURN] level changes: {a['changes']} ({a['changes']/N*100:.2f}%/frame)")

    if a["min_fps_seen"] < cfg.MIN_FPS * 1.5:
        print(f"\n[!] achieved fps dipped to {a['min_fps_seen']} (MIN_FPS={cfg.MIN_FPS}); "
              f"some degraded frames may be an offline-speed artifact, not real behaviour.")

    ok = len(a["spine_hits"]) == 0
    print("\n" + "=" * 60)
    print(f"VERDICT: safety spine {'HOLDS' if ok else 'VIOLATED'} — "
          f"{a['proceed']} PROCEED frames, {len(a['spine_hits'])} into a near in-path hazard.")
    print("=" * 60)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Audit the ADAS safety spine over a full clip.")
    ap.add_argument("--video", default="dashcam.mp4", help="dashcam clip to audit")
    ap.add_argument("--max-frames", type=int, default=10**9)
    ap.add_argument("--spine-dist", type=float, default=15.0,
                    help="a PROCEED with a confirmed in-path hazard nearer than this fails the audit")
    ap.add_argument("--json", default=None, help="optional path to dump the raw metrics")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    a = audit(args.video, args.max_frames, args.spine_dist, args.quiet)
    ok = report(a)

    if args.json:
        import json
        dump = dict(a)
        for k in ("src", "lvl", "causes"):
            dump[k] = dict(dump[k])
        json.dump(dump, open(args.json, "w"), indent=2)
        print(f"[audit] metrics -> {args.json}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
