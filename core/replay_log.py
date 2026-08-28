"""
replay_log.py — reconstruct a drive from a drive_logger .jsonl (the black box).

Proves the log is self-sufficient: it re-renders the whole drive — tracked
boxes, the decision state + reason, the FCW banner, light/limit, and the plan —
from the LOG ALONE, running no models. Overlay it back onto the original video
(--video) or onto a black canvas (log-only), and/or print a drive summary.

    # summary stats (decision histogram, FCW events, ID switches)
    python replay_log.py drive.jsonl

    # rebuild an annotated video from the log + the original footage
    python replay_log.py drive.jsonl --video dashcam.mp4 --save replay.mp4

    # dump a single reconstructed frame to inspect
    python replay_log.py drive.jsonl --video dashcam.mp4 --dump 300 out.jpg

This is why the recorder earns its keep: "what did it see and decide at 00:42,
and why?" is answerable months later, with no re-run.
"""

import argparse
import json
from collections import Counter

import cv2
import numpy as np

# Colour maps mirror the live HUD (kept local so replay has no live-module deps).
_LEVEL_COLORS = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 140, 255),
                 3: (0, 90, 255), 4: (0, 0, 255)}
_RISK_COLORS = {"LOW": (0, 200, 0), "MEDIUM": (0, 200, 255), "HIGH": (0, 0, 255)}
_FCW_COLORS = {1: (0, 200, 255), 2: (0, 90, 255), 3: (0, 0, 255)}
_LIGHT_COLORS = {"RED": (0, 0, 255), "AMBER": (0, 200, 255), "GREEN": (0, 220, 0)}


def load(path):
    header, frames, footer = None, {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("type")
            if t == "header":
                header = obj
            elif t == "frame":
                frames[obj["i"]] = obj
            elif t == "footer":
                footer = obj
    return header, frames, footer


def render(base, rec, w, h):
    """Draw the reconstructed overlay for one frame record onto `base` (in place)."""
    # tracked boxes
    for t in rec.get("trk", []):
        x1, y1, x2, y2 = t["box"]
        col = _RISK_COLORS.get(t.get("rk"), (180, 180, 180))
        cv2.rectangle(base, (x1, y1), (x2, y2), col, 3 if t.get("ip") else 1)
        d = t.get("d")
        tag = f"#{t['id']} {t['lab']}" + (f" {d:.0f}m" if isinstance(d, (int, float)) else "")
        cv2.putText(base, tag, (x1 + 2, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # decision chip (top-right)
    dec = rec.get("dec")
    if dec:
        col = _LEVEL_COLORS.get(dec.get("lvl", 0), (200, 200, 200))
        x0 = max(10, w - 330)
        cv2.rectangle(base, (x0, 10), (x0 + 320, 78), (20, 20, 20), -1)
        cv2.rectangle(base, (x0, 10), (x0 + 320, 78), col, 1)
        cv2.putText(base, f"{dec.get('lon','')}  [{dec.get('rule','')}]", (x0 + 10, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 2, cv2.LINE_AA)
        d, ttc = dec.get("d"), dec.get("ttc")
        line = f"{dec.get('hz') or 'clear'}"
        if isinstance(d, (int, float)):
            line += f" {d:.0f}m"
        if isinstance(ttc, (int, float)):
            line += f" TTC {ttc:.1f}s"
        cv2.putText(base, line, (x0 + 10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (210, 210, 210), 1, cv2.LINE_AA)

    # FCW banner (top-centre) from the logged level
    fcw = rec.get("fcw")
    if fcw and fcw.get("lvl", 0) >= 1:
        col = _FCW_COLORS.get(fcw["lvl"], (0, 0, 255))
        title = {1: "COLLISION RISK", 2: "COLLISION WARNING", 3: "BRAKE"}.get(fcw["lvl"], "")
        bw = 480
        x0 = w // 2 - bw // 2
        cv2.rectangle(base, (x0, 8), (x0 + bw, 52), col, -1)
        cv2.putText(base, title, (x0 + 16, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        ttc = fcw.get("ttc")
        if isinstance(ttc, (int, float)):
            cv2.putText(base, f"TTC {ttc:.1f}s", (x0 + bw - 120, 39),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # light + limit chips (top-left)
    y = 30
    if rec.get("light"):
        col = _LIGHT_COLORS.get(rec["light"], (180, 180, 180))
        cv2.circle(base, (24, y - 6), 8, col, -1)
        cv2.putText(base, f"LIGHT {rec['light']}", (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 2, cv2.LINE_AA)
        y += 28
    if rec.get("limit"):
        cv2.putText(base, f"LIMIT {rec['limit']}", (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 220), 2, cv2.LINE_AA)

    # plan / sim-speed line (bottom)
    plan = rec.get("plan")
    if plan:
        tgt = plan.get("tgt")
        txt = "PLAN " + (f"{tgt*3.6:.0f} km/h" if isinstance(tgt, (int, float)) else "--")
        if plan.get("why"):
            txt += f"  {plan['why']}"
        cv2.putText(base, txt, (14, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 180), 2, cv2.LINE_AA)

    # frame index / time
    cv2.putText(base, f"f{rec['i']}  {rec.get('t',0)/1000:.1f}s  REPLAY", (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return base


def summarise(header, frames, footer):
    print("== drive summary ======================================")
    if header:
        print(f"source   : {header.get('source')}")
        print(f"frames   : {len(frames)}  ({header.get('w')}x{header.get('h')} @ {header.get('fps')} fps)")
        print(f"backend  : {header.get('config',{}).get('learned_backend')} "
              f"({header.get('config',{}).get('ov_device')})")

    levels = Counter()
    fcw_events = Counter()
    lights = Counter()
    id_switch = 0
    prev_hazard = None
    max_trk = 0
    for i in sorted(frames):
        r = frames[i]
        dec = r.get("dec") or {}
        levels[dec.get("lon", "?")] += 1
        fcw = r.get("fcw") or {}
        if fcw.get("lvl", 0) >= 2:
            fcw_events[fcw.get("name", "?")] += 1
        if r.get("light"):
            lights[r["light"]] += 1
        hid = dec.get("hid")
        if hid is not None and prev_hazard is not None and hid != prev_hazard:
            id_switch += 1
        if hid is not None:
            prev_hazard = hid
        max_trk = max(max_trk, len(r.get("trk", [])))

    print("\ndecision levels (frames):")
    for k, n in levels.most_common():
        print(f"   {k:15s} {n}")
    print(f"\nFCW warn/imminent frames: {dict(fcw_events) or 'none'}")
    print(f"light states seen       : {dict(lights) or 'none'}")
    print(f"hazard ID switches      : {id_switch}")
    print(f"max concurrent tracks   : {max_trk}")
    if footer:
        print(f"footer frames           : {footer.get('frames')}")


def main():
    ap = argparse.ArgumentParser(description="Replay / summarise a drive_logger .jsonl.")
    ap.add_argument("log", help="path to the .jsonl drive log")
    ap.add_argument("--video", help="original video, to overlay onto (else black canvas)")
    ap.add_argument("--save", help="write a reconstructed .mp4 here")
    ap.add_argument("--dump", nargs=2, metavar=("FRAME", "OUT"), help="write one frame to an image")
    args = ap.parse_args()

    header, frames, footer = load(args.log)
    if not frames:
        print("no frame records in log.")
        return
    w = header.get("w", 1280) if header else 1280
    h = header.get("h", 720) if header else 720

    if args.dump:
        fi, out = int(args.dump[0]), args.dump[1]
        rec = frames.get(fi) or frames[min(frames, key=lambda k: abs(k - fi))]
        base = np.full((h, w, 3), 30, np.uint8)
        if args.video:
            cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, rec["i"])
            ok, f = cap.read(); cap.release()
            if ok:
                base = f
        cv2.imwrite(out, render(base, rec, w, h))
        print(f"wrote reconstructed frame {rec['i']} -> {out}")
        return

    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = header.get("fps", 30) if header else 30
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        cap = cv2.VideoCapture(args.video) if args.video else None
        for i in sorted(frames):
            rec = frames[i]
            base = np.full((h, w, 3), 30, np.uint8)
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, f = cap.read()
                if ok:
                    base = f
            writer.write(render(base, rec, w, h))
        writer.release()
        if cap is not None:
            cap.release()
        print(f"wrote reconstructed video -> {args.save}")

    summarise(header, frames, footer)


if __name__ == "__main__":
    main()
