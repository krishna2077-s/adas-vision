"""
test_udp_bridge.py — UDP Bridge Verification Tool

Listens on UDP port 5005 and prints incoming JSON telemetry from
udp_bridge.py in a human-readable format. Use this to confirm the
bridge is streaming before connecting MATLAB.

Usage:
    # In terminal 1: python udp_bridge.py --video dashcam.mp4 --async
    # In terminal 2: python test_udp_bridge.py

Press Ctrl+C to stop.
"""

import json
import socket
import sys
import time

UDP_PORT    = 5005
BUFFER_SIZE = 65535


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
    except OSError as e:
        print(f"[ERROR] Cannot bind to port {UDP_PORT}: {e}")
        print("        Make sure udp_bridge.py is running first.")
        sys.exit(1)

    sock.settimeout(5.0)

    print(f"[test_udp_bridge] Listening on UDP port {UDP_PORT} ...")
    print("  Waiting for packets from udp_bridge.py  (Ctrl+C to stop)\n")
    print("-" * 72)

    pkt_count = 0
    last_print = time.monotonic()

    try:
        while True:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                print("[WARN] No packet received in 5 s — is udp_bridge.py running?")
                continue

            try:
                pkt = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON: {e}")
                continue

            pkt_count += 1
            now = time.monotonic()

            # Print every packet but throttle display to ~2 Hz so the
            # terminal is readable.
            if now - last_print < 0.5:
                continue
            last_print = now

            frame_id    = pkt.get("frame_id", "?")
            ts          = pkt.get("timestamp_s", 0.0)
            fps         = pkt.get("fps", 0.0)
            decision    = pkt.get("decision", {})
            lane        = pkt.get("lane", {})
            tracks      = pkt.get("tracks", [])
            ego         = pkt.get("ego_from_matlab")

            # ── Header ──────────────────────────────────────────────────
            print(f"\n[Frame {frame_id:>6}]  t={ts:7.2f}s  FPS={fps:5.1f}  "
                  f"pkts_total={pkt_count}")

            # ── Decision ─────────────────────────────────────────────────
            lon   = decision.get("longitudinal", "?")
            lat   = decision.get("lateral", "?")
            brake = decision.get("brake_demand", 0.0)
            rule  = decision.get("rule_id", "?")
            print(f"  Decision : {lon:<20} {lat:<20}  brake={brake:.2f}  [{rule}]")
            print(f"  Reason   : {decision.get('reason', '')}")

            # ── Lane ─────────────────────────────────────────────────────
            offset  = lane.get("offset_px", 0)
            conf    = lane.get("confidence", 0.0)
            steer   = lane.get("steering", "?")
            left_ok = "YES" if lane.get("left_lane")  else "NO "
            rght_ok = "YES" if lane.get("right_lane") else "NO "
            print(f"  Lane     : offset={offset:+4d}px  conf={conf:.0%}  "
                  f"L={left_ok}  R={rght_ok}  steering={steer}")

            # ── Tracks ───────────────────────────────────────────────────
            if tracks:
                print(f"  Tracks   : {len(tracks)} object(s)")
                for t in tracks[:5]:   # show up to 5
                    dist = f"{t['distance_m']:.1f}m" if t["distance_m"] else "  ?"
                    ttc  = f"{t['ttc_s']:.1f}s"      if t["ttc_s"]      else "  ?"
                    flag = "IN-PATH" if t["in_path"] else "       "
                    print(f"    #{t['track_id']:>3}  {t['label']:<14}  "
                          f"dist={dist:>6}  TTC={ttc:>5}  risk={t['risk']:<6}  {flag}")
                if len(tracks) > 5:
                    print(f"    ... and {len(tracks) - 5} more")
            else:
                print("  Tracks   : none")

            # ── MATLAB ego feedback ──────────────────────────────────────
            if ego:
                print(f"  EGO (MATLAB): x={ego[0]:.2f} m  y={ego[1]:.2f} m")
            else:
                print("  EGO (MATLAB): not connected yet")

            print("-" * 72)

    except KeyboardInterrupt:
        print(f"\n[test_udp_bridge] Stopped. Received {pkt_count} packets total.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
