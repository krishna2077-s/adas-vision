"""
test_udp_bridge.py — Test UDP receiver for ADAS Vision telemetry bridge.

This standalone test receiver verifies that UDP telemetry packets broadcast by
`udp_bridge.py` are properly structured, transmitted, and decoded before connecting
to downstream clients such as MATLAB / Simulink or external dashboards.

Usage:
    python test_udp_bridge.py
    python test_udp_bridge.py --port 5005
    python test_udp_bridge.py --host 0.0.0.0 --port 5005

Features:
    - Listens on UDP port (default 5005).
    - Decodes incoming JSON packets and pretty-prints per-frame telemetry.
    - Displays frame ID, timestamp, FPS, traffic light state, road boundary,
      driving decision, and tracked objects.
    - Calculates and logs throughput stats (packets/sec and avg size) every 100 packets.
    - Handles graceful termination on Ctrl+C.
    - Uses Python standard library only (no external dependencies).
"""

import argparse
import json
import socket
import sys
import time
from typing import Any, Dict, List, Optional


def _extract(data: dict, *keys: str, default: Any = None) -> Any:
    """Helper to retrieve first matching key from a dictionary."""
    if not isinstance(data, dict):
        return default
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    return default


def _format_float(val: Any, decimals: int = 1, suffix: str = "") -> str:
    """Format a float value or return 'N/A'."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}{suffix}"
    return "N/A" if val is None else str(val)


def print_frame_summary(data: Dict[str, Any], raw_size: int) -> None:
    """Pretty-print the parsed telemetry packet for a single frame."""
    # Frame metadata
    frame_id = _extract(data, "frame_id", "frame", "frame_index", "i", default="N/A")
    timestamp = _extract(data, "timestamp", "timestamp_s", "time", "t", "t_ms", default="N/A")
    fps = _extract(data, "fps", default=None)

    # Road boundary
    road = _extract(data, "road_boundary", "road", "lane", default={})
    if not isinstance(road, dict):
        road = {}
    center_x = _extract(road, "center_x", "lane_center_x", default=_extract(data, "center_x", "lane_center_x", default="N/A"))
    confidence = _extract(road, "confidence", "conf", default=_extract(data, "lane_confidence", "confidence", default=None))
    source = _extract(road, "source", "type", default=_extract(data, "road_source", "source", default="N/A"))

    # Decision
    decision = _extract(data, "decision", "dec", default={})
    if not isinstance(decision, dict):
        decision = {}
    longitudinal = _extract(decision, "longitudinal", "lon", default=_extract(data, "longitudinal", default="N/A"))
    rule_id = _extract(decision, "rule_id", "rule", default=_extract(data, "rule_id", default="N/A"))
    reason = _extract(decision, "reason", "why", "hz", default=_extract(data, "reason", default=""))
    if fps is None and isinstance(decision, dict):
        fps = _extract(decision, "fps", default=None)

    # Truncate reason to 80 chars
    reason_str = str(reason) if reason is not None else ""
    if len(reason_str) > 80:
        reason_str = reason_str[:77] + "..."

    # Traffic light state
    traffic_light = _extract(data, "traffic_light_state", "light_state", "traffic_light", "light", "tl_state", default="N/A")

    # Tracks
    tracks = _extract(data, "tracks", "trk", default=[])
    if not isinstance(tracks, list):
        tracks = []

    # Print summary box
    print("=" * 80)
    ts_str = f"{timestamp:.3f}s" if isinstance(timestamp, (int, float)) and timestamp < 10000 else str(timestamp)
    print(f"[FRAME {frame_id}] Time: {ts_str} | FPS: {_format_float(fps, 1)} | Traffic Light: {traffic_light} | Packet: {raw_size} B")
    print("-" * 80)
    print(f"  Decision      : {longitudinal} (Rule: {rule_id})")
    if reason_str:
        print(f"  Reason        : {reason_str}")
    print(f"  Road Boundary : Center X: {center_x} | Conf: {_format_float(confidence, 2)} | Source: {source}")

    print(f"  Tracks ({len(tracks)}):")
    if tracks:
        for t in tracks:
            if not isinstance(t, dict):
                continue
            t_id = _extract(t, "track_id", "id", default="?")
            t_cls = _extract(t, "class", "class_name", "label", "lab", "name", default="unknown")
            t_dist = _extract(t, "distance_m", "distance", "dist", "d", default=None)
            t_ttc = _extract(t, "ttc_s", "ttc", default=None)
            t_in_path = _extract(t, "in_path", "ip", default=None)

            in_path_str = "True" if t_in_path is True else ("False" if t_in_path is False else str(t_in_path))
            print(f"    • ID {t_id:<3} | Class: {t_cls:<12} | Dist: {_format_float(t_dist, 1, 'm'):>7} | "
                  f"TTC: {_format_float(t_ttc, 1, 's'):>6} | In Path: {in_path_str}")
    else:
        print("    (none)")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test UDP receiver for verifying ADAS Vision telemetry bridge before MATLAB integration."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host/IP address to bind the UDP socket (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="UDP port to listen on (default: 5005)",
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.host, args.port))
    except Exception as e:
        print(f"[ERROR] Failed to bind UDP socket to {args.host}:{args.port} - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] UDP Test Receiver listening on {args.host}:{args.port}")
    print("[INFO] Waiting for telemetry packets from udp_bridge.py (Press Ctrl+C to stop)...")
    print("-" * 80)

    packet_count = 0
    window_packets = 0
    window_bytes = 0
    window_start_time = time.time()
    total_bytes = 0

    try:
        while True:
            raw_data, addr = sock.recvfrom(65535)
            now = time.time()
            packet_len = len(raw_data)

            packet_count += 1
            window_packets += 1
            window_bytes += packet_len
            total_bytes += packet_len

            try:
                text_payload = raw_data.decode("utf-8")
                payload = json.loads(text_payload)
            except UnicodeDecodeError:
                print(f"[WARN] Received invalid non-UTF-8 packet ({packet_len} bytes) from {addr}")
                continue
            except json.JSONDecodeError as err:
                print(f"[WARN] Failed to parse JSON packet from {addr}: {err}")
                continue

            if isinstance(payload, dict):
                print_frame_summary(payload, packet_len)
            else:
                print(f"[WARN] Received JSON is not an object: {payload}")

            # Every 100 packets, print statistics
            if packet_count % 100 == 0:
                elapsed = now - window_start_time
                pkt_rate = window_packets / elapsed if elapsed > 0 else 0.0
                avg_size = window_bytes / window_packets if window_packets > 0 else 0.0
                print(f">>> [STATS] Received {packet_count} packets | Rate: {pkt_rate:.1f} pkts/s | Avg size: {avg_size:.1f} bytes | Total: {total_bytes / 1024:.1f} KB")
                window_start_time = time.time()
                window_packets = 0
                window_bytes = 0

    except KeyboardInterrupt:
        print("\n[INFO] Receiver stopping due to Ctrl+C interrupt...")
    finally:
        sock.close()
        print(f"[INFO] Socket closed. Total packets received: {packet_count} ({total_bytes / 1024:.1f} KB). Clean shutdown complete.")


if __name__ == "__main__":
    main()
