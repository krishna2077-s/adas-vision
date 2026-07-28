"""
driver_monitor_demo.py — standalone webcam demo of the driver-attention monitor.

Run:
    python driver_monitor_demo.py            # default webcam (index 0)
    python driver_monitor_demo.py --cam 1
Press Q to quit.

Advisory / research only.
"""

import argparse

import cv2

from driver_monitoring import DriverMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.cam}")
    dm = DriverMonitor()
    if not dm.available:
        raise SystemExit("Haar cascades unavailable in this OpenCV build")

    print("Driver monitor running — press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res, vis = dm.process(frame)
        cv2.imshow("Driver Monitor (Q quits)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
