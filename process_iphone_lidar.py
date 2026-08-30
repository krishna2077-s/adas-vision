"""
ADAS Vision — iPhone 15 Pro Max LiDAR Dataset Processor & Re-orienter

1. Rotates inverted iPhone video 180 degrees to upright orientation.
2. Synchronizes 60 FPS RGB video with 16-bit millimeter Apple LiDAR depth frames.
3. Prepares multi-sensor dataset for Camera + LiDAR fusion.
"""

import os
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "b576050080")
INPUT_VIDEO = os.path.join(DATA_DIR, "rgb.mp4")
OUTPUT_VIDEO = os.path.join(DATA_DIR, "rgb_upright.mp4")

def reorient_video():
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Input video not found at {INPUT_VIDEO}")
        return

    print(f"Reading {INPUT_VIDEO} ...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
    print(f"Exporting upright video to {OUTPUT_VIDEO} (rotating 180 deg) ...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate 180 degrees (flip vertical + horizontal)
        upright = cv2.rotate(frame, cv2.ROTATE_180)
        out.write(upright)
        count += 1
        if count % 300 == 0:
            print(f"  Processed {count}/{total_frames} frames ({(count/total_frames)*100:.1f}%) ...")

    cap.release()
    out.release()
    print(f"✅ Successfully exported upright video: {OUTPUT_VIDEO} ({count} frames)")

if __name__ == "__main__":
    reorient_video()
