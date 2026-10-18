import sys
import cv2
from pathlib import Path

# Add core/ and perception/ to sys.path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "core"))
sys.path.insert(0, str(_ROOT / "perception"))

import config as cfg
from driver_monitoring import DriverMonitor

def test_image():
    img_path = r"C:\Users\Rakesh Sharma\.gemini\antigravity\brain\47d794cc-1f6d-4e10-9a52-8e8628f6ad91\.user_uploaded\media_1791095434397.png"
    out_path = r"C:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\output_phone_test.png"
    
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load image!")
        return

    h, w = frame.shape[:2]
    print(f"Loaded image: {w}x{h}")
    
    dm = DriverMonitor(frame_width=w, frame_height=h)
    
    # Run a few times to simulate history (so distraction flag triggers)
    result = None
    vis = None
    for _ in range(60): # 60 frames = 2 seconds of distraction
        result, vis = dm.process(frame)
        
    print(f"Final DMS State: {result.state}")
    print(f"Phone detected: {result.phone_detected}")
    print(f"Behavior: {result.behavior_label} ({result.behavior_conf*100:.1f}%)")
    print(f"Gaze Zone: {result.gaze_zone if result.gaze_zone else 'NONE'}")
    
    cv2.imwrite(out_path, vis)
    print(f"Saved annotated image to {out_path}")

if __name__ == "__main__":
    test_image()
