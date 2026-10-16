import cv2
import sys
from pathlib import Path

# Add core/perception to path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "perception"))

from driver_monitoring import DriverMonitor

def verify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return False
        
    dm = DriverMonitor()
    face_found = False
    
    # Check the first 60 frames (2 seconds) for a face
    for _ in range(60):
        ret, frame = cap.read()
        if not ret:
            break
            
        res, _ = dm.process(frame)
        if res.state != "NO DRIVER":
            face_found = True
            break
            
    cap.release()
    
    if face_found:
        print("SUCCESS: Face detected in video. This is a valid driver-facing video.")
    else:
        print("FAILURE: No face detected. This might be another road-facing video.")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_video.py <video_path>")
    else:
        verify_video(sys.argv[1])
