import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'perception'))

from perception.driver_monitoring import DriverMonitor

def evaluate_video(video_path):
    print(f"Evaluating: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    dm = DriverMonitor(frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    frame_count = 0
    phone_frames = 0
    distracted_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count > 150:
            break
            
        res, _ = dm.process(frame)
        
        if res.state == "PHONE":
            phone_frames += 1
        elif res.state == "DISTRACTED":
            distracted_frames += 1
            
        if frame_count % 100 == 0:
            print(f"Frame {frame_count}: State={res.state}")

    print(f"Total frames: {frame_count}")
    print(f"Phone frames: {phone_frames}")
    print(f"Distracted frames: {distracted_frames}")
    cap.release()

if __name__ == "__main__":
    import config as cfg
    cfg.DMS_ENABLE_PHONE = True
    cfg.DMS_ENABLE_HANDS = True
    
    base_path = r"c:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\dmd\gA\1\s3"
    face_video = os.path.join(base_path, "gA_1_s3_2019-03-14T14;31;08+01;00_rgb_face.mp4")
    body_video = os.path.join(base_path, "gA_1_s3_2019-03-14T14;31;08+01;00_rgb_body.mp4")
    
    # evaluate_video(face_video)
    evaluate_video(body_video)
