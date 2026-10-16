import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'perception'))

from ultralytics import YOLO
import config as cfg

model = YOLO(cfg.YOLO_MODEL)
base_path = r"c:\Users\Rakesh Sharma\OneDrive\Desktop\New folder\adas-vision\dmd\gA\1\s3"
body_video = os.path.join(base_path, "gA_1_s3_2019-03-14T14;31;08+01;00_rgb_body.mp4")

cap = cv2.VideoCapture(body_video)
frame_count = 0
found = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 30 != 0:
        continue
        
    results = model.predict(frame, conf=0.1, verbose=False, imgsz=640)
    for r in results:
        for c, conf in zip(r.boxes.cls, r.boxes.conf):
            cls_name = model.names[int(c)]
            if cls_name in ["cell phone", "remote", "book", "tv"]:
                print(f"Frame {frame_count}: {cls_name} (conf={float(conf):.2f})")
                found = True

if not found:
    print("No phone/remote found.")
cap.release()
