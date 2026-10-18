import urllib.request
import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", model_path)

base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

image = np.zeros((480, 640, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
detection_result = detector.detect(mp_image)
print("Result:", detection_result)
