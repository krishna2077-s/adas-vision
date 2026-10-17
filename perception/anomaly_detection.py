import logging
import cv2
import numpy as np

import config as cfg
from object_detection import Detection, ObjectDetectionResult

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Out-of-Distribution (OOD) Anomaly Detector.
    
    This module cross-references the purely visual drivable-area mask
    (from the LearnedRoadDetector) with the explicit object detections (from YOLO).
    If a massive non-drivable blob appears in the vehicle's direct path, but YOLO
    does NOT recognize it as a known class (e.g., an unusual animal, debris, or 
    overturned cargo), this module flags it as an 'anomaly' to trigger the AEB/FCW.
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height
        
        # We define a strict path corridor (trapezoid) directly in front of the ego vehicle.
        # Anomalies outside this corridor are ignored to prevent false positives from grass/trees.
        self.path_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # Corridor geometry
        horizon_y = int(self.h * cfg.LEARNED_HORIZON_FRAC)
        bottom_y = int(self.h * getattr(cfg, "LEARNED_BONNET_FRAC", 0.93))
        
        # Top width (narrow) and bottom width (wide)
        top_hw = cfg.PATH_CORRIDOR_MIN_PX
        bot_hw = cfg.PATH_CORRIDOR_MAX_PX
        
        pts = np.array([[
            [self.w//2 - top_hw, horizon_y],
            [self.w//2 + top_hw, horizon_y],
            [self.w//2 + bot_hw, bottom_y],
            [self.w//2 - bot_hw, bottom_y]
        ]], dtype=np.int32)
        
        cv2.fillPoly(self.path_mask, pts, 255)

    def process(
        self, 
        drivable_mask: np.ndarray, 
        obj_result: ObjectDetectionResult, 
        lane_center_x: int,
        annotated: np.ndarray
    ) -> ObjectDetectionResult:
        """
        Scans for unlabeled obstacles in the drivable area.
        """
        # --- Safety Check 1: Is the CNN completely blind? ---
        # If the domain simulation (Pro-RandConv) or extreme weather destroys the texture,
        # the CNN might output a nearly empty road mask.
        # If less than 2% of the screen is road, the CNN is blind. Do not flag anomalies based on it.
        total_pixels = self.w * self.h
        road_pixels = cv2.countNonZero(drivable_mask)
        if road_pixels < (total_pixels * 0.02):
            return obj_result

        # 1. Shift the path mask to align with the current steering center
        if lane_center_x is None:
            lane_center_x = self.w // 2
            
        shift_x = lane_center_x - (self.w // 2)
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        dynamic_path = cv2.warpAffine(self.path_mask, M, (self.w, self.h))
        
        # --- Safety Check 2: Dynamic Horizon ---
        # Find the highest point of the detected road to prevent flagging the sky/background
        # as a massive obstacle just because it falls inside our path trapezoid.
        road_coords = np.argwhere(drivable_mask > 0)
        if road_coords.size > 0:
            highest_road_y = np.min(road_coords[:, 0])
            # Erase the top of the path corridor above the actual road horizon
            dynamic_path[0:highest_road_y + 10, :] = 0
        
        # 2. Find anomalies: pixels that are IN the path, but NOT drivable
        obstacles = cv2.bitwise_not(drivable_mask)
        corridor_obstacles = cv2.bitwise_and(obstacles, dynamic_path)
        
        # 3. Mask out known YOLO objects
        for det in obj_result.detections:
            pad = 20
            x1 = max(0, det.x1 - pad)
            y1 = max(0, det.y1 - pad)
            x2 = min(self.w, det.x2 + pad)
            y2 = min(self.h, det.y2 + pad)
            cv2.rectangle(corridor_obstacles, (x1, y1), (x2, y2), 0, -1)
            
        # 4. Find large remaining blobs
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corridor_obstacles, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Minimum size threshold for a critical physical hazard (bumped up to avoid small puddles/patches)
            if area > 4000:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Filter out flat horizontal artifacts (lane markers, crosswalks, reflective paint, shadows)
                if w > 2.5 * h:
                    continue
                    
                # Filter out extremely thin slivers
                if h < 30:
                    continue
                
                # Estimate distance from perspective: bottom of blob = contact point
                # Objects lower in the frame (higher y) are closer to the camera.
                # Map bottom-y from [horizon..bonnet] to [~30m..~2m] linearly.
                bottom_y = y + h
                horizon_y = int(self.h * cfg.LEARNED_HORIZON_FRAC)
                bonnet_y = int(self.h * getattr(cfg, "LEARNED_BONNET_FRAC", 0.93))
                span = max(1, bonnet_y - horizon_y)
                depth_ratio = max(0.0, min(1.0, (bottom_y - horizon_y) / span))
                est_dist = 30.0 - depth_ratio * 28.0  # 30m at horizon, 2m at bonnet
                est_dist = max(2.0, round(est_dist, 1))
                    
                anomaly = Detection(
                    label="ANOMALY",
                    confidence=1.0, 
                    x1=x, y1=y, x2=x+w, y2=y+h,
                    distance_m=est_dist, 
                    risk="HIGH",
                    in_path=True
                )
                
                obj_result.detections.append(anomaly)
                obj_result.highest_risk = "HIGH"
                if obj_result.nearest_in_path is None or obj_result.nearest_in_path.distance_m > est_dist:
                    obj_result.nearest_in_path = anomaly
                
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(annotated, f"UNKNOWN HAZARD! {est_dist:.0f}m", (x, max(15, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        return obj_result
