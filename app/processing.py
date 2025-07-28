# FILE: app/processing.py
# This is the heart of your project. The ViolenceProcessor class encapsulates all CV logic.
#type: ignore

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import supervision as sv

# Import project-specific modules
from utils import config
from utils.logger import log_violence_event

class ViolenceProcessor:
    def __init__(self):
        """Initializes all models, trackers, and state variables."""
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # State variables for optical flow
        self.old_gray = None
        self.p0 = None
        self.mask = None
        
        # Supervision annotators for drawing
        self.box_annotator = config.BOX_ANNOTATOR
        self.label_annotator = config.RICH_LABEL_ANNOTATOR

    def _get_features_in_rois(self, gray_frame, boxes):
        """Finds good features (corners) only within person bounding boxes."""
        all_keypoints = []
        for (x1, y1, x2, y2) in boxes:
            roi_gray = gray_frame[y1:y2, x1:x2]
            keypoints = cv2.goodFeaturesToTrack(roi_gray, mask=None, **config.SHI_TOMASI_PARAMS)
            if keypoints is not None:
                # Adjust keypoint coordinates to be relative to the full frame
                keypoints[:, 0, 0] += x1
                keypoints[:, 0, 1] += y1
                all_keypoints.extend(keypoints)
        return np.array(all_keypoints, dtype=np.float32) if all_keypoints else None

    def process_frame(self, frame, frame_number):
        """
        Processes a single video frame to detect violence.
        
        Args:
            frame (np.ndarray): The input video frame.
            frame_number (int): The current frame number.
            
        Returns:
            tuple: A tuple containing the annotated person detection frame
                   and the annotated violence detection frame.
        """
        # --- 1. Person Detection ---
        results = self.model(frame, stream=True, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        person_detections = sv.Detections.from_ultralytics(next(results))
        # Filter for 'person' class only (class_id=0 for COCO)
        person_detections = person_detections[person_detections.class_id == 0]
        
        # Create an annotated frame for the person detection window
        person_annotated_frame = frame.copy()
        labels = [f"Person {confidence:.2f}" for confidence in person_detections.confidence]
        person_annotated_frame = self.box_annotator.annotate(scene=person_annotated_frame, detections=person_detections)
        person_annotated_frame = self.label_annotator.annotate(scene=person_annotated_frame, detections=person_detections, labels=labels)
        
        # --- 2. Motion and Pose Analysis ---
        violence_annotated_frame = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.mask is None:
            self.mask = np.zeros_like(frame)
            
        motion_energy = 0
        
        # Initialize or update features to track
        if self.p0 is None or len(self.p0) == 0:
            self.p0 = self._get_features_in_rois(gray_frame, person_detections.xyxy.astype(int))
            self.mask = np.zeros_like(frame)
            self.old_gray = gray_frame.copy()

        # Calculate optical flow only if we have features to track
        if self.old_gray is not None and self.p0 is not None and len(self.p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, self.p0, None, **config.LK_PARAMS)
            
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    motion_vector = np.array([a - c, b - d])
                    motion_energy += np.linalg.norm(motion_vector)
                    
                    # Draw optical flow lines
                    cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    cv2.circle(violence_annotated_frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                
                # Update points for the next frame
                self.p0 = good_new.reshape(-1, 1, 2)
        
        # Add the flow visualization to the frame
        violence_annotated_frame = cv2.add(violence_annotated_frame, self.mask)

        # --- 3. Violence Heuristic ---
        people_boxes = person_detections.xyxy.astype(int)
        if len(people_boxes) > 0:
            for (x1, y1, x2, y2) in people_boxes:
                person_roi_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(person_roi_rgb)
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    mp_pose = mp.solutions.pose.PoseLandmark
                    
                    # Calculate hand speed (simplified metric)
                    left_hand_speed = abs(landmarks[mp_pose.LEFT_WRIST].y - landmarks[mp_pose.LEFT_ELBOW].y)
                    right_hand_speed = abs(landmarks[mp_pose.RIGHT_WRIST].y - landmarks[mp_pose.RIGHT_ELBOW].y)
                    hand_speed = (left_hand_speed + right_hand_speed) * 100 # Scale for better thresholding
                    
                    # Apply Violence Rule
                    if motion_energy > config.MOTION_ENERGY_THRESHOLD and hand_speed > config.HAND_SPEED_THRESHOLD:
                        log_violence_event(frame_number, motion_energy, hand_speed)
                        cv2.putText(violence_annotated_frame, config.VIOLENCE_LABEL, 
                                    (50, 50), config.FONT, 1, config.VIOLENCE_TEXT_COLOR, 3)
                        break # Only need to detect once per frame

        # --- 4. Update State for Next Frame ---
        self.old_gray = gray_frame.copy()
        # Periodically refresh features to get better tracking
        if frame_number % 10 == 0:
            self.p0 = self._get_features_in_rois(gray_frame, people_boxes)
        
        return person_annotated_frame, violence_annotated_frame

# -------------------------------------------------------------