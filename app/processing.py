# type: ignore
# FILE: app/processing.py
# This file now contains the corrected lock and the full hand speed logic.

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












#############################################################
# Code for streamlit's webrtc camera input  processing      #      
#############################################################



# import cv2
# import numpy as np
# import supervision as sv
# from ultralytics import YOLO
# import mediapipe as mp
# import threading  # Correct import for the Lock
# from streamlit_webrtc import VideoTransformerBase
# import av

# from utils import config, logger

# # The main processing logic is now encapsulated in this class
# class ViolenceVideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         # Load models and settings here so they are not reloaded on every frame.
#         self.model = YOLO("yolov8n.pt")
#         self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.mp_pose = mp.solutions.pose # Alias for easy access to landmarks
#         self.box_annotator = sv.BoxCornerAnnotator(thickness=1, corner_length=10)
#         self.label_annotator = sv.LabelAnnotator()
#         self.frame_count = 0
#         self.old_gray = None
#         self.p0 = None
#         # This lock is important for thread safety when updating p0
#         self.lock = threading.Lock() # Correctly use threading.Lock

#     def process_single_frame(self, frame):
#         """Processes one frame of video."""
#         self.frame_count += 1
#         if self.frame_count % config.FRAME_SKIP_RATE != 0:
#             # On skipped frames, return blank overlays to avoid flicker
#             return frame, np.zeros_like(frame), np.zeros_like(frame)

#         # Create copies for different processing steps
#         person_detection_frame = frame.copy()
#         violence_detection_frame = frame.copy()
        
#         # --- Person Detection ---
#         results = self.model(frame, stream=True, conf=0.4, verbose=False)
#         people_boxes = []
#         for r in results:
#             detections = sv.Detections.from_ultralytics(r)
#             detections = detections[detections.class_id == 0]
#             people_boxes.extend(detections.xyxy.tolist())
            
#             labels = [f"{self.model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
#             person_detection_frame = self.box_annotator.annotate(scene=person_detection_frame, detections=detections)
#             person_detection_frame = self.label_annotator.annotate(scene=person_detection_frame, detections=detections, labels=labels)

#         # --- Motion & Violence Analysis ---
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         motion_energy = 0
#         violence_detected = False
        
#         with self.lock:
#             # Initialize old_gray on the first run
#             if self.old_gray is None:
#                 self.old_gray = gray_frame.copy()

#             # Calculate optical flow only if we have points from the previous frame
#             if self.p0 is not None and len(self.p0) > 0:
#                 p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, self.p0, None, **config.LK_PARAMS)
#                 if p1 is not None and st is not None:
#                     good_new = p1[st == 1]
#                     good_old = self.p0[st == 1]
#                     for new, old in zip(good_new, good_old):
#                         motion_energy += np.linalg.norm(new - old)
#                     # For the next frame, the new points are the good points from this frame
#                     self.p0 = good_new.reshape(-1, 1, 2)
#                 else:
#                     # If flow calculation fails, reset points
#                     self.p0 = None
            
#             # Condition to find new points: if we have no points or too few have survived.
#             if self.p0 is None or len(self.p0) < 10:
#                 mask = np.zeros_like(gray_frame)
#                 for x1, y1, x2, y2 in people_boxes:
#                     mask[int(y1):int(y2), int(x1):int(x2)] = 255
                
#                 # Find new features to track within the person masks
#                 if np.any(mask):
#                     self.p0 = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **config.SHI_TOMASI_PARAMS)
#                 else:
#                     self.p0 = None # Ensure p0 is None if no people are found

#             # --- Hand Speed Check (Pose Estimation) ---
#             if people_boxes:
#                 for box in people_boxes:
#                     x1, y1, x2, y2 = map(int, box)
#                     person_roi = frame[y1:y2, x1:x2]
#                     if person_roi.size == 0: continue
                    
#                     h_roi, w_roi, _ = person_roi.shape
#                     person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
#                     pose_results = self.pose.process(person_rgb)
                    
#                     if pose_results.pose_landmarks:
#                         landmarks = pose_results.pose_landmarks.landmark
#                         # Restore full hand speed logic
#                         left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
#                         right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
#                         left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
#                         right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]

#                         lw_pos = np.array([left_wrist.x * w_roi, left_wrist.y * h_roi])
#                         rw_pos = np.array([right_wrist.x * w_roi, right_wrist.y * h_roi])
#                         le_pos = np.array([left_elbow.x * w_roi, left_elbow.y * h_roi])
#                         re_pos = np.array([right_elbow.x * w_roi, right_elbow.y * h_roi])
                        
#                         hand_speed = np.linalg.norm(lw_pos - le_pos) + np.linalg.norm(rw_pos - re_pos)
                        
#                         if motion_energy > config.MOTION_ENERGY_THRESHOLD and hand_speed > config.HAND_SPEED_THRESHOLD:
#                             violence_detected = True
#                             break
            
#         # Update old_gray for the next frame's calculation
#         self.old_gray = gray_frame.copy()

#         if violence_detected:
#             logger.log_violence_event(self.frame_count)
#             cv2.putText(violence_detection_frame, "VIOLENCE DETECTED", (50, 50), 
#                         config.FONT, 1.5, config.VIOLENCE_BOX_COLOR, 3)

#         return frame, person_detection_frame, violence_detection_frame

#     def recv(self, frame):
#         """This method is called for each frame from the browser for webrtc."""
#         frm = frame.to_ndarray(format="bgr24")
#         original, person, violence = self.process_single_frame(frm)
        
#         h, w, _ = original.shape
#         combined_frame = np.zeros((h, w * 3, 3), dtype=np.uint8)
#         combined_frame[:, :w] = original
#         combined_frame[:, w:w*2] = person
#         combined_frame[:, w*2:] = violence
        
#         return av.VideoFrame.from_ndarray(combined_frame, format="bgr24")

