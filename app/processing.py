# FILE: app/processing.py
# This file is now a class specifically for streamlit-webrtc.
# type: ignore
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from utils import config, logger

# The main processing logic is now encapsulated in this class
class ViolenceVideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Load models and settings here so they are not reloaded on every frame.
        self.model = YOLO("yolov8n.pt")
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.box_annotator = sv.BoxCornerAnnotator(thickness=1, corner_length=10)
        self.label_annotator = sv.LabelAnnotator()
        self.frame_count = 0
        self.old_gray = None
        self.p0 = None
        # This lock is important for thread safety when updating p0
        self.lock = mp.Lock()

    def process_single_frame(self, frame):
        """Processes one frame of video."""
        self.frame_count += 1
        if self.frame_count % config.FRAME_SKIP_RATE != 0:
            # If we skip, we still need to return a valid frame layout
            return frame, np.zeros_like(frame), np.zeros_like(frame)

        # Create copies for different processing steps
        person_detection_frame = frame.copy()
        violence_detection_frame = frame.copy()
        
        # --- Person Detection ---
        results = self.model(frame, stream=True, conf=0.4, verbose=False)
        people_boxes = []
        for r in results:
            detections = sv.Detections.from_ultralytics(r)
            # Filter for 'person' class only (class_id=0)
            detections = detections[detections.class_id == 0]
            people_boxes.extend(detections.xyxy.tolist())
            
            # Annotate the person detection frame
            labels = [f"{self.model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
            person_detection_frame = self.box_annotator.annotate(scene=person_detection_frame, detections=detections)
            person_detection_frame = self.label_annotator.annotate(scene=person_detection_frame, detections=detections, labels=labels)

        # --- Motion & Violence Analysis ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_energy = 0
        violence_detected = False
        
        with self.lock:
            if self.old_gray is not None and self.p0 is not None and len(self.p0) > 0:
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, self.p0, None, **config.LK_PARAMS)
                
                if p1 is not None:
                    good_new = p1[st == 1]
                    for new in good_new:
                        motion_vector = new - self.p0[st == 1][np.where((p1 == new).all(axis=1))[0][0]]
                        motion_energy += np.linalg.norm(motion_vector)

                    # Update points for next frame
                    self.p0 = good_new.reshape(-1, 1, 2)

            # --- Hand Speed Check (Pose Estimation) ---
            if people_boxes:
                for box in people_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size == 0: continue
                    
                    person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(person_rgb)
                    
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        # Get landmarks for hands and elbows
                        # ... (your hand speed logic) ...
                        hand_speed = 0 # Placeholder for your calculation
                        
                        if motion_energy > config.MOTION_ENERGY_THRESHOLD and hand_speed > config.HAND_SPEED_THRESHOLD:
                            violence_detected = True
                            break
            
            # Find new corners to track for the next frame
            mask = np.zeros_like(gray_frame)
            for x1, y1, x2, y2 in people_boxes:
                mask[int(y1):int(y2), int(x1):int(x2)] = 255
            self.p0 = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **config.SHI_TOMASI_PARAMS)

        self.old_gray = gray_frame.copy()

        if violence_detected:
            logger.log_violence_event(self.frame_count)
            cv2.putText(violence_detection_frame, "VIOLENCE DETECTED", (50, 50), 
                        config.FONT, 1.5, config.VIOLENCE_BOX_COLOR, 3)

        return frame, person_detection_frame, violence_detection_frame

    def recv(self, frame):
        """This method is called for each frame from the browser."""
        frm = frame.to_ndarray(format="bgr24")
        
        original, person, violence = self.process_single_frame(frm)

        # Combine the three windows into one single frame to send back
        h, w, _ = original.shape
        combined_frame = np.zeros((h, w * 3, 3), dtype=np.uint8)
        combined_frame[:, :w] = original
        combined_frame[:, w:w*2] = person
        combined_frame[:, w*2:] = violence
        
        return av.VideoFrame.from_ndarray(combined_frame, format="bgr24")