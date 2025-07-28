# -------------------------------------------------------------
# FILE: utils/config.py
# This file centralizes all your parameters and settings for easy tuning.

import cv2
import supervision as sv

# --- Model and Asset Paths ---
YOLO_MODEL_PATH = "yolov8n.pt"
FONT_PATH = "assets/Roboto-VariableFont_wdth,wght.ttf" # Make sure this path is correct

# --- Detection and Tracking Parameters ---
CONFIDENCE_THRESHOLD = 0.4
FRAME_SKIP_RATE = 2  # Process every Nth frame to improve performance

# --- Optical Flow (Shi-Tomasi & Lucas-Kanade) Parameters ---
SHI_TOMASI_PARAMS = dict(maxCorners=500, qualityLevel=0.3, minDistance=5, blockSize=4)
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# --- Violence Heuristic Thresholds ---
# These values are crucial and may need tuning based on your video source.
MOTION_ENERGY_THRESHOLD = 300.0
HAND_SPEED_THRESHOLD = 60.0

# --- Annotation and Display Settings ---
BOX_ANNOTATOR = sv.BoxCornerAnnotator(thickness=1)
RICH_LABEL_ANNOTATOR = sv.RichLabelAnnotator(text_position=sv.Position.CENTER)
VIOLENCE_LABEL = "VIOLENCE DETECTED!"
VIOLENCE_TEXT_COLOR = (0, 0, 255)  # Red in BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX




# import cv2
# import supervision as sv

# # --- Model and Asset Paths ---
# YOLO_MODEL_PATH = "yolov8n.pt"
# FONT_PATH = "assets/Roboto-VariableFont_wdth,wght.ttf" # Make sure this path is correct

# # --- Detection and Tracking Parameters ---
# CONFIDENCE_THRESHOLD = 0.4
# FRAME_SKIP_RATE = 2  # Process every Nth frame to improve performance

# # --- Optical Flow (Shi-Tomasi & Lucas-Kanade) Parameters ---
# SHI_TOMASI_PARAMS = dict(maxCorners=500, qualityLevel=0.3, minDistance=5, blockSize=4)
# LK_PARAMS = dict(
#     winSize=(15, 15),
#     maxLevel=2,
#     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
# )

# # --- Violence Heuristic Thresholds ---
# # These values are crucial and may need tuning based on your video source.
# MOTION_ENERGY_THRESHOLD = 500.0
# HAND_SPEED_THRESHOLD = 40.0

# # --- Annotation and Display Settings ---
# BOX_ANNOTATOR = sv.BoxCornerAnnotator(thickness=1)
# RICH_LABEL_ANNOTATOR = sv.RichLabelAnnotator(text_position=sv.Position.CENTER)
# VIOLENCE_LABEL = "VIOLENCE DETECTED!"
# VIOLENCE_TEXT_COLOR = (0, 0, 255)  # Red in BGR
# FONT = cv2.FONT_HERSHEY_SIMPLEX
