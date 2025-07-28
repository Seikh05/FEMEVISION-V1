# FILE: app/ui.py
# type: ignore

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time  # To keep track of frames for the processor

from app.processing import ViolenceProcessor
from utils import config

def run_file_analysis(video_path, placeholders):
    """This function now only handles uploaded video files."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return

    processor = ViolenceProcessor()
    frame_count = 0
    st.session_state.run = True

    while cap.isOpened() and st.session_state.get('run', False):
        ret, frame = cap.read()
        if not ret:
            st.success("Video processing finished.")
            break

        frame_count += 1
        if frame_count % config.FRAME_SKIP_RATE != 0:
            continue

        person_frame, violence_frame = processor.process_frame(frame, frame_count)

        placeholders['original'].image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        placeholders['person'].image(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        placeholders['violence'].image(cv2.cvtColor(violence_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
    st.session_state.run = False

def main_ui():
    """Defines the Streamlit user interface."""
    st.set_page_config(page_title="Real-time Violence Detection", layout="wide")
    st.title("🚨 Real-time Violence and Aggression Detection")
    st.write(
        "This application uses computer vision to detect potential violence. "
        "Use the camera input for live analysis or upload a video file."
    )

    # --- Sidebar for File Upload ---
    st.sidebar.title("Configuration")
    st.sidebar.header("File Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    analyze_file_button = st.sidebar.button("Analyze Uploaded File")
    stop_analysis_button = st.sidebar.button("Stop Analysis", key="stop")
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Team FemeVision.")

    # --- Main Area for Video Display ---
    col1, col2, col3 = st.columns(3)
    col1.header("Live Footage")
    col2.header("Person Detection (YOLOv8)")
    col3.header("Violence Analysis")
    
    # Placeholders for the output windows
    person_placeholder = col2.empty()
    violence_placeholder = col3.empty()

    # Session state initialization
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'processor' not in st.session_state:
        st.session_state.processor = ViolenceProcessor()

    if stop_analysis_button:
        st.session_state.run = False

    # --- Logic for handling inputs ---
    
    # Use st.camera_input for live webcam feed
    img_file_buffer = col1.camera_input("Enable Webcam", key="camera")
    
    if img_file_buffer:
        # If there's a camera input, prioritize it.
        st.session_state.frame_count += 1
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process and display the single frame from the camera
        person_frame, violence_frame = st.session_state.processor.process_frame(frame, st.session_state.frame_count)
        person_placeholder.image(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        violence_placeholder.image(cv2.cvtColor(violence_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    elif analyze_file_button and uploaded_file:
        # If no camera, but a file is uploaded and analyze is clicked
        placeholders = {
            'original': col1.empty(),
            'person': person_placeholder,
            'violence': violence_placeholder
        }
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        run_file_analysis(video_path, placeholders)
        os.remove(video_path)
    
    else:
        # Default placeholder when idle
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "System Idle", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        person_placeholder.image(placeholder_img, use_container_width=True)
        violence_placeholder.image(placeholder_img, use_container_width=True)
        if not img_file_buffer:
             col1.image(placeholder_img, use_container_width=True)

# -------------------------------------------------------------