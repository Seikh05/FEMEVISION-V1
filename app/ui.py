# type: ignore
# FILE: app/ui.py

import streamlit as st
import cv2
import numpy as np
import tempfile  # Required for handling uploaded files
import os        # Required for path operations

from app.processing import ViolenceProcessor
from utils import config

def run_analysis(video_source, placeholders):
    """
    This new function contains the main analysis loop. It can accept either a
    webcam index (0) or a file path as its source.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error(f"Error: Could not open video source.")
        return

    processor = ViolenceProcessor()
    frame_count = 0
    st.session_state.run = True  # Flag to control the loop

    while cap.isOpened() and st.session_state.get('run', False):
        ret, frame = cap.read()
        if not ret:
            st.success("Video processing finished.")
            st.session_state.run = False
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
        "This application uses a combination of YOLO, Optical Flow, and MediaPipe Pose "
        "to detect potential violence in real-time or from an uploaded video file."
    )

    # --- Sidebar ---
    st.sidebar.title("Configuration")

    st.sidebar.header("Live Analysis")
    run_webcam = st.sidebar.button("Start Webcam")

    st.sidebar.header("File Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])
    analyze_file = st.sidebar.button("Analyze Uploaded File")

    st.sidebar.markdown("---")
    stop_analysis = st.sidebar.button("Stop Analysis", key="stop")

    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Team FemeVision.")

    # --- Main Area for Video Display ---
    col1, col2, col3 = st.columns(3)
    # Adding the headers back in!
    col1.header("Live Footage Window")
    col2.header("Person Detection Window (YOLOv8)")
    col3.header("Violence Analysis Window")
    placeholders = {
        'original': col1.empty(),
        'person': col2.empty(),
        'violence': col3.empty()
    }
    
    # Initialize session state for the run flag
    if 'run' not in st.session_state:
        st.session_state.run = False

    if stop_analysis:
        st.session_state.run = False

    if run_webcam:
        st.session_state.run = True

    if analyze_file:
        if uploaded_file is None:
            st.sidebar.warning("Please upload a video file first.")
        else:
            st.session_state.run = True
            
    # Central loop logic
    if st.session_state.get('run', False):
        if uploaded_file and analyze_file:
            # When analyzing a file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            
            run_analysis(video_path, placeholders)
            os.remove(video_path) # Clean up the temp file
        else:
            # When running the webcam
            run_analysis(0, placeholders)
    else:
        # Display a placeholder image when nothing is running
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "System Idle", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for placeholder in placeholders.values():
            placeholder.image(placeholder_img, use_container_width=True)

# -------------------------------------------------------------



    #   placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
    #     cv2.putText(placeholder_img, "Webcam is Off", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #     original_video_placeholder.image(placeholder_img, use_column_width=True)
    #     person_detection_placeholder.image(placeholder_img, use_column_width=True)
    #     violence_detection_placeholder.image(placeholder_img, use_column_width=True)