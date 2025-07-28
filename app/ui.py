# type: ignore
# FILE: app/ui.py
# This file now has two tabs for Live and File analysis.
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import tempfile
import os
import cv2
import numpy as np

from app.processing import ViolenceVideoTransformer
from utils import config

def run_file_analysis(video_path, placeholders):
    """Handles the processing for an uploaded video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return

    # Use the same processing class
    processor = ViolenceVideoTransformer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("Video processing finished.")
            break

        original, person, violence = processor.process_single_frame(frame)

        # Display frames in their respective columns
        placeholders['original'].image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)
        placeholders['person'].image(cv2.cvtColor(person, cv2.COLOR_BGR2RGB), use_container_width=True)
        placeholders['violence'].image(cv2.cvtColor(violence, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()

def main_ui():
    st.set_page_config(page_title="Real-time Violence Detection", layout="wide")
    st.title("🚨 Real-time Violence and Aggression Detection")
    st.sidebar.write("Developed by Team FemeVision.")

    live_tab, file_tab = st.tabs(["Live Analysis (Webcam)", "File Analysis (Upload)"])

    # --- Live Analysis Tab ---
    with live_tab:
        st.header("Real-time Webcam Feed")
        st.write("Click START to begin processing your live webcam feed.")
        webrtc_streamer(
            key="violence-detection-live",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=ViolenceVideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.info("The three processed windows (Live, Person Detection, Violence Analysis) are combined into the single video feed above.")

    # --- File Analysis Tab ---
    with file_tab:
        st.header("Analyze a Video File")
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

        if uploaded_file:
            if st.button("Analyze Uploaded File"):
                col1, col2, col3 = st.columns(3)
                col1.header("Original Video")
                col2.header("Person Detection")
                col3.header("Violence Analysis")

                placeholders = {
                    'original': col1.empty(),
                    'person': col2.empty(),
                    'violence': col3.empty()
                }

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                
                with st.spinner('Analyzing video...'):
                    run_file_analysis(video_path, placeholders)
                
                os.remove(video_path) # Clean up the temp file


# -------------------------------------------------------------------