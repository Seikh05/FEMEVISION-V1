# type: ignore
# FILE: app/ui.py
# This file is now much simpler. It just starts the webrtc_streamer.
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import tempfile
import os

# We now import the transformer class from the processing file
from app.processing import ViolenceVideoTransformer

def main_ui():
    st.set_page_config(page_title="Real-time Violence Detection", layout="wide")
    st.title("🚨 Real-time Violence and Aggression Detection")
    st.write(
        "This application uses computer vision to detect potential violence in real-time from your webcam."
        " **Click START to begin.**"
    )

    # The main real-time streaming component
    webrtc_streamer(
        key="violence-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=ViolenceVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.sidebar.title("Configuration")
    st.sidebar.info("The application processes the live webcam feed. The combined output is shown in the main window.")
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Team FemeVision.")
