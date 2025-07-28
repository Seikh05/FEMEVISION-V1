# FILE: utils/logger.py
# A simple utility for logging detected violence events to a local file.

from datetime import datetime

LOG_FILE_PATH = "violence_log.txt"

def log_violence_event(frame_number, motion_energy, hand_speed):
    """Logs a violence event with a timestamp and relevant metrics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (
        f"ALERT! Timestamp: {timestamp}, Frame: {frame_number}, "
        f"Motion Energy: {motion_energy:.2f}, Hand Speed: {hand_speed:.2f}\n"
    )
    
    try:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(log_message)
        print(log_message, end="") # Also print to console for real-time feedback
    except Exception as e:
        print(f"Error: Could not write to log file. {e}")
        # -------------------------------------------------------------