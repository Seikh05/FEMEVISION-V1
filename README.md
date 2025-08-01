# 🚨 Real-time Violence Detection

A computer vision project by **Team FemeVision** that analyzes video to detect potential violence using motion and pose analysis.

![Streamlit App UI](https://github.com/Seikh05/FEMEVISION-V1/blob/main/assets/app_UI.jpg)

## Core Technologies

  - **Person Detection:** YOLOv8
  - **Motion Analysis:** Optical Flow (Shi-Tomasi & Lucas-Kanade)
  - **Pose Heuristics:** MediaPipe Pose
  - **Web Interface:** Streamlit

-----

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Seikh05/FEMEVISION-V1.git
    cd FEMEVISION-V1
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Use Python 3.9, 3.10, or 3.11
    py -3.10 -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the app:**

    ```bash
    streamlit run app.py
    ```