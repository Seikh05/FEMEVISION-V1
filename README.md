# 🚨 Real-time Violence and Aggression Detection

This project, developed by **Team FemeVision**, is a real-time violence detection system built with Python and Streamlit. It uses a combination of computer vision techniques to identify potential aggression from a live webcam feed or an uploaded video file.

## Features
-   **Person Detection:** Uses the **YOLOv8** model to accurately detect people in the frame.
-   **Motion Analysis:** Employs **Shi-Tomasi corner detection** and **Lucas-Kanade optical flow** to analyze the magnitude and chaos of movement.
-   **Pose Heuristics:** Leverages **MediaPipe Pose** to analyze body language, such as hand speed, as a secondary indicator of aggression.
-   **Dual-Source Input:** Can process both a **live webcam feed** and **uploaded video files** (`.mp4`, `.mov`, etc.).
-   **Web Interface:** Built with **Streamlit** for a clean, interactive user experience.

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Seikh05/your-repo-name.git](https://github.com/Seikh05/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create a virtual environment:**
    ```bash
    # It is recommended to use Python 3.9, 3.10, or 3.11
    py -3.10 -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ````

---

### **Step 4: Create a New Repository on GitHub**

1.  Go to [GitHub.com](https://github.com/) and log in.
2.  Click the **+** icon in the top-right corner and select **"New repository"**.
3.  **Repository name:** `violence-detection-streamlit` (or any name you like).
4.  **Description:** "A real-time violence detection system using YOLO, Optical Flow, and Streamlit."
5.  Keep it **Public**.
6.  **IMPORTANT:** **Do NOT** check the boxes for "Add a README file", "Add .gitignore", or "Choose a license". You have already created these files.
7.  Click **"Create repository"**.

On the next page, GitHub will show you a URL under "Quick setup". Copy this URL. It will look like this: `https://github.com/your-username/violence-detection-streamlit.git`.

---

### **Step 5: Upload Your Code**

Now, open your terminal or command prompt and run these commands from inside your `FEMEVISION-V1` folder.

1.  **Initialize Git:**
    ```bash
    git init
    ```

2.  **Add all your project files to be tracked:**
    ```bash
    git add .
    ```

3.  **Make your first commit (a snapshot of your code):**
    ```bash
    git commit -m "Initial commit: Project setup and core logic"
    ```

4.  **Set your main branch name to `main` (the new standard):**
    ```bash
    git branch -M main
    ```

5.  **Connect your local folder to the GitHub repository you created:**
    (Replace the URL with the one you copied from GitHub)
    ```bash
    git remote add origin [https://github.com/your-username/violence-detection-streamlit.git](https://github.com/your-username/violence-detection-streamlit.git)
    ```

6.  **Push your code to GitHub:**
    ```bash
    git push -u origin main
    ```

That's it! 🎉 If you refresh your GitHub repository page, you will see all your files. Your project is now published and ready to be shared.