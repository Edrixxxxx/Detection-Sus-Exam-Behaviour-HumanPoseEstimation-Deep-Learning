# Desktop Application Entry Point (`Main`)

This module provides the graphical desktop application entrypoint for the **Real-Time Examination Behavior Proctoring System** based on **PyQt6**, **YOLO26s-Pose**, **ByteTrack**, and **PyTorch LSTM Classifier**.

---

## 🚀 How to Launch the UI

You can start the user interface in any of the following ways **without typing complex command-line arguments**:

### Method 1: Double-Click Batch File (No Terminal Required)
Simply double-click:
```
run_ui.bat
```
located at the root of the repository.

### Method 2: From Terminal
From the repository root directory:
```bash
python main.py
```
Or from within the `1_Source_Code` directory:
```bash
python main.py
```
Or directly from `1_Source_Code/Main`:
```bash
python main.py
```

---

## 🖥️ UI Capabilities & User Guide

### 1. Video Source Selection
- **📹 Live Webcam Mode**:
  - Automatically probes and lists available connected USB or integrated webcams.
  - Click **"Scan Cameras"** to refresh detected devices.
  - Option to enter a custom camera index or RTSP / IP camera URL (e.g. `rtsp://...`).
- **📁 Recorded Video Clip Mode**:
  - Click **"Browse Video File..."** to open any pre-recorded exam video clip (`.mp4`, `.avi`, `.mov`, `.mkv`).
  - Toggle **"Loop Playback"** to automatically replay the video clip continuously for repeatable testing.

### 2. Surveillance Action Controls
- **▶ Start Proctoring**: Initializes YOLO26s-pose estimation and behavioral classification.
- **⏸ Pause / Resume**: Pauses inference without closing or releasing the video source.
- **⏹ Stop**: Gracefully stops the video stream, closes background worker threads, and clears GPU memory.
- **📸 Capture Snapshot**: Instantly captures the current annotated video frame (with bounding boxes, skeleton keypoints, and classification labels) and saves it to `4_Data_and_Schema/evidence_snapshots/`.
- **🔴 Record Video**: Automatically writes the live annotated video output to a timestamped `.mp4` file.

### 3. Real-Time KPI Dashboard
- **Active Examinees**: Real-time count of simultaneously tracked students in the camera view.
- **Total Alerts**: Cumulative counter of flagged suspicious behaviors.
- **Pipeline FPS**: Real-time processing speed and latency benchmark.
- **Classroom Status**: Overall threat indicator (**NORMAL**, **ALERT**, or **CRITICAL**).

### 4. Interactive Live Alert Feed
- Real-time tabular event log displaying:
  - Timestamp (`HH:MM:SS`)
  - Examinee ID (`Student #1`, `Student #2`, etc.)
  - Detected Behavior (`SIDE GLANCING`, `PASSING OF NOTES`, `USE OF UNAUTHORIZED OBJECT`, `HAND SIGNAL`)
  - Model Confidence Percentage
  - Severity Level (**Yellow**, **Orange**, **Red**)
- **🔊 Chime on Red Alert**: Audible audio tone triggered on high-severity cheating alerts.
- **💾 Export CSV**: Save audit logs directly into an Excel/CSV spreadsheet report.
- **🗑️ Clear Log**: Reset the active alert session.

### 5. Configurable Settings & Thresholds
- **Model Adapter Selection**:
  - `Deep Learning: PyTorch LSTM` (Uses trained weights `best_lstm_model.pt`)
  - `Hybrid Ensemble: LSTM + Geometric Heuristics` (Combines neural prediction with geometry rules)
  - `Deterministic Geometric Heuristics` (Rule-based evaluation without neural training)
  - `Mock Simulation Stream` (Simulated test patterns for demonstrations)
- **Suspicious Threshold Slider**: Adjust detection sensitivity dynamically between 20% and 95%.
- **Display Toggles**: Easily show/hide bounding boxes, skeleton bones, status badges, and alert banners.
