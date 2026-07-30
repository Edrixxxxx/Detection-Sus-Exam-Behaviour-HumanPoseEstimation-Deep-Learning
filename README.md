# Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![UI Framework](https://img.shields.io/badge/PyQt6-Desktop_App-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange.svg)]()

> **Academic Research Project**  
> Don Mariano Marcos Memorial State University (DMMMSU) — South La Union Campus  
> Department of Computer Science | BS Computer Science (AY 2025–2026)

---

## 👥 Project Team & Contributors

### Research Group Members
| # | Full Name | Student ID | Role / Focus Area | Contact Email |
|---|---|---|---|---|
| 1 | **Edrich Josh Mabalot** | `[202X-XXXXX]` | Leader | `member1@dmmmsu.edu.ph` |
| 2 | **Jake Gutoc** | `[202X-XXXXX]` | Member | `member2@dmmmsu.edu.ph` |
| 3 | **Jefferson Italia** | `[202X-XXXXX]` | Member | `member3@dmmmsu.edu.ph` |
| 4 | **Kimberly Narval** | `[202X-XXXXX]` | Member | `member4@dmmmsu.edu.ph` |
| 5 | **Ramiel Angelo Dominic Tique** | `[202X-XXXXX]` | Member | `member5@dmmmsu.edu.ph` |

### Academic Supervision
- **Thesis Adviser:** `[Adviser Name / Title]`
- **Department:** Department of Computer Science, DMMMSU-SLUC

---

## 📌 Project Overview

This project presents a **real-time, desktop-based automated proctoring surveillance system** designed to detect suspicious examinee behaviors during physical classroom examinations. Using a **dual-camera setup** and a **6-stage hybrid deep learning and heuristic pipeline**, the system analyzes physical postures, facial orientations, hand gestures, spatial examinee interactions, and seat movements in real time.

### Key Highlights
- **Desktop Architecture (PyQt6):** Built natively for desktop deployment to utilize direct GPU hardware access, achieve sub-50ms per-frame pipeline latency, ensure offline capability, and guarantee student data privacy.
- **Dual-Camera Alignment:** Combines an overhead camera (Camera 1: Middle Top View) and a side camera (Camera 2: Top Side View) with $3 \times 3$ Homography spatial warping to eliminate physical body and desk occlusions.
- **Hybrid Classification Engine:** Combines deterministic spatial-temporal heuristics for head movements (*side glancing, head down, standing*) with an XGBoost decision tree classifier for manual cheating gestures (*passing notes, hand signaling*).
- **ArcFace Facial Re-ID:** Performs automated student identification and seat-swap detection using 512-dimensional ArcFace (`antelopev2`) facial embeddings matched against baseline rosters every 10 seconds.
- **Platt-Calibrated Confidence:** Applies Platt scaling logistic transformations to raw classifier logits to convert model scores into well-calibrated posterior probabilities $P_{\text{calibrated}} \in [0.0, 1.0]$.
- **Graduated 3-Tier Alert System:** Dispatches graduated proctor alerts (**Yellow**, **Orange**, **Red**) based on a 3-second (9-sample) sliding window temporal consensus, automatically capturing timestamped bounding-box evidence snapshots for high-severity violations.

---

## 🏗 System Architecture & Pipeline

```
Camera 1 (Middle Top 1080p @ 30fps) ──┐
Camera 2 (Top Side 1080p @ 30fps)   ──┴──► ┌────────────────────────────────────────┐
                                           │ Stage 1: Capture & Pre-processing      │
                                           │ - Laplacian Blur Check (Var >= 50)     │
                                           │ - 64-bit pHash Deduplication (Dist>=5) │
                                           │ - 640x640 Letterboxing & Homography   │
                                           └───────────────────┬────────────────────┘
                                                               │ (Clean Tensors @ 3 FPS)
                                                               ▼
                                           ┌────────────────────────────────────────┐
                                           │ Stage 2: Pose, Tracking & Re-ID        │
                                           │ - FP16 YOLOv26s-pose (17 keypoints)    │
                                           │ - ByteTrack Motion Association         │
                                           │ - ArcFace Face Match & Seat-Swap Check │
                                           │ - 60-Frame State History Buffer (20s)  │
                                           └───────────────────┬────────────────────┘
                                                               │ (Pose Trajectories)
                                                               ▼
                                           ┌────────────────────────────────────────┐
                                           │ Stage 3: 28-Feature Extraction         │
                                           │ - Static Posture Geometry (F1 - F16)   │
                                           │ - Temporal Movement Dynamics (F17-F22) │
                                           │ - Spatial Context Interactions (F23-F26)│
                                           │ - Inter-Shoulder Body Normalization    │
                                           └───────────────────┬────────────────────┘
                                                               │ (Feature Vectors)
                                                               ▼
                                           ┌────────────────────────────────────────┐
                                           │ Stage 4: Hybrid Behaviour Classifier   │
                                           │ - Head Pose Heuristic Rules            │
                                           │ - Hand Gesture XGBoost Classifier      │
                                           │ - Workspace Object Heuristics          │
                                           │ - Platt Scaling Sigmoid Calibration    │
                                           └───────────────────┬────────────────────┘
                                                               │ (Calibrated Probabilities)
                                                               ▼
                                           ┌────────────────────────────────────────┐
                                           │ Stage 5: Temporal Consensus & Alerting │
                                           │ - 3-Second Sliding Buffer Consensus    │
                                           │ - Graduated Alert Engine               │
                                           │ - Evidence Snapshot & Video Archiving  │
                                           └───────────────────┬────────────────────┘
                                                               │ (Dispatched Alerts)
                                                               ▼
                                           ┌────────────────────────────────────────┐
                                           │ Stage 6: Real-Time PyQt6 Proctor UI    │
                                           │ - Live Bounding Box Video Overlays     │
                                           │ - Interactive Alert Feed & Audio Alarm │
                                           │ - Post-Exam Compliance Audit Report    │
                                           └────────────────────────────────────────┘
```

---

## 📊 28-Feature Set Reference

All distance metrics are normalized by the examinee's **inter-shoulder width** to achieve scale invariance across varying student heights and camera distances:

| Group | Feature IDs | Description |
|---|---|---|
| **Head Pose** | Features 1–5 | Head Yaw, Pitch, Roll angles, lateral displacement, and head-shoulder vertical ratio. |
| **Hand Position** | Features 6–10 | Left/Right wrist-shoulder distances, average hand extension, and wrist Y-coordinates relative to desk height. |
| **Torso Geometry** | Features 11–12 | Torso lateral lean angle and vertical spine compression ratio. |
| **Limb Segments** | Features 13–16 | Left/Right upper arm (shoulder-elbow) and forearm (elbow-wrist) segment lengths. |
| **Temporal Dynamics** | Features 17–22 | Yaw velocity ($\Delta \text{Yaw}/\Delta t$), yaw acceleration, wrist speed ($\text{px/sec}$), head persistence duration, turn frequency, and hand spatial variance over 60 frames. |
| **Contextual** | Features 23–26 | Neighbor wrist distance, mutual head turn flag, seat grid position, and surrounding neighbor count. |
| **Confidence** | Features 27–28 | Mean keypoint detection confidence $c_{\text{mean}}$ and person detection confidence. |

---

## 🚨 Graduated Alert Threshold Matrix

| Alert Level | Condition / Trigger Rule | Proctor UI Action | Evidence Artifact Captured |
|---|---|---|---|
| **🟡 Yellow Alert** *(Low Severity)* | Single detection ($P_{\text{calibrated}} > 0.50$) or $< 30\%$ of 3-second buffer flagged. | Silent logging to session database; subtle indicator. | Session log entry |
| **🟠 Orange Alert** *(Medium Severity)* | $30\%\text{--}60\%$ of 3-second buffer flagged or behavior sustained $\ge 3$ seconds. | Visual sidebar pop-up toast notification on proctor UI. | Timestamped log + snapshot |
| **🔴 Red Alert** *(High Severity)* | $> 60\%$ of 3-second buffer flagged with average confidence $> 0.70$. | Fullscreen Red flashing border, audible alarm chime, push notification. | Timestamped JPEG bounding box screenshot + 10s MP4 clip |

---

## 📁 Repository Documentation Index

All system architecture diagrams, specifications, charts, and data models are documented inside the **[Charts & Graphss](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss)** folder:

| Document File | Topic / Specification Content |
|---|---|
| **[DFD.md](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss/DFD.md)** | **Data Flow Diagram:** Context Diagram (Level 0), Major System Processes (Level 1), and Sub-Processes (Level 2). |
| **[ERD.md](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss/ERD.md)** | **Entity Relationship Diagram:** Logical database schema, 15 entity definitions, data types, PK/FK constraints, and storage estimation. |
| **[HIPO.md](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss/HIPO.md)** | **Hierarchy Plus Input-Process-Output:** Visual Table of Contents (VTOC) and detailed IPO specifications for all 21 sub-modules. |
| **[Structured_Chart.md](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss/Structured_Chart.md)** | **Structure Chart:** Top-down program execution hierarchy, module call trees, Data Couples, and Control Couples dictionaries. |
| **[Pseudo_Code.md](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss/Pseudo_Code.md)** | **Pseudocode Specifications:** Algorithmic pseudocode covering multi-threaded loops, YOLO inference, tracking, feature extraction, and alerting. |
| **[Structured_English.md](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss/Structured_English.md)** | **Structured English:** Narrative operational logic using restricted English constructs (`IF-THEN-ELSE`, `FOR EACH`) for all 6 pipeline stages. |
| **[Data_Dictionary.md](file:///c:/Users/gelin/OneDrive/Desktop/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/Charts%20%26%20Graphss/Data_Dictionary.md)** | **Data Dictionary:** Centralized metadata definitions for data stores `D1a`–`D8`, 28 feature elements, logits, and data flow mappings. |

---

## 🛠 System Environment & Dependencies

### Hardware Requirements
- **CPU:** Intel Core i7 / AMD Ryzen 7 (8+ cores recommended)
- **GPU:** NVIDIA GeForce RTX 3060 / 4060 or higher (CUDA 11.8 support, $\ge 8$ GB VRAM)
- **RAM:** 16 GB DDR4 / DDR5
- **Cameras:** Dual 1080p USB / RTSP Webcams (30 FPS capability)

### Software Prerequisites
- **Operating System:** Windows 10 / 11 (64-bit)
- **Python Version:** Python 3.10.x
- **Core Libraries:**
  ```bash
  # Deep Learning & Computer Vision
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ultralytics               # YOLOv26s-pose backend
  opencv-python            # Video capture & homography warping
  insightface              # ArcFace antelopev2 face embeddings

  # Machine Learning & Analytics
  xgboost                  # Hand gesture classifier
  scikit-learn             # Platt scaling calibration & metrics
  numpy pandas             # Feature matrix manipulation

  # Desktop User Interface
  PyQt6                    # Desktop GUI application framework
  ```

---

## ⚡ Quick Start Guide

### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/YourUsername/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning.git
cd Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configuration & Model Initialization
1. Ensure dual cameras are connected and updated in `config.yaml` stream URLs.
2. Download pre-trained `YOLOv26s-pose.pt` and place in `Main/models/`.
3. Load student facial roster images into `Datasets/roster/` for ArcFace embedding generation.

### 3. Launching Application
```bash
python Main/main.py
```

---

## 📜 License & Citation

This project is developed as an academic thesis requirement for the Bachelor of Science in Computer Science program at DMMMSU - South La Union Campus. All rights reserved.

---

*Last updated: Academic Year 2025–2026*