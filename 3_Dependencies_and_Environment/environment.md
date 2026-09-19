# Environment & Dependency Specifications

This directory contains configuration files and specifications for the computing environment, hardware requirements, and external software libraries required to run the automated examination proctoring system.

---

## 1. Hardware Requirements

| Hardware Component | Minimum Requirement | Recommended Specification |
|---|---|---|
| **CPU** | Intel Core i5 / AMD Ryzen 5 (4+ cores) | Intel Core i7 / AMD Ryzen 7 (8+ cores) |
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB VRAM) | NVIDIA GeForce RTX 3060 / 4060 or higher ($\ge 8$ GB VRAM) |
| **CUDA Capability** | CUDA 11.8 or CUDA 12.x | CUDA 12.4+ |
| **System RAM** | 8 GB DDR4 | 16 GB DDR4/DDR5 |
| **Cameras** | Single 720p USB Webcam (Prototype) | Dual 1080p @ 30 FPS USB / RTSP Webcams (Full System) |

---

## 2. Software Prerequisites

- **Operating System:** Windows 10 / 11 (64-bit)
- **Python Runtime:** Python 3.10 to Python 3.13 (64-bit)
- **CUDA Toolkit & cuDNN:** Compatible NVIDIA display drivers and CUDA runtime

---

## 3. Python Package Dependencies

The primary package requirements are listed in [`requirements.txt`](file:///c:/Users/USER/Desktop/SchoolWorks/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/3_Dependencies_and_Environment/requirements.txt):

```text
ultralytics>=8.3.0       # Human pose estimation backend (YOLO26s/YOLO11 pose models)
torch>=2.0.0             # Deep learning tensor computation & PyTorch neural networks
torchvision>=0.15.0       # Vision datasets and transformations
opencv-python>=4.8.0     # Video capture, image transformations, HUD drawing
numpy>=1.22.0            # High-performance multi-dimensional array manipulation
matplotlib>=3.7.0        # Generation of loss curves and confusion matrix plots
lap>=0.5.12              # Linear Assignment Problem solver used by ByteTrack
pyyaml>=6.0              # YAML configuration parsing
pillow>=10.0.0           # Image handling and GUI graphics support
```

### Optional & Future Deployment Packages
- `PyQt6`: Native desktop GUI framework for the proctoring dashboard.
- `insightface`: ArcFace facial recognition embeddings (`antelopev2`).
- `xgboost`: Gradient boosted decision trees for hand gesture classification.
- `scikit-learn`: Platt scaling probability calibration and diagnostic metrics.

---

## 4. Quick Environment Setup

```bash
# Navigate to the repository root
cd Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```
