# Automated Examination Behavior Proctoring System
### Powered by YOLO26s-Pose, ByteTrack, and Sequential PyTorch LSTM Classifier

---

## 1. Overview & Architecture

This project is an end-to-end intelligent proctoring prototype designed to automatically identify and distinguish **5 distinct examination behaviors** from raw video footage or live webcam streams using Human Pose Estimation (HPE) and sequential Deep Learning.

### Target Examination Classes (5 Classes)
1. **`normal`**: Natural seated exam behavior (calm writing, reading question sheets, steady head and posture).
2. **`hand_signal`**: Hand raises, unusual waving, elevated wrist gesturing, or finger signaling towards peers.
3. **`passing_of_notes`**: Arm stretching laterally across desks to pass notes, cheat sheets, or supplies.
4. **`side_glancing`**: Head yawing/turning sideways to peek at a neighboring student's exam paper or desk.
5. **`use_of_unauthorized_object`**: Head pitching deeply downwards towards the lap or desk drawer (e.g. concealed smartphone, notes).

```
+-----------------------------------------------------------------------------------+
|                                Raw Video / Live Camera                            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        1. Human Pose Estimation (YOLO26s-Pose)                    |
|             Extracts 17 COCO keypoints (x, y, confidence) per student             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         2. Multi-Student Tracking (ByteTrack)                     |
|           Maintains persistent Student Track IDs across successive frames         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                  3. Keypoint Normalization & Sliding Temporal Window              |
|        - Invariant to desk distance, camera perspective, and student height       |
|        - FIFO buffer per student of length T = 30 frames (~1.0 sec at 30 fps)     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       4. Sequential Deep Learning (PyTorch LSTM)                  |
|          Bidirectional LSTM + Temporal Pooling + 5-Class MLP Classifier:          |
|    [0: normal, 1: hand_signal, 2: passing_of_notes, 3: side_glancing, 4: object]  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             5. Real-Time Visual Proctor HUD                       |
|     - Bounding boxes + Color-coded Pose Skeletons                                 |
|     - Color-coded badges & glowing alert banners for each specific violation      |
|     - Debounced alert suppression to prevent single-frame false alarms            |
|     - Real-time FPS, active examinee count, and alert incident statistics         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Project Directory Structure

```
prototype/
├── clipper.py                # Video clipper and labeler GUI (keyboard-driven segmentation)
├── clipper_config.json       # Clipper configuration and class key bindings
├── config.py                 # Central configurations (5 classes, paths, hyperparameters)
├── model.py                  # PyTorch LSTM classification model architecture
├── dataset.py                # Dataset loader, pose normalization, and 70-15-15 split
├── tracker.py                # YOLO26s-pose + ByteTrack tracking and buffer manager
├── extract_dataset.py        # Video sequence feature extraction pipeline for 5 classes
├── train.py                  # Model training, early stopping, and 70-15-15 test evaluation
├── inference.py              # Real-time live webcam and video file proctoring engine
├── test_prototype.py         # Automated verification suite (5-class testing)
├── INTEGRATION.md            # Clipper integration documentation and AGPL-3.0 attribution
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw_videos/           # Place your raw video clips in their respective class folders
│   │   ├── normal/
│   │   ├── hand_signal/
│   │   ├── passing_of_notes/
│   │   ├── side_glancing/
│   │   └── use_of_unauthorized_object/
│   └── processed/
│       └── exam_dataset.npz  # Extracted sequences ready for LSTM training
├── weights/
│   └── best_lstm_model.pt    # Saved best checkpoint of the trained 5-class LSTM classifier
└── reports/
    ├── confusion_matrix.png  # 5x5 Confusion matrix plot on 15% held-out test data
    ├── training_curves.png   # Train/Validation loss and accuracy curves
    └── test_evaluation_report.txt # Detailed metrics (Precision, Recall, F1, Support)
```

---

## 3. Environment & Prerequisites

The prototype runs on **Windows 11** with **Python 3.13** and **PyTorch with CUDA GPU acceleration** (NVIDIA RTX).

### Installation
To install or verify dependencies:
```powershell
pip install -r requirements.txt
```

Verify GPU acceleration:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Complete Step-by-Step Guide

### STEP 1: Label and Trim Raw Videos (Using `clipper.py`)

You can easily label long exam recordings using the integrated keyboard-driven **Video Clipper GUI** (`clipper.py`). The tool cuts video intervals and exports short clips directly into `data/raw_videos/<class>/`.

```powershell
python clipper.py
```

#### Clipper GUI Workflow:
1. Click **"Load Video"** and select a raw examination recording (`.mp4`, `.avi`, `.mov`).
2. Scrub to the start of a behavior segment using the slider or `<<` / `>>` step buttons (or press `Space` to play/pause).
3. Press a single keyboard shortcut to select the behavior class:
   - `n` = `normal`
   - `h` = `hand_signal`
   - `p` = `passing_of_notes`
   - `s` = `side_glancing`
   - `u` = `use_of_unauthorized_object`
4. Click **"Set Start Frame"** at the beginning of the behavior.
5. Scrub forward to the end of the behavior and click **"Set End Frame"**.
6. Click **"Add Clip to Queue"** (or press `Delete`/`Backspace` to remove a clip).
7. Repeat for all behaviors in the video.
8. Click **"Extract All Queued Clips"** to batch-export every segment directly into `data/raw_videos/<class>/`!

---

#### Target Class Behavior Definitions:

1. **`data/raw_videos/normal/`**:
   - Students writing calmly on test sheets.
   - Reading question papers with neutral head orientation.
   - Thinking or holding chin steadily.
   - Natural page flipping.

2. **`data/raw_videos/hand_signal/`**:
   - Students raising one hand slightly above desk level to signal a peer.
   - Distinct finger counting or gesturing across desks.
   - Waving or tapping hands to catch attention.

3. **`data/raw_videos/passing_of_notes/`**:
   - Arm reaching across desk aisles or sideways towards a neighboring student.
   - Passing an eraser, paper slip, or pen under or over the desk.

4. **`data/raw_videos/side_glancing/`**:
   - Turning the head horizontally (left or right) repeatedly towards the adjacent examinee's desk.
   - Stretching the neck or leaning sideways to look at another student's exam sheet.

5. **`data/raw_videos/use_of_unauthorized_object/`**:
   - Tilting the head deeply downward towards the lap or drawer.
   - Dipping both hands down under the desk table to interact with a concealed smartphone or cheat sheet.

> **Recommended Clip Length:**
> - 15 to 60 seconds per behavior segment is optimal.
> - At 30 fps, a 30-second clip generates ~540 sliding windows with `extract_dataset.py`.

---

### STEP 2: Extract Pose Sequences (`extract_dataset.py`)

Once your video files are saved into their class folders under `data/raw_videos/`, execute:

```powershell
python extract_dataset.py
```

#### What happens during extraction:
1. Scans all 5 class folders and loads `yolo26s-pose.pt`.
2. Tracks every person in each video using **ByteTrack**, ensuring temporal sequences belong to the same person.
3. Normalizes keypoints relative to torso midpoint and bounding box size (invariance to camera distance and seat location).
4. Slices continuous sliding windows of $T = 30$ frames (with `--stride 5` by default).
5. Compresses and saves the dataset to:
   ```
   data/processed/exam_dataset.npz
   ```

#### Generating Demonstration Data Immediately:
If you want to test the full pipeline before recording physical exam footage, generate synthetic demonstration sequences for all 5 classes with:
```powershell
python extract_dataset.py --generate-synthetic
```

---

### STEP 3: Train the 5-Class LSTM Model (70-15-15 Split) (`train.py`)

Run the training pipeline:

```powershell
python train.py --epochs 30 --batch-size 32
```

#### Stratified 70-15-15 Split:
- **70% Training Set**: Updates model weights via backpropagation.
- **15% Validation Set**: Evaluated after every epoch to adjust learning rate and trigger **Early Stopping** (saving `weights/best_lstm_model.pt`).
- **15% Held-Out Test Set**: Completely unseen during training. Evaluated after training completes to report honest real-world metrics.

#### Output Reports Generated:
- `reports/training_curves.png`: Training vs. Validation Loss and Accuracy plots.
- `reports/confusion_matrix.png`: 5x5 Confusion matrix heatmap across all classes.
- `reports/test_evaluation_report.txt`: Exact Precision, Recall, and F1-score for each of the 5 classes.

---

### STEP 4: Run Real-Time Proctoring Inference (`inference.py`)

#### Option A: Running on a Video File
```powershell
python inference.py --source path/to/exam_video.mp4
```

To record an annotated video with bounding boxes, skeletons, and alert badges:
```powershell
python inference.py --source path/to/exam_video.mp4 --save-video output_annotated.mp4
```

#### Option B: Running on Live Webcam
```powershell
python inference.py --source 0
```

#### Option C: Headless Mode (No GUI)
```powershell
python inference.py --source exam.mp4 --save-video proctored.mp4 --no-display
```

#### Visual Alerts & Color Palette:
| Behavior Class | Visual Color | Header Badge | Alert Banner |
|---|---|---|---|
| `normal` | **Green** | `ID: X \| NORMAL 98%` | *None (Normal Exam Behavior)* |
| `hand_signal` | **Amber Orange** | `ID: X \| HAND SIGNAL 89%` | `! ALERT: HAND SIGNAL !` |
| `passing_of_notes` | **Purple/Magenta** | `ID: X \| PASSING OF NOTES 93%`| `! ALERT: PASSING OF NOTES !` |
| `side_glancing` | **Deep Orange** | `ID: X \| SIDE GLANCING 86%` | `! ALERT: SIDE GLANCING !` |
| `use_of_unauthorized_object`| **Crimson Red** | `ID: X \| USE OF UNAUTHORIZED OBJECT 95%`| `! ALERT: USE OF UNAUTHORIZED OBJECT !` |

---

### STEP 5: Verification Suite (`test_prototype.py`)

Run the automated test suite to verify keypoint normalization, model forward pass, 70-15-15 stratified split, and end-to-end training for all 5 classes:

```powershell
python test_prototype.py
```

Expected Output:
```
==================================================
 Running Prototype Verification Suite (5 Classes)
==================================================
-> Testing keypoint normalization...
   [PASS] Keypoint normalization works as expected.
-> Testing ExamBehaviorLSTM forward pass with 5 classes...
   [PASS] Model forward pass and softmax probabilities verified (5 classes).
-> Testing 70-15-15 stratified dataset split across 5 classes...
   [PASS] 70-15-15 Stratified Split verified across 5 classes.
-> Testing synthetic extraction and training pipeline for 5 classes...
   [PASS] End-to-end 5-class training and test evaluation succeeded.

[ALL 5-CLASS TESTS PASSED SUCCESSFULLY!]
```

---

## 5. Hyperparameter Customization (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CLASSES` | 5 items | `["normal", "hand_signal", "passing_of_notes", "side_glancing", "use_of_unauthorized_object"]` |
| `NUM_CLASSES` | `5` | Total number of target behavior classes. |
| `SEQUENCE_LENGTH` | `30` | Number of consecutive frames per classification window (~1.0s at 30 fps). |
| `WINDOW_STRIDE` | `5` | Frame step when extracting sliding windows from raw videos. |
| `FEATURE_DIM` | `34` | 17 COCO keypoints $\times$ 2 normalized $(x, y)$ coordinates. |
| `LSTM_HIDDEN_DIM`| `64` | Hidden dimension of the LSTM network. |
| `LSTM_NUM_LAYERS`| `2` | Number of stacked recurrent layers. |
| `LSTM_BIDIRECTIONAL`| `True` | Bidirectional temporal context for richer gesture modeling. |
| `TRAIN_RATIO` | `0.70` | 70% Training partition. |
| `VAL_RATIO` | `0.15` | 15% Validation partition. |
| `TEST_RATIO` | `0.15` | 15% Held-out Test partition. |
| `SUSPICIOUS_CONFIDENCE_THRESHOLD` | `0.60` | Minimum confidence score required to trigger a proctor alert. |
| `ALERT_DEBOUNCE_FRAMES` | `5` | Frame debouncer preventing single-frame momentary alert flickering. |
