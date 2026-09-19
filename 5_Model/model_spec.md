# Machine Learning Model Specifications

This directory contains trained model checkpoints, candidate model weights, and architectural specifications for the automated examination behavior proctoring system.

---

## 1. Directory Structure

```
5_Model/
├── weights/
│   ├── best_lstm_model.pt        # Trained 5-class sequential LSTM PyTorch checkpoint
│   └── test_lstm_model.pt        # Verification checkpoint generated during testing
└── model_spec.md                 # Technical model specification (this file)
```

---

## 2. Recurrent Deep Learning Architecture (`ExamBehaviorLSTM`)

The temporal sequence classifier is implemented in [`1_Source_Code/model.py`](file:///c:/Users/USER/Desktop/SchoolWorks/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/1_Source_Code/model.py) as `ExamBehaviorLSTM(nn.Module)`.

```
Input Sequence: (batch_size, T=30, D=34)
                     │
                     ▼
       LayerNorm across Feature Dim (34)
                     │
                     ▼
 2-Layer Bidirectional LSTM (hidden_size=64, dropout=0.3)
   -> Output Shape: (batch_size, 30, 64 * 2 = 128)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
Final Time Step Output    Mean Temporal Pooling
     (Shape: 128)              (Shape: 128)
         └───────────┬───────────┘
                     │
         Concatenation -> (batch_size, 256)
                     │
                     ▼
  Classification Head (MLP):
  - Linear(256 -> 64)
  - LayerNorm(64)
  - ReLU()
  - Dropout(p=0.3)
  - Linear(64 -> 5)
                     │
                     ▼
 Output Logits: (batch_size, 5) -> Softmax Probabilities
```

### Architectural Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `input_dim` | 34 | 17 normalized $(x, y)$ keypoint coordinates |
| `sequence_length` | 30 | 30 frames ($\sim 1.0$ second window) |
| `hidden_dim` | 64 | LSTM cell hidden state dimensions |
| `num_layers` | 2 | Stacked LSTM layer depth |
| `bidirectional` | True | Processes sequence in both forward and reverse directions |
| `dropout` | 0.3 | Dropout probability between layers and in MLP head |
| `num_classes` | 5 | Target classes: normal, hand_signal, passing_of_notes, side_glancing, object |

---

## 3. Pose Estimation & Tracking Models

- **Pose Estimation Model:** YOLO Pose backend (`YOLOv26s-pose` or `YOLO11s-pose` via Ultralytics).
  - Detected Keypoints: 17 anatomical COCO keypoints per person with confidence scores.
  - Candidate weight paths resolved via [`1_Source_Code/config.py`](file:///c:/Users/USER/Desktop/SchoolWorks/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/1_Source_Code/config.py):
    - `5_Model/weights/yolo26s-pose.pt`
    - `5_Model/weights/yolo26s-pose.engine`
    - `../yolo25s-pose/yolo26s-pose.pt`
    - `yolo11s-pose.pt` (Ultralytics auto-download fallback)
- **Multi-Object Tracking:** ByteTrack tracker using `bytetrack.yaml`.
  - Maintains persistent track IDs per examinee across successive video frames.

---

## 4. Evaluation Metrics (Held-Out Test Set)

The current trained weights [`best_lstm_model.pt`](file:///c:/Users/USER/Desktop/SchoolWorks/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/5_Model/weights/best_lstm_model.pt) achieve:
- **Test Accuracy:** 100.00%
- **Macro Precision:** 100.00%
- **Macro Recall:** 100.00%
- **Macro F1-Score:** 100.00%
- Detailed per-class breakdowns and confusion matrices are documented in [`2_Documentation/reports/`](file:///c:/Users/USER/Desktop/SchoolWorks/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/2_Documentation/reports).
