# Data & Schema Specifications

This directory defines the physical and logical schemas for examination behavior datasets, pose representations, spatial-temporal sequence tensors, and database entities.

---

## 1. Directory Structure

```
4_Data_and_Schema/
├── raw_videos/                       # Raw video clips organized into class folders
│   ├── normal/                       # Standard seated exam behaviors
│   ├── hand_signal/                  # Hand waving, raised gestures, signaling
│   ├── passing_of_notes/             # Reaching/stretching across desks
│   ├── side_glancing/                # Looking sideways at peer's desk
│   └── use_of_unauthorized_object/   # Looking downwards at phone/notes in lap
├── processed/
│   ├── exam_dataset.npz              # Extracted sequences ready for LSTM training
│   └── test_demo_dataset.npz         # Synthetic verification dataset
└── schemas/
    ├── data_schema.md                # Schema definitions (this file)
    └── dataset.md                    # Dataset notes and roster reference
```

---

## 2. Examination Behavior Classes (5 Target Classes)

| Class Index | Class Name | Description | Key Body Language Indicators |
|---|---|---|---|
| `0` | `normal` | Legitimate exam writing and reading | Steady head, upright seated posture, arms within desk boundary. |
| `1` | `hand_signal` | Non-verbal signaling between students | Elevated wrist coordinates, hand waving, arm lifting. |
| `2` | `passing_of_notes` | Physical exchange of unauthorized material | Lateral arm extension across desk partitions, lateral shoulder tilt. |
| `3` | `side_glancing` | Peeking at neighbor's exam paper | Head yaw angle exceeding threshold, asymmetrical ear-to-shoulder displacement. |
| `4` | `use_of_unauthorized_object` | Looking at concealed smartphone or notes | Deep head pitch downward, sustained downward gaze into lap/drawer. |

---

## 3. Keypoint Normalization & Input Tensor Schema

### COCO 17-Keypoint Mapping
1. **0: Nose**
2. **1: Left Eye**, **2: Right Eye**
3. **3: Left Ear**, **4: Right Ear**
4. **5: Left Shoulder**, **6: Right Shoulder**
5. **7: Left Elbow**, **8: Right Elbow**
6. **9: Left Wrist**, **10: Right Wrist**
7. **11: Left Hip**, **12: Right Hip**
8. **13: Left Knee**, **14: Right Knee**
9. **15: Left Ankle**, **16: Right Ankle**

### Spatial Normalization Logic
Given 17 raw keypoints $[x_i, y_i]$ for $i \in \{0, \dots, 16\}$:
1. **Center Calculation:** Midpoint of hips $(\text{Left Hip} + \text{Right Hip})/2$; if unavailable, midpoint of shoulders; otherwise bounding box center.
2. **Scale Factor:** Bounding box diagonal $\sqrt{w^2 + h^2}$ or estimated torso length.
3. **Normalized Coordinates:**
   $$\tilde{x}_i = \frac{x_i - \text{center}_x}{\text{scale}}, \quad \tilde{y}_i = \frac{y_i - \text{center}_y}{\text{scale}}$$
4. **Per-Frame Feature Dimension:** $17 \times 2 = 34$ continuous float values.

### Temporal Sequence Dimensions
- **Sequence Length ($T$):** 30 consecutive frames ($\approx 1.0$ second at 30 FPS).
- **Window Stride:** 5 frames during training sequence extraction.
- **Dataset Tensor Shape (`exam_dataset.npz`):**
  - **`X`:** Shape $(N, 30, 34)$ — `float32` array of temporal pose sequences.
  - **`y`:** Shape $(N,)$ — `int64` array of class indices $[0, 4]$.

---

## 4. Full Architecture 28-Feature Schema

For the comprehensive 6-stage pipeline (detailed in [`2_Documentation/charts_and_graphs/Data_Dictionary.md`](file:///c:/Users/USER/Desktop/SchoolWorks/tisis%20sir%20joe/Detection-Sus-Exam-Behaviour-HumanPoseEstimation-Deep-Learning/2_Documentation/charts_and_graphs/Data_Dictionary.md)):
- **Features 1–5:** Head Yaw, Pitch, Roll angles, lateral displacement, head-shoulder ratio.
- **Features 6–10:** Left/Right wrist-shoulder distances, extension, desk-relative height.
- **Features 11–12:** Torso lateral lean angle, vertical spine compression ratio.
- **Features 13–16:** Upper arm and forearm segment lengths.
- **Features 17–22:** Yaw velocity, acceleration, wrist speed, turn frequency, hand variance.
- **Features 23–26:** Neighbor wrist distance, mutual head turn flag, seat grid, neighbor count.
- **Features 27–28:** Keypoint detection confidence, person detection confidence.
