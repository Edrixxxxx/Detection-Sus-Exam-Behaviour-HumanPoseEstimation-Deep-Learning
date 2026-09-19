# System Data Dictionary Specifications

## Project Information
- **Project Title:** Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning
- **Institution:** DMMMSU - South La Union Campus | BS Computer Science (AY 2025-2026)
- **Deployment Platform:** Desktop Application (PyQt6 with GPU-accelerated PyTorch/CUDA backend)
- **Inference Specification:** 3 FPS inference rate (333 ms sample window), 60-frame state buffer (20-second temporal memory), sub-50ms processing latency per frame pipeline.

---

# Overview of Data Dictionary

A **Data Dictionary** is a centralized reference repository that defines the metadata, data structures, data elements, data types, valid value ranges, constraints, and relationships for all data stores and data flows within a software system.

In the **Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning** project, the Data Dictionary provides an unambiguous, standardized definition of all inputs, internal state representations, features, model logits, and output artifacts processed across the 6-stage video analytics pipeline.

---

# Data Stores Master Table

| Store ID | Data Store Name | Primary Physical Storage | Description & Content | Primary Key / Access Index |
|---|---|---|---|---|
| **D1a** | Raw Frame Buffer | RAM (Ring Buffer) | Unprocessed dual-camera 1080p @ 30 FPS RGB image matrices. | `(camera_id, frame_index)` |
| **D1b** | Clean Frame Buffer | RAM / GPU VRAM | Resized ($640 \times 640$), letterboxed, normalized, perspective-aligned image tensors. | `frame_id` |
| **D2a** | Detections Store | RAM | Per-frame bounding box coordinates $[x, y, w, h]$ and 17 COCO 2D keypoints $[x_i, y_i, c_i]$. | `detection_id` |
| **D2b** | Track Store | RAM / SQLite | Persistent ByteTrack movement IDs, ArcFace student identities, and 60-frame sliding state history queues. | `track_id` |
| **D3** | Feature Store | RAM / SQLite | Standardized, body-normalized 28-element feature vectors per examinee per frame. | `feature_id` |
| **D4** | Classification Store | RAM / SQLite | Raw classifier logits and Platt-calibrated probability scores ($P_{\text{calibrated}} \in [0.0, 1.0]$). | `classification_id` |
| **D5** | Alert Log | SQLite Database | Recorded incident events, timestamps, alert levels (**Yellow**, **Orange**, **Red**), and evidence file paths. | `alert_id` |
| **D6** | Model Weights Store | Disk / GPU VRAM | Pre-trained model weights (YOLOv26s-pose FP16, XGBoost classifier) and Platt scaling parameters $(A, B)$. | `model_id` |
| **D7** | Configuration Store | YAML File / RAM | Homography matrices, blur variance limits, pHash limits, body norm parameters, and alert thresholds. | `config_id` |
| **D8** | Video Archive | Disk (MP4 Files) | Recorded dual 1080p video streams stored for post-exam compliance review. | `video_file_id` |

---

# Data Elements Specifications

## 1. Examinee & Biometric Data Elements

| Field Name | Data Type | Length / Format | Valid Range / Values | Nullable | Description |
|---|---|---|---|---|---|
| `student_id` | Integer | Primary Key | $1$ to $999,999$ | NO | Unique internal identifier for registered examinees. |
| `student_number` | String | VARCHAR(30) | Unique Alphanumeric | NO | Institutional student identification number. |
| `first_name` | String | VARCHAR(50) | Alphabetic text | NO | Student given name. |
| `last_name` | String | VARCHAR(50) | Alphabetic text | NO | Student surname. |
| `face_embedding_512d` | Binary / Blob | 512 Float32 Array | $[-1.0, 1.0]$ per dim | NO | ArcFace (`antelopev2`) 512-dimensional facial recognition feature vector. |
| `assigned_seat` | String | VARCHAR(10) | e.g., "R1-C3" | NO | Registered seat grid location (Row and Column). |

---

## 2. Video Capture & Frame Pre-Processing Data Elements

| Field Name | Data Type | Length / Format | Valid Range / Values | Nullable | Description |
|---|---|---|---|---|---|
| `frame_id` | Big Integer | Auto Increment | $1$ to $2^{63}-1$ | NO | Monotonically increasing unique frame identifier. |
| `camera_id` | Integer | SmallInt | $1$ (Cam 1) or $2$ (Cam 2) | NO | Camera device descriptor ($1 = \text{Top Middle}$, $2 = \text{Top Side}$). |
| `timestamp_sec` | Float | Float64 (IEEE 754) | $\ge 0.000$ seconds | NO | Relative elapsed session timestamp in seconds. |
| `raw_frame_matrix` | Image Array | $1080 \times 1920 \times 3$ | RGB Pixels $[0, 255]$ | NO | Raw uncompressed frame matrix from OpenCV capture loop. |
| `clean_tensor` | Tensor Array | $640 \times 640 \times 3$ | BGR Float $[0.0, 1.0]$ | NO | Letterboxed, normalized, perspective-aligned image tensor. |
| `phash_value` | String | 64-bit Hex String | 16 Hex Characters | NO | Perceptual hash value for duplicate frame detection. |
| `blur_laplacian_var` | Float | Float32 | $\ge 0.0$ | NO | Computed 2D Laplacian variance metric for blur filtering. |

---

## 3. Keypoint Detection & Tracking Data Elements

| Field Name | Data Type | Length / Format | Valid Range / Values | Nullable | Description |
|---|---|---|---|---|---|
| `track_id` | Integer | Integer | $1$ to $999,999$ | NO | Persistent tracking ID assigned by ByteTrack algorithm. |
| `bbox_x` | Float | Float32 | $[0.0, 640.0]$ | NO | Top-left x-coordinate of examinee bounding box. |
| `bbox_y` | Float | Float32 | $[0.0, 640.0]$ | NO | Top-left y-coordinate of examinee bounding box. |
| `bbox_w` | Float | Float32 | $> 0.0$ | NO | Width of examinee bounding box. |
| `bbox_h` | Float | Float32 | $> 0.0$ | NO | Height of examinee bounding box. |
| `keypoint_index` | Integer | SmallInt | $0$ to $16$ | NO | COCO anatomical keypoint index (e.g., $0=\text{nose}$, $1=\text{L\_eye}$, $5=\text{L\_shoulder}$). |
| `pos_x` | Float | Float32 | $[0.0, 640.0]$ | NO | 2D x-coordinate of keypoint in frame space. |
| `pos_y` | Float | Float32 | $[0.0, 640.0]$ | NO | 2D y-coordinate of keypoint in frame space. |
| `confidence_score` | Float | Float32 | $[0.0, 1.0]$ | NO | Detection confidence score for keypoint $c_i$. |

---

## 4. 28-Element Feature Vector Dictionary

The feature vector consists of 28 unpadded elements normalized by inter-shoulder width $d_{\text{shoulder}}$:

| Feature # | Feature Name | Category | Data Type | Valid Range | Unit / Scale | Description |
|---|---|---|---|---|---|---|
| **F1** | `yaw_angle` | Head Pose | Float32 | $[-180^\circ, +180^\circ]$ | Degrees | Head yaw rotation angle (horizontal turn). |
| **F2** | `pitch_angle` | Head Pose | Float32 | $[-90^\circ, +90^\circ]$ | Degrees | Head pitch inclination angle (nod/tilt). |
| **F3** | `roll_angle` | Head Pose | Float32 | $[-90^\circ, +90^\circ]$ | Degrees | Head roll tilt angle. |
| **F4** | `lateral_head_disp` | Head Pose | Float32 | $[0.0, 3.0]$ | Scaled | Lateral head displacement from spine axis. |
| **F5** | `head_shoulder_ratio` | Head Pose | Float32 | $[0.0, 2.0]$ | Ratio | Vertical distance ratio between head and shoulders. |
| **F6** | `left_wrist_dist` | Hand Position | Float32 | $[0.0, 4.0]$ | Scaled | Left wrist-to-shoulder Euclidean distance. |
| **F7** | `right_wrist_dist` | Hand Position | Float32 | $[0.0, 4.0]$ | Scaled | Right wrist-to-shoulder Euclidean distance. |
| **F8** | `hand_wrist_dist_avg` | Hand Position | Float32 | $[0.0, 4.0]$ | Scaled | Average extension distance of both wrists. |
| **F9** | `left_wrist_desk_y` | Hand Position | Float32 | $[0.0, 2.0]$ | Scaled | Left wrist Y-coordinate relative to desk height. |
| **F10** | `right_wrist_desk_y` | Hand Position | Float32 | $[0.0, 2.0]$ | Scaled | Right wrist Y-coordinate relative to desk height. |
| **F11** | `torso_lean_angle` | Torso | Float32 | $[-90^\circ, +90^\circ]$ | Degrees | Torso lateral inclination angle relative to vertical. |
| **F12** | `torso_height_ratio` | Torso | Float32 | $[0.0, 2.0]$ | Ratio | Spine compression/extension height ratio. |
| **F13** | `left_upper_arm_dist` | Limb Distance | Float32 | $[0.0, 3.0]$ | Scaled | Left shoulder-to-elbow segment length. |
| **F14** | `right_upper_arm_dist`| Limb Distance | Float32 | $[0.0, 3.0]$ | Scaled | Right shoulder-to-elbow segment length. |
| **F15** | `left_forearm_dist` | Limb Distance | Float32 | $[0.0, 3.0]$ | Scaled | Left elbow-to-wrist segment length. |
| **F16** | `right_forearm_dist` | Limb Distance | Float32 | $[0.0, 3.0]$ | Scaled | Right elbow-to-wrist segment length. |
| **F17** | `yaw_velocity` | Temporal | Float32 | $[-360.0, +360.0]$ | deg/sec | Rate of change of head yaw angle ($\Delta \text{Yaw}/\Delta t$). |
| **F18** | `yaw_acceleration` | Temporal | Float32 | $[-720.0, +720.0]$ | $\text{deg/sec}^2$ | Angular acceleration of head yaw ($\Delta^2 \text{Yaw}/\Delta t^2$). |
| **F19** | `hand_speed_px_sec` | Temporal | Float32 | $[0.0, 1000.0]$ | px/sec | Spatial movement velocity of wrists. |
| **F20** | `head_persistence_sec`| Temporal | Float32 | $[0.0, 20.0]$ | Seconds | Continuous duration off-center head posture is held. |
| **F21** | `direction_change_freq`| Temporal | Float32 | $[0.0, 10.0]$ | Hz / Count | Frequency of head turn direction reversals over window. |
| **F22** | `hand_position_var` | Temporal | Float32 | $[0.0, 500.0]$ | $\text{px}^2$ | 2D spatial variance of hand position over 60 frames. |
| **F23** | `neighbor_wrist_dist` | Contextual | Float32 | $[0.0, 5.0]$ | Scaled | Distance between examinee wrists and adjacent desk. |
| **F24** | `mutual_head_turn_flag`| Contextual | Binary | $0$ or $1$ | Indicator | $1 = \text{Reciprocal head turn with neighbor detected}$. |
| **F25** | `seat_position_grid` | Contextual | String / Float | Grid Coord | Grid Pos | Examinee seating position in classroom layout. |
| **F26** | `neighbor_count` | Contextual | Integer | $[0, 8]$ | Count | Number of active surrounding examinees. |
| **F27** | `mean_keypoint_conf` | Confidence | Float32 | $[0.0, 1.0]$ | Score | Average keypoint detection confidence $c_{\text{mean}}$. |
| **F28** | `detection_confidence` | Confidence | Float32 | $[0.0, 1.0]$ | Score | Person bounding box object detection confidence. |

---

## 5. Classification & Platt Calibration Data Elements

| Field Name | Data Type | Valid Range / Values | Description |
|---|---|---|---|
| `behavior_class` | String | `Side_Glancing`, `Head_Down`, `Standing`, `Passing_Notes`, `Hand_Signal`, `Unauthorized_Object` | Target suspicious behavior category label. |
| `raw_heuristic_score` | Float | $[0.0, 1.0]$ | Uncalibrated score output from deterministic head/object rules. |
| `raw_xgboost_score` | Float | $[0.0, 1.0]$ | Uncalibrated probability output from XGBoost classifier. |
| `platt_param_a` | Float | Real number | Sigmoid slope parameter $A$ used in Platt scaling equation. |
| `platt_param_b` | Float | Real number | Sigmoid intercept parameter $B$ used in Platt scaling equation. |
| `platt_calibrated_prob`| Float | $[0.0, 1.0]$ | Final calibrated posterior probability $P(y=1\|f) = \frac{1}{1 + \exp(A \cdot f + B)}$. |
| `is_suspicious` | Boolean | `TRUE` or `FALSE` | `TRUE` if `platt_calibrated_prob` $> 0.50$. |

---

## 6. Alert Incident & Evidence Data Elements

| Field Name | Data Type | Valid Range / Values | Description |
|---|---|---|---|
| `alert_id` | Integer | $1$ to $999,999$ | Unique primary key identifier for logged alert incidents. |
| `alert_level` | String | `YELLOW`, `ORANGE`, `RED` | Assigned graduated alert severity tier. |
| `buffer_flag_ratio` | Float | $[0.0, 1.0]$ | Ratio of flagged frames within 3-second (9-sample) sliding window buffer. |
| `avg_window_confidence`| Float | $[0.0, 1.0]$ | Mean calibrated confidence score across 3-second consensus window. |
| `image_file_path` | String | File System Path | Path to saved JPEG evidence snapshot with bounding box overlays. |
| `video_clip_path` | String | File System Path | Path to saved 10-second MP4 video clip of flagged incident. |
| `proctor_acknowledged` | Boolean | `TRUE` or `FALSE` | Status indicating whether proctor acknowledged alert on PyQt6 UI. |

---

# Data Flows Mapping Table

| Flow ID | Data Flow Name | Source Entity / Process | Destination Process / Store | Data Structure Passed |
|---|---|---|---|---|
| **DF1** | Live Dual Video Feeds | Cameras 1 & 2 | 1.1 Dual-Camera Frame Capture | Dual 1080p @ 30 FPS RGB video streams |
| **DF2** | Raw Video Stream | 1.1 Frame Capture | D1a Raw Frame Buffer | Uncompressed RGB image matrices |
| **DF3** | Filtered Unique Frames | 1.3 Duplicate Removal | 1.4 Normalization | Quality-passed, non-duplicate RGB frames |
| **DF4** | Clean Frame Tensors | 1.5 Calibration | D1b Clean Frame Buffer | Letterboxed $640 \times 640$ normalized float tensors |
| **DF5** | Detections & Keypoints | 2.1 YOLO Inference | D2a Detections Store | Bbox $[x,y,w,h]$ + 17 keypoint pairs $[x_i, y_i, c_i]$ |
| **DF6** | Persistent Pose Tracks | 2.2 ByteTrack | D2b Track Store | Track ID + spatial keypoint coordinates |
| **DF7** | 60-Frame Pose History | D2b Track Store | 3.1–3.3 Feature Extractors | 20-second sliding queue of keypoint trajectories |
| **DF8** | 28-Feature Vectors | 3.4 Scale Normalizer | D3 Feature Store | Standardized body-normalized float32[28] array |
| **DF9** | Raw Classifier Logits | 4.1–4.3 Classifiers | 4.4 Platt Calibration | Uncalibrated heuristic and XGBoost model outputs |
| **DF10**| Calibrated Probabilities | 4.4 Platt Calibration | D4 Classification Store | Probabilities $P_{\text{calibrated}} \in [0.0, 1.0]$ per class |
| **DF11**| 3s Consensus Scores | 5.1 Temporal Consensus | 5.2 Alert Level Determination| 9-sample sliding window consensus confidence |
| **DF12**| Graduated UI Alert | 5.3 Alert Dispatch | Proctor Dashboard (PyQt6) | Severity payload (**Yellow**, **Orange**, **Red**) |
| **DF13**| Incident Evidence | 5.3 Alert Dispatch | D5 Alert Log | Event metadata + JPEG snapshot file path |

---

*Last updated: Academic Year 2025–2026*
