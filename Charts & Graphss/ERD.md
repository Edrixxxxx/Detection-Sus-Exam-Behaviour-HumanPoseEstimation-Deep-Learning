# Entity Relationship Diagram (ERD) Specifications & Schema Documentation

## Project Information
- **Project Title:** Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning
- **Institution:** DMMMSU - South La Union Campus | BS Computer Science (AY 2025-2026)
- **Deployment Platform:** Desktop Application (PyQt6 with GPU-accelerated PyTorch/CUDA backend)
- **Inference Specification:** 3 FPS inference rate (333 ms sample window), 60-frame state buffer (20-second temporal memory), sub-50ms processing latency per frame pipeline.

---

# Descriptive Definition of Entity Relationship Diagram (ERD)

An **Entity Relationship Diagram (ERD)** is a structural database model that visually represents the logical data architecture, entity abstractions, attribute definitions, primary/foreign key constraints, and relational cardinalities governing a software system.

In the context of the **Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning** project, the ERD defines the persistent and transactional data infrastructure required to support real-time automated proctoring and post-examination compliance auditing. The system collects multi-modal data streams—ranging from dual 1080p video feeds and 17 COCO human keypoint coordinates to 28-element spatial-temporal feature vectors, ArcFace 512-dimensional facial recognition embeddings, hybrid classification scores, and graduated proctoring alerts.

### Grounded Project Attributes & Scope
Based on the project specifications documented in `project_progress.md`, the ERD models:
1. **Examinee & Identity Roster Data:** Stores student profile metadata, seat assignments, and baseline ArcFace (`antelopev2`) facial feature vectors used for automated identity verification and real-time seat-swap detection.
2. **Dual-Camera & Spatial Calibration Metrics:** Represents hardware configurations for Camera 1 (Middle Top View) and Camera 2 (Top Side View), storing camera spatial positions, streaming URIs, and $3 \times 3$ homography transformation matrices for perspective alignment.
3. **Spatial-Temporal Pose Tracking Data:** Captures per-frame person bounding boxes $(x, y, w, h)$, persistent ByteTrack movement IDs, 17 anatomical keypoints $(x_i, y_i, c_i)$, and historical 60-frame (20-second) sliding state buffers.
4. **28-Element Feature Vectors:** Persists normalized feature extractions divided into four groups: static posture angles (Features 1–16), temporal dynamics and angular velocities (Features 17–22), contextual spatial examinee interactions (Features 23–26), and confidence/body-proportion normalization scale parameters (Features 27–28).
5. **Hybrid Classification & Calibration Logits:** Stores uncalibrated raw model/heuristic scores alongside Platt-scaled calibrated probability values ($P_{\text{calibrated}} \in [0.0, 1.0]$) for six target cheating behaviors (*side glancing, head down, standing, passing notes, hand signaling, unauthorized object use*).
6. **Graduated Proctoring Alerts & Evidence:** Records temporal 3-second consensus evaluations and graduated alert dispatches across three severity tiers (**Yellow**, **Orange**, **Red**), linking alert records to timestamped bounding-box evidence screenshots and raw video archive segments.

---

# Visual Entity Relationship Diagram

![Entity Relationship Diagram](images/ERD.png)

# Detailed Entity Specifications & Data Dictionary

### 1. `INSTITUTION_DEPARTMENT`
Stores top-level organizational metadata for academic departments conducting examination monitoring.
- **`dept_id`** (INT, PK, AUTO_INCREMENT): Unique primary key identifier.
- **`dept_name`** (VARCHAR(100), NOT NULL): Department name (e.g., "Department of Computer Science").
- **`campus_location`** (VARCHAR(100), NOT NULL): Campus name (e.g., "DMMMSU - South La Union Campus").

### 2. `PROCTOR_USER`
Stores credentials and profile details of proctors/instructors operating the PyQt6 monitoring application.
- **`proctor_id`** (INT, PK, AUTO_INCREMENT): Unique proctor identifier.
- **`full_name`** (VARCHAR(100), NOT NULL): Instructor full name.
- **`email`** (VARCHAR(100), UNIQUE, NOT NULL): Official email address.
- **`role`** (VARCHAR(50), DEFAULT 'Proctor'): User access authorization tier.

### 3. `EXAM_SESSION`
Represents an individual examination monitoring session event.
- **`session_id`** (INT, PK, AUTO_INCREMENT): Primary key session ID.
- **`dept_id`** (INT, FK -> `INSTITUTION_DEPARTMENT.dept_id`): Hosting department reference.
- **`proctor_id`** (INT, FK -> `PROCTOR_USER.proctor_id`): Supervising instructor reference.
- **`session_code`** (VARCHAR(50), UNIQUE, NOT NULL): Examination code (e.g., "CS312-EXAM-01").
- **`start_time`** (DATETIME, NOT NULL): Scheduled session start timestamp.
- **`end_time`** (DATETIME, NULL): Session completion timestamp.
- **`room_number`** (VARCHAR(20), NOT NULL): Physical examination room identifier.
- **`total_examinees`** (INT, DEFAULT 0): Count of registered examinees.

### 4. `STUDENT_EXAMINEE`
Stores student profile records and biometrics for automated identification and seat verification.
- **`student_id`** (INT, PK, AUTO_INCREMENT): Unique student primary key.
- **`student_number`** (VARCHAR(30), UNIQUE, NOT NULL): University ID number.
- **`first_name`** (VARCHAR(50), NOT NULL): Student given name.
- **`last_name`** (VARCHAR(50), NOT NULL): Student surname.
- **`face_embedding_512d`** (BLOB, NOT NULL): Binary representation of the 512-dimensional ArcFace (`antelopev2`) facial feature vector.

### 5. `SEAT_ASSIGNMENT`
Maps examinees to specific grid locations within the examination room seating layout.
- **`seat_id`** (INT, PK, AUTO_INCREMENT): Unique seating record identifier.
- **`session_id`** (INT, FK -> `EXAM_SESSION.session_id`): Reference to exam session.
- **`student_id`** (INT, FK -> `STUDENT_EXAMINEE.student_id`): Assigned student reference.
- **`row_index`** (INT, NOT NULL): Seating row index (0-indexed).
- **`col_index`** (INT, NOT NULL): Seating column index (0-indexed).
- **`grid_x`** (FLOAT, NOT NULL): Normalized 2D spatial workspace x-coordinate $[0.0, 1.0]$.
- **`grid_y`** (FLOAT, NOT NULL): Normalized 2D spatial workspace y-coordinate $[0.0, 1.0]$.

### 6. `CAMERA_DEVICE`
Captures configurations and spatial matrices for dual surveillance cameras.
- **`camera_id`** (INT, PK, AUTO_INCREMENT): Unique camera record identifier.
- **`session_id`** (INT, FK -> `EXAM_SESSION.session_id`): Active session reference.
- **`device_name`** (VARCHAR(50), NOT NULL): Camera descriptor ("Camera 1" or "Camera 2").
- **`stream_url`** (VARCHAR(255), NOT NULL): RTSP or USB hardware device connection string.
- **`view_angle`** (VARCHAR(50), NOT NULL): Perspective specification ("Middle Top View" or "Top Side View").
- **`homography_matrix_3x3`** (BLOB, NOT NULL): Serialized $3 \times 3$ floating-point OpenCV homography matrix for dual-camera perspective alignment.

### 7. `FRAME_CAPTURE_LOG`
Stores frame-level metadata generated by Process 1.0 (Video Capture & Pre-processing).
- **`frame_id`** (INT, PK, AUTO_INCREMENT): Unique frame log identifier.
- **`session_id`** (INT, FK -> `EXAM_SESSION.session_id`): Associated session.
- **`camera_id`** (INT, FK -> `CAMERA_DEVICE.camera_id`): Source camera device.
- **`frame_index`** (INT, NOT NULL): Monotonically increasing frame sequence counter.
- **`timestamp_sec`** (FLOAT, NOT NULL): Relative session elapsed time in seconds.
- **`phash_value`** (VARCHAR(16), NOT NULL): 64-bit perceptual hash string for duplicate frame filtering.
- **`blur_laplacian_var`** (FLOAT, NOT NULL): Computed Laplacian variance for image blur detection.

### 8. `POSE_TRACK_INSTANCE`
Stores person detections and persistent tracking instances output by Process 2.0 (Pose Estimation & Tracking).
- **`track_id`** (INT, PK, AUTO_INCREMENT): Unique track instance primary key.
- **`frame_id`** (INT, FK -> `FRAME_CAPTURE_LOG.frame_id`): Source frame reference.
- **`student_id`** (INT, FK -> `STUDENT_EXAMINEE.student_id`): Matched student identity (via ArcFace Re-ID).
- **`byte_track_id`** (INT, NOT NULL): Persistent tracker integer ID assigned by ByteTrack.
- **`bbox_x`** (FLOAT, NOT NULL): Bounding box top-left x-coordinate.
- **`bbox_y`** (FLOAT, NOT NULL): Bounding box top-left y-coordinate.
- **`bbox_w`** (FLOAT, NOT NULL): Bounding box width.
- **`bbox_h`** (FLOAT, NOT NULL): Bounding box height.
- **`tracking_confidence`** (FLOAT, NOT NULL): Object detection confidence score.

### 9. `KEYPOINT_OBSERVATION`
Stores 17 2D anatomical keypoints for each detected person instance.
- **`keypoint_obs_id`** (INT, PK, AUTO_INCREMENT): Primary key identifier.
- **`track_id`** (INT, FK -> `POSE_TRACK_INSTANCE.track_id`): Parent pose track instance.
- **`keypoint_index`** (INT, NOT NULL): COCO anatomical index (0–16: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
- **`pos_x`** (FLOAT, NOT NULL): Keypoint x-coordinate in frame space.
- **`pos_y`** (FLOAT, NOT NULL): Keypoint y-coordinate in frame space.
- **`confidence_score`** (FLOAT, NOT NULL): Keypoint detection confidence $c_i \in [0.0, 1.0]$.

### 10. `FEATURE_VECTOR_28`
Persists the 28 body-normalized features calculated by Process 3.0 (Feature Extraction).
- **`feature_id`** (INT, PK, AUTO_INCREMENT): Unique feature vector identifier.
- **`track_id`** (INT, FK -> `POSE_TRACK_INSTANCE.track_id`): Associated pose track.
- **`yaw_angle`** (FLOAT, NOT NULL): Feature 1: Head yaw orientation angle (degrees).
- **`pitch_angle`** (FLOAT, NOT NULL): Feature 2: Head pitch orientation angle (degrees).
- **`roll_angle`** (FLOAT, NOT NULL): Feature 3: Head roll orientation angle (degrees).
- **`hand_wrist_dist_avg`** (FLOAT, NOT NULL): Feature 8: Inter-wrist to shoulder average Euclidean distance.
- **`torso_lean_angle`** (FLOAT, NOT NULL): Feature 11: Spine inclination angle relative to vertical.
- **`yaw_velocity`** (FLOAT, NOT NULL): Feature 17: Temporal rate of change of head yaw ($\Delta \text{Yaw}/\Delta t$).
- **`hand_speed_px_sec`** (FLOAT, NOT NULL): Feature 19: Wrist spatial velocity over sliding window.
- **`neighbor_wrist_dist`** (FLOAT, NOT NULL): Feature 23: Spatial distance to adjacent examinee's workspace.
- **`body_norm_factor`** (FLOAT, NOT NULL): Feature 28: Inter-shoulder normalization factor used to scale distance features.

### 11. `CLASSIFICATION_OUTPUT`
Stores raw and Platt-calibrated behavior probability scores from Process 4.0 (Behaviour Classification).
- **`classification_id`** (INT, PK, AUTO_INCREMENT): Classification result identifier.
- **`feature_id`** (INT, FK -> `FEATURE_VECTOR_28.feature_id`): Parent feature vector reference.
- **`behavior_class`** (VARCHAR(50), NOT NULL): Target behavior class (*Side Glancing, Head Down, Standing, Passing Notes, Hand Signal, Unauthorized Object*).
- **`raw_heuristic_score`** (FLOAT, NULL): Output score from deterministic head/object heuristics.
- **`raw_xgboost_score`** (FLOAT, NULL): Output probability from XGBoost decision tree classifier.
- **`platt_calibrated_prob`** (FLOAT, NOT NULL): Final calibrated probability $P_{\text{calibrated}} \in [0.0, 1.0]$.
- **`is_suspicious`** (BOOLEAN, NOT NULL): Flag indicating whether score exceeds base decision threshold ($> 0.50$).

### 12. `ALERT_INCIDENT`
Records temporal consensus evaluations and graduated proctoring alerts dispatched by Process 5.0.
- **`alert_id`** (INT, PK, AUTO_INCREMENT): Unique alert record identifier.
- **`classification_id`** (INT, FK -> `CLASSIFICATION_OUTPUT.classification_id`): Triggering classification result.
- **`session_id`** (INT, FK -> `EXAM_SESSION.session_id`): Exam session reference.
- **`student_id`** (INT, FK -> `STUDENT_EXAMINEE.student_id`): Flagged examinee ID.
- **`alert_level`** (VARCHAR(10), NOT NULL): Severity tier (**Yellow**, **Orange**, or **Red**).
- **`buffer_flag_ratio`** (FLOAT, NOT NULL): Ratio of flagged frames within 3-second sliding window buffer.
- **`timestamp`** (DATETIME, NOT NULL): System dispatch timestamp.
- **`proctor_acknowledged`** (BOOLEAN, DEFAULT FALSE): Status of proctor UI acknowledgement.

### 13. `EVIDENCE_SNAPSHOT`
Stores file system paths and overlay coordinates for evidence artifacts generated during Red tier alerts.
- **`snapshot_id`** (INT, PK, AUTO_INCREMENT): Unique snapshot identifier.
- **`alert_id`** (INT, FK -> `ALERT_INCIDENT.alert_id`): Associated alert incident.
- **`image_file_path`** (VARCHAR(255), NOT NULL): Disk storage path to JPEG evidence snapshot with bounding box overlays.
- **`video_clip_path`** (VARCHAR(255), NULL): Disk path to 10-second MP4 video buffer clip.
- **`bbox_overlay_x`** (INT, NOT NULL): Overlay bounding box x-coordinate.
- **`bbox_overlay_y`** (INT, NOT NULL): Overlay bounding box y-coordinate.

### 14. `SYSTEM_CONFIGURATION`
Stores system runtime parameters, thresholds, and quality filtering limits.
- **`config_id`** (INT, PK, AUTO_INCREMENT): Configuration entry identifier.
- **`session_id`** (INT, FK -> `EXAM_SESSION.session_id`): Session reference.
- **`yellow_alert_threshold`** (FLOAT, DEFAULT 0.50): Base confidence limit for Yellow tier alerts.
- **`orange_alert_threshold`** (FLOAT, DEFAULT 0.30): Buffer percentage threshold for Orange tier alerts.
- **`red_alert_threshold`** (FLOAT, DEFAULT 0.60): Buffer percentage threshold for Red tier alerts.
- **`blur_variance_limit`** (FLOAT, DEFAULT 50.0): Minimum Laplacian variance for blur filtering.
- **`phash_hamming_limit`** (INT, DEFAULT 5): Maximum Hamming distance for duplicate frame suppression.

### 15. `MODEL_WEIGHTS_METADATA`
Tracks machine learning model versions and calibration coefficients loaded in memory.
- **`model_id`** (INT, PK, AUTO_INCREMENT): Unique model metadata primary key.
- **`session_id`** (INT, FK -> `EXAM_SESSION.session_id`): Active session reference.
- **`yolo_model_version`** (VARCHAR(50), DEFAULT 'YOLOv26s-pose-FP16'): Pose model version string.
- **`xgboost_model_version`** (VARCHAR(50), DEFAULT 'XGBoost-v1.4-hand'): Hand gesture classifier version.
- **`platt_param_a`** (FLOAT, NOT NULL): Sigmoid slope parameter $A$.
- **`platt_param_b`** (FLOAT, NOT NULL): Sigmoid intercept parameter $B$.

---

# Relationship & Cardinality Summary Matrix

| Primary Entity | Foreign Entity | Relationship | Cardinality | FK Field in Child Entity | Description / Integrity Constraints |
|---|---|---|---|---|---|
| `INSTITUTION_DEPARTMENT` | `EXAM_SESSION` | Hosts | 1 : N | `EXAM_SESSION.dept_id` | One department hosts multiple exam sessions. |
| `PROCTOR_USER` | `EXAM_SESSION` | Supervises | 1 : N | `EXAM_SESSION.proctor_id` | One instructor supervises multiple exam sessions. |
| `EXAM_SESSION` | `SEAT_ASSIGNMENT` | Defines | 1 : N | `SEAT_ASSIGNMENT.session_id` | Session defines seating layout grid. CASCADE DELETE. |
| `STUDENT_EXAMINEE` | `SEAT_ASSIGNMENT` | Assigned To | 1 : N | `SEAT_ASSIGNMENT.student_id` | Student is assigned to specific session seat. |
| `EXAM_SESSION` | `CAMERA_DEVICE` | Utilizes | 1 : N | `CAMERA_DEVICE.session_id` | Session utilizes dual cameras (Cam 1 & Cam 2). |
| `EXAM_SESSION` | `FRAME_CAPTURE_LOG` | Captures | 1 : N | `FRAME_CAPTURE_LOG.session_id` | Session generates continuous frame logs. |
| `CAMERA_DEVICE` | `FRAME_CAPTURE_LOG` | Records | 1 : N | `FRAME_CAPTURE_LOG.camera_id` | Camera streams individual video frame logs. |
| `FRAME_CAPTURE_LOG` | `POSE_TRACK_INSTANCE` | Detects | 1 : N | `POSE_TRACK_INSTANCE.frame_id` | Frame contains multiple examinee pose detections. |
| `STUDENT_EXAMINEE` | `POSE_TRACK_INSTANCE` | Identified As | 1 : N | `POSE_TRACK_INSTANCE.student_id` | Re-ID links tracking instance to student biometrics. |
| `POSE_TRACK_INSTANCE` | `KEYPOINT_OBSERVATION` | Contains | 1 : 17 | `KEYPOINT_OBSERVATION.track_id` | Track instance contains 17 COCO keypoints. |
| `POSE_TRACK_INSTANCE` | `FEATURE_VECTOR_28` | Generates | 1 : 1 | `FEATURE_VECTOR_28.track_id` | Pose history generates one 28-feature vector. |
| `FEATURE_VECTOR_28` | `CLASSIFICATION_OUTPUT` | Evaluates | 1 : N | `CLASSIFICATION_OUTPUT.feature_id` | Feature vector evaluated across behavior classes. |
| `CLASSIFICATION_OUTPUT` | `ALERT_INCIDENT` | Triggers | 1 : N | `ALERT_INCIDENT.classification_id` | Suspicious scores trigger graduated alerts. |
| `ALERT_INCIDENT` | `EVIDENCE_SNAPSHOT` | Attaches | 1 : 1 | `EVIDENCE_SNAPSHOT.alert_id` | Red alerts attach visual evidence snapshot. |
| `EXAM_SESSION` | `SYSTEM_CONFIGURATION` | Configures | 1 : 1 | `SYSTEM_CONFIGURATION.session_id` | Session loads runtime threshold parameters. |
| `EXAM_SESSION` | `MODEL_WEIGHTS_METADATA` | Executes | 1 : N | `MODEL_WEIGHTS_METADATA.session_id` | Session executes neural model weights. |

---

# Data Storage & Volume Estimation

| Data Store / Table | Storage Format / Engine | Indexing Strategy | Estimated Volume (1-Hour Session, 30 Examinees @ 3 FPS) | Retention & Archival Policy |
|---|---|---|---|---|
| **`STUDENT_EXAMINEE`** | SQLite / PostgreSQL | `student_number` (UNIQUE) | ~30 rows (~250 KB with ArcFace embeddings) | Permanent student roster master table |
| **`FRAME_CAPTURE_LOG`** | SQLite / PostgreSQL | `(session_id, frame_index)` | ~21,600 rows per camera (~4.2 MB) | Purged 30 days post-exam audit |
| **`POSE_TRACK_INSTANCE`** | SQLite / PostgreSQL | `(frame_id, byte_track_id)` | ~324,000 rows (~45 MB) | Retained for temporal buffer analysis |
| **`KEYPOINT_OBSERVATION`** | SQLite / Binary Blob | `track_id` | ~5.5 million rows (~220 MB raw or binary) | Compressed to binary keypoint archive |
| **`FEATURE_VECTOR_28`** | SQLite / PostgreSQL | `track_id` | ~324,000 rows (~65 MB) | Retained for classifier re-training |
| **`CLASSIFICATION_OUTPUT`**| SQLite / PostgreSQL | `(feature_id, behavior_class)` | ~324,000 rows (~38 MB) | Logged for statistical audit report |
| **`ALERT_INCIDENT`** | SQLite / PostgreSQL | `(session_id, alert_level)` | ~50–200 incident rows (< 1 MB) | Permanent compliance record |
| **`EVIDENCE_SNAPSHOT`** | JPEG Files on Disk | File system path key | ~20–50 JPEG images (~25 MB) | Permanent compliance evidence archive |
| **`D8 Video Archive`** | MP4 Video (H.264 FP16) | Time-coded index | 2 x 1080p @ 30fps MP4 streams (~3.2 GB) | Archived for offline proctor review |

---

*Last updated: Academic Year 2025–2026*
