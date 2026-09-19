# Structured English Specifications

## Project Information
- **Project Title:** Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning
- **Institution:** DMMMSU - South La Union Campus | BS Computer Science (AY 2025-2026)
- **Deployment Platform:** Desktop Application (PyQt6 with GPU-accelerated PyTorch/CUDA backend)
- **Inference Specification:** 3 FPS inference rate (333 ms sample window), 60-frame state buffer (20-second temporal memory), sub-50ms processing latency per frame pipeline.

---

# Overview of Structured English

**Structured English** is a narrative specification method that describes procedural logic, business rules, and algorithmic decision structures using restricted, plain-English statements combined with structured programming constructs (`IF-THEN-ELSE`, `FOR EACH`, `DO-WHILE`, `CASE/SWITCH`).

Unlike formal pseudocode—which resembles code syntax—Structured English uses clear, human-readable action verbs and standardized logic blocks to document operational workflows so that software developers, systems analysts, and academic reviewers can unambiguously audit system behavior.

---

# Subsystem Logic in Structured English

## Process 1.0 — Dual-Camera Video Capture, Quality Control & Perspective Warping

```text
FOR EACH active camera stream (Camera 1: Middle Top View, Camera 2: Top Side View):
    
    DO THE FOLLOWING CONTINUOUSLY at 30 frames per second:
        
        CAPTURE raw 1080p RGB video frame.
        
        // Quality Control Filter 1: Blur Check
        CALCULATE image blur metric using Laplacian variance.
        IF Laplacian variance is LESS THAN 50.0 THEN
            REJECT raw frame as blurry.
            DISCARD frame and SKIP to next iteration.
        END IF
        
        // Quality Control Filter 2: Duplicate Frame Check
        COMPUTE 64-bit perceptual hash (pHash) for the current frame.
        COMPARE current pHash against the previous frame's pHash.
        IF the Hamming distance is LESS THAN 5 THEN
            REJECT frame as redundant near-duplicate.
            DISCARD frame and SKIP to next iteration.
        END IF
        
        // Normalization & Perspective Alignment
        RESIZE frame to 640 x 640 pixels using letterbox border padding (RGB 128, 128, 128).
        SCALE pixel intensity values to normalized floating-point range [0.0, 1.0].
        APPLY 3x3 Homography transformation matrix to warp Camera 2 perspective onto Camera 1 plane.
        
        SEND normalized clean frame tensor to the Frame Buffer Queue at 3 FPS sampling rate.
        
    END DO

END FOR EACH
```

---

## Process 2.0 — Pose Estimation, Multi-Object Tracking & Facial Identity Verification

```text
FOR EACH incoming clean frame tensor received from the Frame Buffer Queue (every 333 milliseconds):

    EXECUTE TensorRT half-precision (FP16) YOLOv26s-pose model inference on GPU.
    EXTRACT bounding box coordinates [x, y, w, h] and 17 COCO anatomical keypoints [x_i, y_i, c_i] for all detected individuals.

    FOR EACH detected person in the current frame:

        // Multi-Object Motion Association
        ASSOCIATE person detection with existing tracks using ByteTrack Kalman filtering and IoU matching.
        ASSIGN or MAINTAIN persistent Track ID.

        // Periodic Biometric Facial Re-Identification (Every 10 seconds)
        IF 10 seconds have elapsed since last identity check THEN
            CROP facial region from frame using head keypoint coordinates.
            EXTRACT 512-dimensional facial embedding vector using ArcFace (antelopev2).
            COMPARE face embedding against the assigned student roster embeddings.
            
            IF facial similarity score is LESS THAN 0.60 THEN
                FLAG potential unauthorized seat-swap anomaly.
                LOG seat-swap warning with Student ID and Timestamp.
            END IF
        END IF

        // Historical Pose Queue Update
        APPEND current 17 keypoint coordinates to person's 60-frame sliding state history buffer.
        IF buffer length EXCEEDS 60 frames (20 seconds) THEN
            REMOVE oldest frame keypoints from buffer.
        END IF

    END FOR EACH

END FOR EACH
```

---

## Process 3.0 — 28-Feature Vector Extraction & Inter-Shoulder Scale Normalization

```text
FOR EACH active examinee track ID in the current inference frame:

    RETRIEVE keypoint history from the 60-frame sliding state buffer.

    // 1. Compute Static Posture Geometry (Features 1 to 16)
    CALCULATE Head Yaw, Pitch, and Roll angles from eye-nose-ear spatial vectors.
    CALCULATE lateral head displacement and head-to-shoulder vertical distance ratio.
    CALCULATE Left and Right wrist-to-shoulder Euclidean distances and average hand extension.
    CALCULATE Left and Right wrist Y-coordinates relative to desk height threshold.
    CALCULATE torso lateral lean angle and vertical spine compression ratio.
    CALCULATE shoulder-to-elbow and elbow-to-wrist segment lengths for Left and Right arms.

    // 2. Compute Temporal Movement Dynamics (Features 17 to 22)
    CALCULATE Head Yaw velocity (change in Yaw over time) and acceleration.
    CALCULATE wrist movement velocity (pixels per second) over sliding window.
    CALCULATE head-turn persistence duration (continuous seconds off-center).
    CALCULATE directional head turn oscillation frequency.
    CALCULATE 2D spatial position variance of hands over 60 frames.

    // 3. Compute Contextual & Spatial Interactions (Features 23 to 26)
    CALCULATE distance between examinee's wrists and adjacent examinee's workspace.
    EVALUATE mutual head-turn alignment indicator between adjacent examinees.
    RETRIEVE examinee grid coordinates within classroom seating layout.
    COUNT number of active surrounding examinees.

    // 4. Compute Confidence Metrics & Perform Scale Normalization (Features 27 to 28)
    CALCULATE mean keypoint confidence score across all 17 keypoints.
    RECORD overall person detection confidence.
    MEASURE inter-shoulder Euclidean distance (distance between left and right shoulders).

    FOR EACH spatial distance feature (Features 4, 6–10, 13–16, 23):
        DIVIDE feature value by inter-shoulder distance to achieve body-proportion scale invariance.
    END FOR EACH

    ASSEMBLE standardized 28-element feature vector and COMMIT to Feature Store.

END FOR EACH
```

---

## Process 4.0 — Hybrid Behavior Classification & Platt Scaling Calibration

```text
FOR EACH 28-element feature vector in the Feature Store:

    // 1. Evaluate Head Behavior Heuristics
    IF absolute Head Yaw angle EXCEEDS 30 degrees AND persistence EXCEEDS 1.5 seconds THEN
        ASSIGN raw score for "Side Glancing" behavior.
    ELSE IF Head Pitch angle is LESS THAN -25 degrees AND persistence EXCEEDS 3.0 seconds THEN
        ASSIGN raw score for "Head Down" behavior.
    ELSE IF vertical body displacement EXCEEDS 30 percent of body height THEN
        ASSIGN raw score for "Standing Up" behavior.
    END IF

    // 2. Evaluate Hand Behavior ML Model
    PASS hand position, limb distance, and temporal variance features into trained XGBoost decision trees.
    OBTAIN raw model output probabilities for "Passing Notes" and "Hand Signaling".

    // 3. Evaluate Workspace Context Heuristic
    IF wrist Y-position is BELOW desk threshold AND distance to neighbor workspace is LESS THAN 0.2 THEN
        ASSIGN raw score for "Unauthorized Object Use".
    END IF

    // 4. Perform Platt Scaling Sigmoid Calibration
    FOR EACH behavior class (Side Glancing, Head Down, Standing, Passing Notes, Hand Signal, Unauthorized Object):
        RETRIEVE pre-fitted Platt scaling parameters A and B for the behavior class.
        APPLY Sigmoid transformation equation:
            Calibrated Probability = 1 / (1 + EXP(A * Raw Score + B))
        STORE calibrated probability P_calibrated within range [0.0, 1.0].
    END FOR EACH

END FOR EACH
```

---

## Process 5.0 — 3-Second Temporal Consensus & Graduated Alert Dispatch

```text
FOR EACH classified examinee track:

    PUSH current frame calibrated probabilities into 3-second (9-sample) sliding buffer queue.
    COMPUTE ratio of flagged frames (frames with probability > 0.50) within 9-sample window.
    COMPUTE average calibrated confidence score across window.

    EVALUATE Graduated Severity Tier:

        IF buffer flag ratio EXCEEDS 60% AND average confidence EXCEEDS 0.70 THEN
            // Tier 3: RED ALERT (Critical Violation)
            SET Alert Level to "RED".
            LOG incident to Session Database.
            TRIGGER proctor interface visual screen flash (Red overlay).
            PLAY audible alarm chime.
            CAPTURE bounding-box evidence screenshot JPEG.
            ARCHIVE 10-second video buffer clip MP4.
            DISPATCH desktop push notification to proctor.

        ELSE IF buffer flag ratio is BETWEEN 30% AND 60% OR behavior SUSTAINED for 3+ seconds THEN
            // Tier 2: ORANGE ALERT (Medium Severity Warning)
            SET Alert Level to "ORANGE".
            LOG incident to Session Database.
            DISPLAY sidebar pop-up toast notification on proctor interface.

        ELSE IF buffer flag ratio is GREATER THAN 0% AND average confidence EXCEEDS 0.50 THEN
            // Tier 1: YELLOW ALERT (Low Severity Minor Anomaly)
            SET Alert Level to "YELLOW".
            LOG incident quietly to Session Database for audit trail.

        ELSE
            // No Alert
            SET Alert Level to "NONE".
        END IF

END FOR EACH
```

---

## Process 6.0 — Real-Time Proctor Dashboard & Post-Exam Audit Report Export

```text
// Real-Time Proctor UI Rendering Loop (PyQt6 Main Thread)
WHILE examination session is ACTIVE:
    RENDER dual-camera live video feeds on proctor application dashboard.
    DRAW color-coded bounding boxes over examinees (Green = Normal, Yellow/Orange/Red = Alert).
    DISPLAY real-time incident event log sidebar.
    
    IF proctor CLICKS on alert toast THEN
        DISPLAY student details, seat number, behavior type, and confidence score.
    END IF
END WHILE

// Post-Examination Compliance Audit Export
WHEN proctor ENDS examination session:
    GENERATE summary statistics of total alerts, behavior breakdowns, and student timelines.
    COMPILE evidence snapshots and video clips into post-exam audit report.
    EXPORT report in PDF and HTML formats for institutional review.
```

---


