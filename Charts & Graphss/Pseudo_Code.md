# System Pseudocode Specifications

## Project Information
- **Project Title:** Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning
- **Institution:** DMMMSU - South La Union Campus | BS Computer Science (AY 2025-2026)
- **Deployment Platform:** Desktop Application (PyQt6 with GPU-accelerated PyTorch/CUDA backend)
- **Inference Specification:** 3 FPS inference rate (333 ms sample window), 60-frame state buffer (20-second temporal memory), sub-50ms processing latency per frame pipeline.

---

# Overview

This document presents the **Pseudocode Specifications** for the automated suspicious examination behavior detection system. The pseudocode provides an algorithmic, step-by-step description of the execution pipeline, detailing multi-threaded video capture, FP16 YOLOv26s-pose keypoint estimation, ByteTrack motion tracking, ArcFace biometrics, 28-feature extraction, hybrid behavior classification, Platt scaling calibration, 3-second temporal buffer consensus aggregation, and graduated alert dispatches (**Yellow**, **Orange**, **Red**).

---

# Top-Level System Execution Controller

```text
ALGORITHM System_Main_Loop()
BEGIN
    // Step 1: Initialize System Infrastructure
    LOAD system_configuration FROM "config.yaml"
    LOAD yolo_pose_model (FP16 TensorRT Engine)
    LOAD xgboost_hand_classifier ("hand_model.bin")
    LOAD platt_calibration_parameters (A, B)
    LOAD camera_calibration_homography_matrix H
    LOAD student_roster_embeddings (ArcFace 512d vectors)

    INITIALIZE thread_safe_queue frame_queue
    INITIALIZE thread_safe_queue pose_queue
    INITIALIZE thread_safe_queue alert_queue
    INITIALIZE sliding_state_buffer (60 frames / 20 seconds)

    // Step 2: Spawn Concurrent Processing Threads
    START_THREAD Video_Capture_Thread(frame_queue)
    START_THREAD Inference_Pipeline_Thread(frame_queue, pose_queue, alert_queue)
    START_THREAD Proctor_UI_Thread(alert_queue)

    WAIT FOR USER_EXIT_SIGNAL
    TERMINATE ALL THREADS
END ALGORITHM
```

---

# Module Pseudocode Specifications

## Module 1: Video Capture & Pre-Processing

```text
ALGORITHM Video_Capture_Thread(frame_queue)
BEGIN
    CONNECT Camera_1 (Middle Top View, 1080p @ 30fps)
    CONNECT Camera_2 (Top Side View, 1080p @ 30fps)

    WHILE system_is_running DO
        READ raw_frame_cam1 FROM Camera_1
        READ raw_frame_cam2 FROM Camera_2

        // Quality Check 1: Blur Detection via Laplacian Variance
        laplacian_var = CALCULATE_LAPLACIAN_VARIANCE(raw_frame_cam1)
        IF laplacian_var < 50.0 THEN
            DROP raw_frame_cam1 // Reject blurry frame
            CONTINUE
        END IF

        // Quality Check 2: Duplicate Frame Suppression via pHash
        current_phash = COMPUTE_PERCEPTUAL_HASH(raw_frame_cam1)
        IF HAMMING_DISTANCE(current_phash, previous_phash) < 5 THEN
            DROP raw_frame_cam1 // Reject redundant frame
            CONTINUE
        END IF
        previous_phash = current_phash

        // Normalization & Homography Perspective Alignment
        letterboxed_tensor = RESIZE_AND_LETTERBOX(raw_frame_cam1, target_size=(640, 640))
        aligned_tensor = APPLY_HOMOGRAPHY(letterboxed_tensor, H)
        scaled_tensor = NORMALIZE_PIXELS(aligned_tensor, range=[0.0, 1.0])

        ENQUEUE scaled_tensor INTO frame_queue (every 333ms / 3 FPS)
    END WHILE
END ALGORITHM
```

---

## Module 2: Pose Estimation, Tracking & Identity Verification

```text
ALGORITHM Estimate_Pose_And_Track(frame_tensor)
BEGIN
    // Step 1: FP16 YOLOv26s-pose Inference
    detections = RUN_YOLOV26S_POSE_FP16(frame_tensor)
    // Output: Array of {bbox [x, y, w, h], keypoints_17 [x_i, y_i, c_i]}

    FOR EACH person IN detections DO
        // Step 2: ByteTrack ID Assignment
        track_id = ASSIGN_BYTETRACK_ID(person.bbox, person.keypoints_17)
        person.track_id = track_id

        // Step 3: ArcFace Identity Verification (Executes every 10 seconds)
        IF current_time % 10.0 == 0 THEN
            face_crop = CROP_FACE(frame_tensor, person.bbox, person.keypoints_17)
            face_embedding = EXTRACT_ARCFACE_512D(face_crop)
            matched_student = MATCH_ROSTER(face_embedding, student_roster_embeddings)

            IF matched_student.assigned_seat != person.current_seat THEN
                EMIT_WARNING("SEAT SWAP DETECTED for Student ID: " + matched_student.id)
            END IF
        END IF

        // Step 4: Update 60-Frame State History Buffer
        UPDATE_SLIDING_BUFFER(track_id, person.keypoints_17, max_capacity=60)
    END FOR

    RETURN updated_tracks
END ALGORITHM
```

---

## Module 3: 28-Feature Extraction & Normalization

```text
ALGORITHM Extract_28_Features(person_track, sliding_buffer_60)
BEGIN
    INITIALIZE feature_vector = ARRAY OF SIZE 28

    // Group 1: Static Posture Geometry (Features 1 to 16)
    feature_vector[1..3]  = COMPUTE_HEAD_YAW_PITCH_ROLL(person_track.keypoints)
    feature_vector[4]     = COMPUTE_LATERAL_HEAD_DISPLACEMENT(person_track.keypoints)
    feature_vector[5]     = COMPUTE_HEAD_SHOULDER_VERTICAL_RATIO(person_track.keypoints)
    feature_vector[6..8]  = COMPUTE_WRIST_SHOULDER_DISTANCES(person_track.keypoints)
    feature_vector[9..10] = COMPUTE_WRIST_DESK_Y_POSITION(person_track.keypoints)
    feature_vector[11..12]= COMPUTE_TORSO_LEAN_AND_HEIGHT_RATIO(person_track.keypoints)
    feature_vector[13..16]= COMPUTE_LIMB_SEGMENT_DISTANCES(person_track.keypoints)

    // Group 2: Temporal Movement Dynamics (Features 17 to 22)
    feature_vector[17..18]= COMPUTE_YAW_VELOCITY_AND_ACCELERATION(sliding_buffer_60)
    feature_vector[19]    = COMPUTE_WRIST_SPEED_PX_SEC(sliding_buffer_60)
    feature_vector[20]    = COMPUTE_HEAD_PERSISTENCE_DURATION(sliding_buffer_60)
    feature_vector[21]    = COMPUTE_DIRECTIONAL_TURN_FREQUENCY(sliding_buffer_60)
    feature_vector[22]    = COMPUTE_HAND_POSITION_VARIANCE(sliding_buffer_60)

    // Group 3: Contextual Spatial Interactions (Features 23 to 26)
    feature_vector[23]    = COMPUTE_NEIGHBOR_WRIST_DISTANCE(person_track, all_active_tracks)
    feature_vector[24]    = COMPUTE_MUTUAL_HEAD_TURN_INDICATOR(person_track, adjacent_tracks)
    feature_vector[25]    = GET_SEAT_GRID_COORDINATES(person_track)
    feature_vector[26]    = COUNT_SURROUNDING_NEIGHBORS(person_track)

    // Group 4: Confidence & Scale Normalization (Features 27 to 28)
    feature_vector[27]    = CALCULATE_MEAN_KEYPOINT_CONFIDENCE(person_track.keypoints)
    feature_vector[28]    = person_track.detection_confidence

    // Inter-Shoulder Body Normalization
    shoulder_width = EUCLIDEAN_DISTANCE(left_shoulder, right_shoulder)
    FOR EACH spatial_index IN [4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 23] DO
        feature_vector[spatial_index] = feature_vector[spatial_index] / shoulder_width
    END FOR

    RETURN feature_vector
END ALGORITHM
```

---

## Module 4: Hybrid Behaviour Classification & Platt Calibration

```text
ALGORITHM Classify_And_Calibrate_Behaviour(feature_vector)
BEGIN
    INITIALIZE raw_scores = MAP()

    // 1. Head Behaviour Heuristics
    yaw_angle = feature_vector[1]
    pitch_angle = feature_vector[2]
    persistence = feature_vector[20]

    IF ABS(yaw_angle) > 30.0 AND persistence > 1.5 THEN
        raw_scores["Side_Glancing"] = 0.85
    ELSE IF pitch_angle < -25.0 AND persistence > 3.0 THEN
        raw_scores["Head_Down"] = 0.80
    ELSE IF feature_vector[12] > 0.3 THEN
        raw_scores["Standing"] = 0.90
    END IF

    // 2. Hand Behaviour XGBoost Model Inference
    hand_prob_notes = XGBOOST_PREDICT(hand_model, feature_vector[6..10, 13..16, 19, 22])
    raw_scores["Passing_Notes"] = hand_prob_notes

    // 3. Workspace Context Object Heuristic
    IF feature_vector[9] > desk_threshold AND feature_vector[23] < 0.2 THEN
        raw_scores["Unauthorized_Object"] = 0.75
    END IF

    // 4. Platt Scaling Sigmoid Calibration
    calibrated_probabilities = MAP()
    FOR EACH (behavior_class, raw_f) IN raw_scores DO
        A = platt_params[behavior_class].A
        B = platt_params[behavior_class].B
        
        // Sigmoid Transformation Formula
        P_calibrated = 1.0 / (1.0 + EXP(A * raw_f + B))
        calibrated_probabilities[behavior_class] = P_calibrated
    END FOR

    RETURN calibrated_probabilities
END ALGORITHM
```

---

## Module 5: Temporal Buffer Aggregation & Graduated Alert Dispatch

```text
ALGORITHM Aggregate_And_Dispatch_Alerts(track_id, calibrated_probabilities)
BEGIN
    // Step 1: 3-Second Sliding Buffer Consensus (9 Samples at 3 FPS)
    PUSH_TO_TEMPORAL_BUFFER(track_id, calibrated_probabilities, window_size=9)
    
    flagged_count = COUNT_FLAGGED_FRAMES_IN_WINDOW(track_id, threshold=0.50)
    buffer_ratio = flagged_count / 9.0
    avg_confidence = COMPUTE_AVERAGE_CONFIDENCE_IN_WINDOW(track_id)

    // Step 2: Graduated Alert Threshold Rules
    alert_level = "NONE"

    IF buffer_ratio > 0.60 AND avg_confidence > 0.70 THEN
        alert_level = "RED"      // High Severity Critical Violation
    ELSE IF buffer_ratio >= 0.30 OR SUSTAINED_DURATION(track_id) >= 3.0 THEN
        alert_level = "ORANGE"   // Medium Severity Warning
    ELSE IF buffer_ratio > 0.0 AND avg_confidence > 0.50 THEN
        alert_level = "YELLOW"   // Low Severity Minor Anomaly
    END IF

    // Step 3: Dispatch Actions
    SWITCH alert_level DO
        CASE "YELLOW":
            LOG_TO_DATABASE(track_id, "YELLOW", avg_confidence)
            BREAK

        CASE "ORANGE":
            LOG_TO_DATABASE(track_id, "ORANGE", avg_confidence)
            EMIT_UI_SIDEBAR_TOAST(track_id, "Suspicious movement detected")
            BREAK

        CASE "RED":
            LOG_TO_DATABASE(track_id, "RED", avg_confidence)
            TRIGGER_UI_SCREEN_FLASH(color="RED")
            PLAY_AUDIO_CHIME(file="alarm.wav")
            
            // Capture Evidence Artifacts
            snapshot_path = SAVE_EVIDENCE_SNAPSHOT(track_id, frame_tensor, bbox)
            SAVE_VIDEO_CLIP(track_id, buffer_10s, "red_alert_clip.mp4")
            
            EMIT_PROCTOR_PUSH_NOTIFICATION("CRITICAL CHEATING ALERT: Seat " + track_id)
            BREAK
    END SWITCH
END ALGORITHM
```

---

