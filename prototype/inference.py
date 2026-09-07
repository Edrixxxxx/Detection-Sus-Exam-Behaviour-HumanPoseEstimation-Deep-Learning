import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch

from config import (
    YOLO_WEIGHTS,
    LSTM_MODEL_PATH,
    CLASSES,
    IDX_TO_CLASS,
    SUSPICIOUS_CONFIDENCE_THRESHOLD,
    SEQUENCE_LENGTH,
    ALERT_DEBOUNCE_FRAMES,
)
from model import ExamBehaviorLSTM
from tracker import PoseTrackerManager


# COCO Pose Skeleton Connection Pairs for visual rendering
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Facial keypoints
    (5, 6),                                       # Shoulders
    (5, 7), (7, 9),                               # Left arm
    (6, 8), (8, 10),                              # Right arm
    (5, 11), (6, 12), (11, 12),                   # Torso
    (11, 13), (13, 15),                           # Left leg
    (12, 14), (14, 16),                           # Right leg
]


def load_classifier(weights_path: Path, device: torch.device) -> Optional[ExamBehaviorLSTM]:
    """Loads the trained LSTM classifier checkpoint."""
    if not weights_path.exists():
        print(f"[Warning] LSTM model checkpoint not found at: {weights_path}")
        print("Inference will run in pose tracking only mode. Train the LSTM first using 'python train.py'.")
        return None

    checkpoint = torch.load(weights_path, map_location=device)
    model = ExamBehaviorLSTM(
        input_dim=checkpoint.get("input_dim", 34),
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_layers=checkpoint.get("num_layers", 2),
        num_classes=checkpoint.get("num_classes", 2),
        bidirectional=checkpoint.get("bidirectional", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"-> Loaded LSTM classifier successfully from: {weights_path}")
    return model


# Color palette per behavior class (BGR)
CLASS_COLORS = {
    "normal": (0, 200, 50),                       # Green
    "hand_signal": (0, 165, 255),                  # Amber Orange
    "passing_of_notes": (210, 50, 210),            # Purple/Magenta
    "side_glancing": (0, 120, 255),                # Deep Orange
    "use_of_unauthorized_object": (0, 0, 245),     # Bright Crimson Red
}


def draw_proctoring_overlay(
    frame: np.ndarray,
    detection: dict,
    pred_label: str,
    confidence: float,
    is_warming_up: bool,
    warmup_count: int,
):
    """Draws bounding boxes, skeleton, status badges, and class-specific alert tags."""
    bbox = detection["bbox"].astype(int)
    kpts = detection["raw_keypoints"]
    tid = detection["track_id"]

    x1, y1, x2, y2 = bbox

    # Determine colors and text
    if is_warming_up:
        box_color = (180, 180, 180)  # Neutral Gray
        status_text = f"ID: {tid} | WARMING UP ({warmup_count}/{SEQUENCE_LENGTH})"
    else:
        box_color = CLASS_COLORS.get(pred_label, (0, 0, 245))
        label_display = pred_label.upper().replace("_", " ")
        status_text = f"ID: {tid} | {label_display} {confidence*100:.0f}%"

    # 1. Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 2. Status Badge Header
    (w, h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    badge_y1 = max(0, y1 - h - 10)
    badge_y2 = max(h + 10, y1)
    badge_x2 = min(frame.shape[1], x1 + w + 14)

    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
    cv2.putText(
        frame, status_text, (x1 + 6, badge_y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
    )

    # 3. Draw Skeleton Bones
    for p1, p2 in SKELETON_PAIRS:
        pt1 = kpts[p1]
        pt2 = kpts[p2]
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(
                frame,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                box_color, 2, cv2.LINE_AA
            )

    # 4. Draw Keypoint Joints
    for pt in kpts:
        if pt[0] > 0 and pt[1] > 0:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

    # 5. Glowing alert banner on suspicious behaviors
    if pred_label != "normal" and not is_warming_up:
        banner_text = f"! ALERT: {pred_label.upper().replace('_', ' ')} !"
        cv2.putText(
            frame, banner_text, (x1, min(frame.shape[0] - 10, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2, cv2.LINE_AA
        )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    active_examinees: int,
    suspicious_count: int,
):
    """Draws top HUD monitor bar."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & Statistics
    hud_left = f"AI Exam Proctor | YOLO26s-Pose + ByteTrack + LSTM"
    hud_right = f"FPS: {fps:.1f} | Active Students: {active_examinees} | Alerts: {suspicious_count}"

    cv2.putText(frame, hud_left, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    (w, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    right_x = max(frame.shape[1] - w - 16, 20)
    cv2.putText(frame, hud_right, (right_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


def run_inference(
    source: str = "0",
    yolo_weights: str = YOLO_WEIGHTS,
    lstm_weights: Path = LSTM_MODEL_PATH,
    save_video: Optional[str] = None,
    no_display: bool = False,
    threshold: float = SUSPICIOUS_CONFIDENCE_THRESHOLD,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Starting Live Examination Behavior Proctoring")
    print(f" - Source:       {source}")
    print(f" - YOLO Weights: {yolo_weights}")
    print(f" - LSTM Weights: {lstm_weights}")
    print(f" - Device:       {device}")
    print(f" - Threshold:    {threshold*100:.0f}%")
    print(f"=======================================================\n")

    # Parse camera device index vs video file
    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] Could not open video stream or camera: {source}")
        return

    # Video Writer if requested
    writer = None
    if save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, fps_in, (width, height))
        print(f"-> Recording annotated video to: {save_video}")

    # Initialize Modules
    tracker_manager = PoseTrackerManager(weights_path=yolo_weights, sequence_length=SEQUENCE_LENGTH)
    classifier = load_classifier(lstm_weights, device)

    # Frame timing
    prev_time = time.time()
    frame_count = 0
    total_suspicious_incidents = 0

    print("System active. Press 'q' or 'ESC' on the video window to quit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached.")
                break

            frame_count += 1
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # 1. Pose Tracking with ByteTrack
            detections = tracker_manager.process_frame(frame)

            active_examinees = len(detections)
            current_frame_suspicious = 0

            # 2. Sequential Behavior Classification
            for det in detections:
                person = det["tracked_person"]
                is_ready = det["sequence_ready"]
                seq = det["sequence"]

                if classifier is not None and is_ready and seq is not None:
                    # Format sequence tensor
                    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_idx, conf, probs = classifier.predict_single(seq_tensor)

                    # Check against suspicious threshold (class index 0 is normal, 1-4 are suspicious)
                    predicted_class = IDX_TO_CLASS[pred_idx]
                    if pred_idx != 0 and conf >= threshold:
                        person.alert_counter = min(ALERT_DEBOUNCE_FRAMES * 2, person.alert_counter + 2)
                        current_frame_suspicious += 1
                        person.current_prediction = predicted_class
                        person.current_confidence = conf
                    else:
                        person.alert_counter = max(0, person.alert_counter - 1)
                        if person.alert_counter == 0:
                            person.current_prediction = "normal"
                            person.current_confidence = conf if pred_idx == 0 else (1.0 - conf)

                    draw_proctoring_overlay(
                        frame, det,
                        pred_label=person.current_prediction,
                        confidence=person.current_confidence,
                        is_warming_up=False,
                        warmup_count=SEQUENCE_LENGTH,
                    )
                else:
                    # Buffer warming up
                    warmup_len = len(person.buffer)
                    draw_proctoring_overlay(
                        frame, det,
                        pred_label="normal",
                        confidence=0.0,
                        is_warming_up=True,
                        warmup_count=warmup_len,
                    )

            if current_frame_suspicious > 0:
                total_suspicious_incidents += 1

            # 3. Draw Top HUD
            draw_hud(frame, fps, active_examinees, total_suspicious_incidents)

            # 4. Save frame if writing video
            if writer is not None:
                writer.write(frame)

            # 5. Display Window
            if not no_display:
                cv2.imshow("Exam Behavior Proctoring System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[Done] Video saved successfully to: {save_video}")
        if not no_display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Exam Behavior Classification using YOLO26s-Pose + ByteTrack + LSTM.")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for live webcam or path to raw video file (.mp4)")
    parser.add_argument("--yolo-weights", type=str, default=YOLO_WEIGHTS, help="Path to YOLO pose weights file")
    parser.add_argument("--lstm-weights", type=str, default=str(LSTM_MODEL_PATH), help="Path to trained LSTM classifier checkpoint")
    parser.add_argument("--save-video", type=str, default=None, help="Optional output path to record annotated video")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without opening GUI window")
    parser.add_argument("--threshold", type=float, default=SUSPICIOUS_CONFIDENCE_THRESHOLD, help="Suspicious confidence threshold (0.0 - 1.0)")

    args = parser.parse_args()

    run_inference(
        source=args.source,
        yolo_weights=args.yolo_weights,
        lstm_weights=Path(args.lstm_weights),
        save_video=args.save_video,
        no_display=args.no_display,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
