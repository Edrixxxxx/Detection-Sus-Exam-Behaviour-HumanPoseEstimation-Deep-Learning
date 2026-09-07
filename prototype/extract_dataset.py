import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np

from config import (
    RAW_VIDEOS_DIR,
    RAW_CLASS_DIRS,
    DATASET_FILE,
    SEQUENCE_LENGTH,
    WINDOW_STRIDE,
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    YOLO_WEIGHTS,
    FEATURE_DIM,
)
from tracker import PoseTrackerManager
from dataset import save_processed_dataset


SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def extract_sequences_from_video(
    video_path: Path,
    label: str,
    tracker_manager: PoseTrackerManager,
    stride: int = WINDOW_STRIDE,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extracts sliding window pose sequences from a raw video file.

    Args:
        video_path: Path to raw video file.
        label: Class label (e.g. 'normal', 'hand_signal', etc.).
        tracker_manager: PoseTrackerManager instance.
        stride: Frame stride between consecutive extracted windows.

    Returns:
        Tuple of (list of sequences (T, 34), list of integer labels).
    """
    class_idx = CLASS_TO_IDX[label]
    tracker_manager.reset()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Warning] Could not open video file: {video_path}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"-> Processing '{video_path.name}' [Class: {label}] (Frames: {total_frames}, FPS: {fps:.1f})")

    sequences: List[np.ndarray] = []
    labels: List[int] = []

    last_extracted: dict = {}
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        detections = tracker_manager.process_frame(frame)

        for det in detections:
            tid = det["track_id"]
            if det["sequence_ready"]:
                last_frame = last_extracted.get(tid, -stride)
                if (frame_counter - last_frame) >= stride:
                    seq = det["sequence"]
                    if seq is not None and seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                        sequences.append(seq)
                        labels.append(class_idx)
                        last_extracted[tid] = frame_counter

    cap.release()
    print(f"   Extracted {len(sequences)} windows from '{video_path.name}'.")
    return sequences, labels


def generate_synthetic_demo_dataset(num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates realistic synthetic examination pose sequences across all 5 classes
    for immediate pipeline testing, verification, and baseline initialization.
    """
    print(f"-> Generating {num_samples} synthetic demo sequences for {NUM_CLASSES} classes...")
    rng = np.random.RandomState(42)

    X_list = []
    y_list = []
    samples_per_class = num_samples // NUM_CLASSES

    # Baseline seated student keypoints (COCO 17 keypoints normalized)
    base_pose = np.array([
        0.0, -0.45,   # 0: Nose
        -0.03, -0.48, # 1: L Eye
        0.03, -0.48,  # 2: R Eye
        -0.08, -0.45, # 3: L Ear
        0.08, -0.45,  # 4: R Ear
        -0.25, -0.25, # 5: L Shoulder
        0.25, -0.25,  # 6: R Shoulder
        -0.30, 0.05,  # 7: L Elbow
        0.30, 0.05,   # 8: R Elbow
        -0.20, 0.25,  # 9: L Wrist
        0.20, 0.25,   # 10: R Wrist
        -0.15, 0.35,  # 11: L Hip
        0.15, 0.35,   # 12: R Hip
        -0.18, 0.70,  # 13: L Knee
        0.18, 0.70,   # 14: R Knee
        -0.18, 1.0,   # 15: L Ankle
        0.18, 1.0,    # 16: R Ankle
    ], dtype=np.float32)

    for class_idx in range(NUM_CLASSES):
        cls_name = IDX_TO_CLASS[class_idx]
        for _ in range(samples_per_class):
            seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

            for t in range(SEQUENCE_LENGTH):
                t_ratio = t / float(SEQUENCE_LENGTH)
                pose_t = base_pose.copy()

                if cls_name == "normal":
                    # Class 0: Natural subtle writing motion
                    noise = rng.normal(0, 0.005, size=FEATURE_DIM).astype(np.float32)
                    pose_t[19] += 0.01 * np.sin(t_ratio * 4.0 * np.pi)  # Minor wrist oscillation
                    pose_t += noise

                elif cls_name == "hand_signal":
                    # Class 1: Hand raises / signals neighbor
                    wave = np.sin(t_ratio * 2.0 * np.pi)
                    pose_t[20] -= 0.35 * max(0.0, wave)  # R wrist lifted upwards
                    pose_t[16] -= 0.20 * max(0.0, wave)  # R elbow raised
                    noise = rng.normal(0, 0.006, size=FEATURE_DIM).astype(np.float32)
                    pose_t += noise

                elif cls_name == "passing_of_notes":
                    # Class 2: Arm stretches sideways across desk
                    reach = np.sin(t_ratio * np.pi)
                    pose_t[20] += 0.40 * reach  # R wrist extends far right
                    pose_t[16] += 0.25 * reach  # R elbow flares out
                    # Slight shoulder dip
                    pose_t[12] += 0.08 * reach
                    noise = rng.normal(0, 0.007, size=FEATURE_DIM).astype(np.float32)
                    pose_t += noise

                elif cls_name == "side_glancing":
                    # Class 3: Head turns sideways toward neighbor
                    glance = np.sin(t_ratio * np.pi)
                    # Nose (0,1), Eyes (2,3; 4,5), Ears (6,7; 8,9) translate horizontally
                    pose_t[0] += 0.22 * glance
                    pose_t[2] += 0.22 * glance
                    pose_t[4] += 0.22 * glance
                    pose_t[6] += 0.22 * glance
                    pose_t[8] += 0.22 * glance
                    noise = rng.normal(0, 0.006, size=FEATURE_DIM).astype(np.float32)
                    pose_t += noise

                elif cls_name == "use_of_unauthorized_object":
                    # Class 4: Head tilts deep down + both hands dipped to lap/phone
                    dip = np.sin(t_ratio * np.pi)
                    # Head pitches down
                    pose_t[1] += 0.25 * dip
                    pose_t[3] += 0.25 * dip
                    pose_t[5] += 0.25 * dip
                    # Both wrists lowered into lap
                    pose_t[19] += 0.30 * dip
                    pose_t[21] += 0.30 * dip
                    noise = rng.normal(0, 0.007, size=FEATURE_DIM).astype(np.float32)
                    pose_t += noise

                seq[t] = pose_t

            X_list.append(seq)
            y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle samples
    indices = rng.permutation(len(y))
    return X[indices], y[indices]


def run_dataset_extraction(
    raw_videos_root: Path = RAW_VIDEOS_DIR,
    output_file: Path = DATASET_FILE,
    stride: int = WINDOW_STRIDE,
    yolo_weights: str = YOLO_WEIGHTS,
    generate_synthetic: bool = False,
):
    """
    Scans raw video class folders and extracts full training dataset.
    """
    if generate_synthetic:
        X, y = generate_synthetic_demo_dataset(num_samples=500)
        save_processed_dataset(output_file, X, y, metadata={"source": "synthetic_demo_5_classes"})
        print(f"\n[Success] 5-class demo dataset saved to: {output_file}")
        return

    # Check for video files in all 5 class folders
    class_videos: Dict[str, List[Path]] = {}
    total_videos = 0

    for cls in CLASSES:
        cls_dir = RAW_CLASS_DIRS[cls]
        if cls_dir.exists():
            vids = [p for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        else:
            vids = []
        class_videos[cls] = vids
        total_videos += len(vids)

    if total_videos == 0:
        print("\n[Notice] No video files found in the class folders under:")
        print(f"  {raw_videos_root}")
        print("\nExpected folder structure:")
        for cls in CLASSES:
            print(f" - {RAW_CLASS_DIRS[cls]}")
        print("\nPlease add video clips (.mp4, .avi, etc.) to the folders above, or run:")
        print("  python extract_dataset.py --generate-synthetic")
        print("to generate demonstration sequences for all 5 classes.")
        return

    print(f"\n=======================================================")
    print(f" Starting Feature Extraction from Raw Video Datasets")
    print(f" - Target Classes:   {NUM_CLASSES} ({', '.join(CLASSES)})")
    print(f" - Total Videos:     {total_videos}")
    print(f" - Sequence Length:  {SEQUENCE_LENGTH} frames")
    print(f" - Window Stride:    {stride} frames")
    print(f" - YOLO Weights:     {yolo_weights}")
    print(f"=======================================================\n")

    for cls, vids in class_videos.items():
        print(f" - {cls:<30}: {len(vids)} video(s)")

    tracker_manager = PoseTrackerManager(weights_path=yolo_weights, sequence_length=SEQUENCE_LENGTH)

    all_sequences: List[np.ndarray] = []
    all_labels: List[int] = []

    for cls in CLASSES:
        vids = class_videos[cls]
        for vid in vids:
            seqs, lbls = extract_sequences_from_video(vid, cls, tracker_manager, stride=stride)
            all_sequences.extend(seqs)
            all_labels.extend(lbls)

    if len(all_sequences) == 0:
        print("[Error] No pose sequences could be extracted from the videos.")
        return

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Summary breakdown
    print(f"\nExtraction Completed:")
    print(f" - Total Samples: {len(y)}")
    for cls in CLASSES:
        c_idx = CLASS_TO_IDX[cls]
        cnt = np.sum(y == c_idx)
        pct = (100.0 * cnt / len(y)) if len(y) > 0 else 0.0
        print(f" - {cls:<30}: {cnt:<6} ({pct:5.1f}%)")

    save_processed_dataset(output_file, X, y)
    print(f"\n[Success] Processed dataset saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract sequential pose features from exam videos for LSTM training (5 Classes).")
    parser.add_argument("--videos-root", type=str, default=str(RAW_VIDEOS_DIR), help="Root path to raw video class folders")
    parser.add_argument("--output", type=str, default=str(DATASET_FILE), help="Output .npz file path")
    parser.add_argument("--stride", type=int, default=WINDOW_STRIDE, help="Frame stride for window extraction")
    parser.add_argument("--yolo-weights", type=str, default=YOLO_WEIGHTS, help="YOLO pose model weights path")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic demonstration exam dataset for 5 classes")

    args = parser.parse_args()

    run_dataset_extraction(
        raw_videos_root=Path(args.videos_root),
        output_file=Path(args.output),
        stride=args.stride,
        yolo_weights=args.yolo_weights,
        generate_synthetic=args.generate_synthetic,
    )


if __name__ == "__main__":
    main()
