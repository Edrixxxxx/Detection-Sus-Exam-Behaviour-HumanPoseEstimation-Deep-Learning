import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from config import (
    FEATURE_DIM,
    NUM_KEYPOINTS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def normalize_keypoints(
    kpts: np.ndarray,
    bbox: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalizes 17 COCO pose keypoints (x, y) to make them invariant to
    camera distance, video resolution, and person placement in the frame.

    Args:
        kpts: np.ndarray of shape (17, 2) or (17, 3) where columns are [x, y] or [x, y, conf].
        bbox: Optional [x1, y1, x2, y2] bounding box of the person.

    Returns:
        np.ndarray of shape (34,) representing flattened normalized (x, y) keypoints.
    """
    coords = kpts[:, :2].astype(np.float32)

    # 1. Determine the center reference point (mid-hip or mid-shoulder or bbox center)
    l_hip, r_hip = coords[11], coords[12]
    l_shoulder, r_shoulder = coords[5], coords[6]

    # Check if hips are non-zero
    if np.any(l_hip > 0) and np.any(r_hip > 0):
        center_x = (l_hip[0] + r_hip[0]) / 2.0
        center_y = (l_hip[1] + r_hip[1]) / 2.0
    elif np.any(l_shoulder > 0) and np.any(r_shoulder > 0):
        center_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
        center_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
    elif bbox is not None and len(bbox) >= 4:
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
    else:
        # Fallback to mean of non-zero coordinates
        valid = coords[np.any(coords > 0, axis=1)]
        if len(valid) > 0:
            center_x, center_y = np.mean(valid[:, 0]), np.mean(valid[:, 1])
        else:
            center_x, center_y = 0.0, 0.0

    # 2. Determine the scale factor (bounding box diagonal or torso height)
    if bbox is not None and len(bbox) >= 4:
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        scale = np.sqrt(w * w + h * h)
    else:
        # Distance between mid-shoulder and mid-hip
        torso_dist = np.linalg.norm(
            np.array([(l_shoulder[0] + r_shoulder[0]) / 2.0, (l_shoulder[1] + r_shoulder[1]) / 2.0]) -
            np.array([center_x, center_y])
        )
        scale = torso_dist * 2.5 if torso_dist > 5.0 else 200.0

    scale = max(scale, 1e-3)

    # 3. Normalize coordinates: zero-center and scale
    normalized = np.zeros_like(coords)
    mask = np.any(coords > 0, axis=1)
    normalized[mask, 0] = (coords[mask, 0] - center_x) / scale
    normalized[mask, 1] = (coords[mask, 1] - center_y) / scale

    return normalized.flatten().astype(np.float32)


class ExamPoseDataset(Dataset):
    """PyTorch Dataset for temporal sequences of normalized human pose keypoints."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: np.ndarray of shape (N, sequence_length, feature_dim)
            y: np.ndarray of shape (N,) with integer class labels (0 or 1)
        """
        assert len(X) == len(y), f"Mismatched data sizes: X={len(X)}, y={len(y)}"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def split_dataset_70_15_15(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Performs a robust, stratified 70-15-15 split to ensure equal class
    distributions in Train, Validation, and Test sets.

    Args:
        X: Sequence features of shape (N, seq_len, feature_dim)
        y: Class labels of shape (N,)
        train_ratio: Default 0.70 (70%)
        val_ratio: Default 0.15 (15%)
        test_ratio: Default 0.15 (15%)
        random_seed: Random seed for reproducibility

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    rng = np.random.RandomState(random_seed)

    train_indices = []
    val_indices = []
    test_indices = []

    unique_classes = np.unique(y)

    # Perform stratified partitioning per class
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        n_samples = len(cls_indices)
        n_train = int(round(n_samples * train_ratio))
        n_val = int(round(n_samples * val_ratio))
        # Ensure at least 1 sample in val and test if possible
        if n_samples >= 3:
            n_train = max(1, min(n_train, n_samples - 2))
            n_val = max(1, min(n_val, n_samples - n_train - 1))

        train_idx = cls_indices[:n_train]
        val_idx = cls_indices[n_train:n_train + n_val]
        test_idx = cls_indices[n_train + n_val:]

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

    # Shuffle each partition
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        (X[train_indices], y[train_indices]),
        (X[val_indices], y[val_indices]),
        (X[test_indices], y[test_indices]),
    )


def save_processed_dataset(
    filepath: Path,
    X: np.ndarray,
    y: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Saves sequences and labels to a compressed .npz file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"X": X, "y": y}
    if metadata is not None:
        save_dict["metadata"] = np.array(metadata, dtype=object)
    np.savez_compressed(filepath, **save_dict)
    print(f"-> Saved dataset to {filepath} [Samples: {len(y)}, X shape: {X.shape}]")


def load_processed_dataset(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Loads sequences and labels from a compressed .npz file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    X, y = data["X"], data["y"]
    return X, y
