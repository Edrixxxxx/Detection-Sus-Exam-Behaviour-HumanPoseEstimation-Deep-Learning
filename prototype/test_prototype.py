import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

from config import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    NUM_CLASSES,
    CLASSES,
    DATASET_FILE,
    LSTM_MODEL_PATH,
    YOLO_WEIGHTS,
    BASE_DIR,
)
from model import ExamBehaviorLSTM
from dataset import (
    normalize_keypoints,
    split_dataset_70_15_15,
    save_processed_dataset,
    load_processed_dataset,
)
from extract_dataset import generate_synthetic_demo_dataset
from train import train_model
from tracker import PoseTrackerManager


def test_normalization():
    print("-> Testing keypoint normalization...")
    dummy_kpts = np.random.uniform(100, 500, size=(17, 2)).astype(np.float32)
    bbox = np.array([100, 100, 500, 500], dtype=np.float32)
    norm = normalize_keypoints(dummy_kpts, bbox)
    assert norm.shape == (34,), f"Expected shape (34,), got {norm.shape}"
    assert not np.isnan(norm).any(), "NaN found in normalized keypoints"
    print("   [PASS] Keypoint normalization works as expected.")


def test_model_forward():
    print(f"-> Testing ExamBehaviorLSTM forward pass with {NUM_CLASSES} classes...")
    model = ExamBehaviorLSTM(input_dim=34, hidden_dim=64, num_layers=2, num_classes=NUM_CLASSES)
    x = torch.randn(4, 30, 34)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES), f"Expected logits shape (4, {NUM_CLASSES}), got {out.shape}"
    probs = model.predict_proba(x)
    assert probs.shape == (4, NUM_CLASSES), f"Expected probs shape (4, {NUM_CLASSES}), got {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4)), "Probabilities do not sum to 1"
    print(f"   [PASS] Model forward pass and softmax probabilities verified ({NUM_CLASSES} classes).")


def test_70_15_15_split():
    print(f"-> Testing 70-15-15 stratified dataset split across {NUM_CLASSES} classes...")
    samples_per_class = 50
    N = samples_per_class * NUM_CLASSES
    X = np.random.randn(N, 30, 34).astype(np.float32)
    y = np.repeat(np.arange(NUM_CLASSES), samples_per_class).astype(np.int64)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset_70_15_15(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    total_samples = len(X_train) + len(X_val) + len(X_test)
    assert total_samples == N, f"Mismatch: expected {N}, got {total_samples}"

    train_pct = len(X_train) / N
    val_pct = len(X_val) / N
    test_pct = len(X_test) / N
    print(f"   Split result: Train={train_pct*100:.1f}%, Val={val_pct*100:.1f}%, Test={test_pct*100:.1f}%")
    assert np.isclose(train_pct, 0.70, atol=0.02), f"Train ratio off: {train_pct}"
    assert np.isclose(val_pct, 0.15, atol=0.02), f"Val ratio off: {val_pct}"
    assert np.isclose(test_pct, 0.15, atol=0.02), f"Test ratio off: {test_pct}"

    # Check class balance in each split
    for name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = [np.sum(split_y == c) for c in range(NUM_CLASSES)]
        print(f"   {name} distribution across classes: {counts}")
        assert (max(counts) - min(counts)) <= 1, f"Imbalanced split in {name}: {counts}"

    print("   [PASS] 70-15-15 Stratified Split verified across 5 classes.")


def test_end_to_end_training():
    print(f"-> Testing synthetic extraction and training pipeline for {NUM_CLASSES} classes...")
    demo_file = BASE_DIR / "data" / "processed" / "test_demo_dataset.npz"
    X, y = generate_synthetic_demo_dataset(num_samples=250)
    save_processed_dataset(demo_file, X, y)

    model_output = BASE_DIR / "weights" / "test_lstm_model.pt"
    train_model(
        dataset_path=demo_file,
        model_output_path=model_output,
        batch_size=16,
        epochs=6,
        lr=0.005,
        patience=5,
    )
    assert model_output.exists(), f"Model checkpoint not created at: {model_output}"
    print("   [PASS] End-to-end 5-class training and test evaluation succeeded.")


def main():
    print("==================================================")
    print(f" Running Prototype Verification Suite ({NUM_CLASSES} Classes)")
    print("==================================================")
    test_normalization()
    test_model_forward()
    test_70_15_15_split()
    test_end_to_end_training()
    print(f"\n[ALL {NUM_CLASSES}-CLASS TESTS PASSED SUCCESSFULLY!]")


if __name__ == "__main__":
    main()
