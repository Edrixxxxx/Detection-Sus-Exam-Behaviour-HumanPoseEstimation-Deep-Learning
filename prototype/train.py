import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    DATASET_FILE,
    LSTM_MODEL_PATH,
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_BIDIRECTIONAL,
    LSTM_DROPOUT,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    BASE_DIR,
)
from model import ExamBehaviorLSTM
from dataset import ExamPoseDataset, load_processed_dataset, split_dataset_70_15_15


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes accuracy, precision, recall, and F1-score without requiring external libraries.
    """
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion matrix: rows = true, cols = predicted
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    f1_list = []
    prec_list = []
    rec_list = []

    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[IDX_TO_CLASS[c]] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(np.sum(cm[c, :])),
        }
        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(prec_list)),
        "macro_recall": float(np.mean(rec_list)),
        "macro_f1": float(np.mean(f1_list)),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def plot_and_save_reports(
    history: Dict[str, list],
    cm: np.ndarray,
    report_dir: Path,
):
    """Saves training curve plots and confusion matrix heatmap."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training & Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.set_title("Cross-Entropy Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#d62728", lw=2)
    ax2.set_title("Classification Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    curve_path = report_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"-> Saved training curves plot: {curve_path}")

    # 2. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        title="Test Set Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    cm_path = report_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"-> Saved confusion matrix plot: {cm_path}")


def train_model(
    dataset_path: Path = DATASET_FILE,
    model_output_path: Path = LSTM_MODEL_PATH,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Training LSTM Exam Behavior Classifier")
    print(f" - Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" - Dataset: {dataset_path}")
    print(f" - Split: {int(TRAIN_RATIO*100)}% Train / {int(VAL_RATIO*100)}% Val / {int(TEST_RATIO*100)}% Test")
    print(f"=======================================================\n")

    # 1. Load Processed Dataset
    if not dataset_path.exists():
        print(f"[Error] Dataset file not found at: {dataset_path}")
        print("Please extract sequences first using:")
        print("  python extract_dataset.py")
        print("Or generate a synthetic demo dataset with:")
        print("  python extract_dataset.py --generate-synthetic")
        return

    X, y = load_processed_dataset(dataset_path)
    total_samples = len(y)
    print(f"Total dataset samples: {total_samples}, Sequence shape: {X.shape[1:]}")

    # 2. Stratified 70-15-15 Split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset_70_15_15(
        X, y,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
    )

    print("\nDataset Partition Breakdown:")
    print(f" {'Subset':<14} {'Total':<8} {'Ratio':<6} " + " ".join([f"{cls[:8]:<9}" for cls in CLASSES]))
    print(" " + "-" * (32 + 10 * NUM_CLASSES))
    for name, y_subset in [("Train (70%)", y_train), ("Val (15%)", y_val), ("Test (15%)", y_test)]:
        pct = 100.0 * len(y_subset) / total_samples
        counts = [f"{np.sum(y_subset == CLASS_TO_IDX[cls]):<9}" for cls in CLASSES]
        print(f" {name:<14} {len(y_subset):<8} {pct:4.1f}%  " + " ".join(counts))

    # 3. Create PyTorch DataLoaders
    train_loader = DataLoader(ExamPoseDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ExamPoseDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ExamPoseDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 4. Handle Potential Class Imbalance with Loss Weights
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    weights = total_samples / (NUM_CLASSES * np.maximum(class_counts, 1).astype(np.float32))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 5. Initialize Model, Optimizer, and LR Scheduler
    model = ExamBehaviorLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        num_classes=NUM_CLASSES,
        bidirectional=LSTM_BIDIRECTIONAL,
        dropout=LSTM_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    # 6. Training Loop with Early Stopping
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"\n-> Beginning training for up to {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

        train_epoch_loss = train_loss_sum / train_total
        train_epoch_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                val_loss_sum += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_epoch_loss = val_loss_sum / val_total
        val_epoch_acc = val_correct / val_total

        scheduler.step(val_epoch_loss)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_epoch_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc*100:5.1f}% || "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc*100:5.1f}%"
        )

        # Early Stopping Check
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save checkpoint
            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": FEATURE_DIM,
                "hidden_dim": LSTM_HIDDEN_DIM,
                "num_layers": LSTM_NUM_LAYERS,
                "num_classes": NUM_CLASSES,
                "bidirectional": LSTM_BIDIRECTIONAL,
                "classes": CLASSES,
                "sequence_length": SEQUENCE_LENGTH,
                "val_loss": val_epoch_loss,
                "val_acc": val_epoch_acc,
            }, model_output_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] Validation loss did not improve for {patience} consecutive epochs.")
                break

    # 7. Final Evaluation on Held-Out Test Set (15%)
    print(f"\n=======================================================")
    print(f" Evaluating on Held-Out Test Set (15% Unseen Data)")
    print(f"=======================================================")

    # Load best checkpoint
    checkpoint = torch.load(model_output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_trues = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_trues.extend(batch_y.numpy())

    test_preds = np.array(test_preds)
    test_trues = np.array(test_trues)

    metrics = calculate_metrics(test_trues, test_preds)

    print(f"\nTest Set Overall Metrics:")
    print(f" - Overall Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f" - Macro-Average Prec:  {metrics['macro_precision']*100:.2f}%")
    print(f" - Macro-Average Recall:{metrics['macro_recall']*100:.2f}%")
    print(f" - Macro-Average F1:    {metrics['macro_f1']*100:.2f}%")

    print(f"\nPer-Class Breakdown:")
    print(f" {'Class':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print(" " + "-" * 56)
    for cls_name, vals in metrics["per_class"].items():
        print(f" {cls_name:<14} {vals['precision']*100:9.2f}% {vals['recall']*100:9.2f}% {vals['f1']*100:9.2f}% {vals['support']:<8}")

    print(f"\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    header = f" {'Actual \\ Pred':<26} " + " ".join([f"{cls[:10]:<12}" for cls in CLASSES])
    print(header)
    print(" " + "-" * len(header))
    for i, actual_cls in enumerate(CLASSES):
        row_vals = " ".join([f"{metrics['confusion_matrix'][i, j]:<12}" for j in range(NUM_CLASSES)])
        print(f" {actual_cls:<26} {row_vals}")

    # 8. Save Visual Reports
    reports_dir = BASE_DIR / "reports"
    plot_and_save_reports(history, metrics["confusion_matrix"], reports_dir)

    # Save metrics summary text file
    metrics_file = reports_dir / "test_evaluation_report.txt"
    with open(metrics_file, "w") as f:
        f.write("Examination Behavior Classifier - Test Evaluation Report\n")
        f.write("========================================================\n")
        f.write(f"Test Accuracy:    {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Macro Precision:  {metrics['macro_precision']*100:.2f}%\n")
        f.write(f"Macro Recall:     {metrics['macro_recall']*100:.2f}%\n")
        f.write(f"Macro F1-Score:   {metrics['macro_f1']*100:.2f}%\n\n")
        f.write("Per-Class Details:\n")
        for cls_name, vals in metrics["per_class"].items():
            f.write(f" - {cls_name}: Prec={vals['precision']*100:.2f}%, Rec={vals['recall']*100:.2f}%, F1={vals['f1']*100:.2f}%, Support={vals['support']}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")

    print(f"-> Saved evaluation report: {metrics_file}")
    print(f"-> Best model saved to:     {model_output_path}")
    print(f"\n[Success] Training and evaluation completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Examination Behavior Classifier with 70-15-15 Split.")
    parser.add_argument("--dataset", type=str, default=str(DATASET_FILE), help="Path to processed .npz dataset")
    parser.add_argument("--output", type=str, default=str(LSTM_MODEL_PATH), help="Path to save trained LSTM model checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience")

    args = parser.parse_args()

    train_model(
        dataset_path=Path(args.dataset),
        model_output_path=Path(args.output),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
