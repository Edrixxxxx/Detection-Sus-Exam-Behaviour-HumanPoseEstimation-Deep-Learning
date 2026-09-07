import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
PROCESSED_DIR = DATA_DIR / "processed"
WEIGHTS_DIR = BASE_DIR / "weights"

# Dataset & Model Paths
DATASET_FILE = PROCESSED_DIR / "exam_dataset.npz"
LSTM_MODEL_PATH = WEIGHTS_DIR / "best_lstm_model.pt"

# Candidate paths for YOLO26s-pose weights
_CANDIDATE_YOLO_WEIGHTS = [
    WEIGHTS_DIR / "yolo26s-pose.pt",
    WEIGHTS_DIR / "yolo26s-pose.engine",
    BASE_DIR.parent / "yolo25s-pose" / "yolo26s-pose.pt",
    BASE_DIR.parent / "yolo25s-pose" / "yolo26s-pose.engine",
    BASE_DIR / "yolo26s-pose.pt",
    BASE_DIR / "yolo11s-pose.pt",
]

def resolve_yolo_weights() -> str:
    """Find the best available YOLO pose weight file."""
    for p in _CANDIDATE_YOLO_WEIGHTS:
        if p.exists():
            return str(p)
    default_path = BASE_DIR.parent / "yolo25s-pose" / "yolo26s-pose.pt"
    if default_path.exists():
        return str(default_path)
    return "yolo11s-pose.pt"

YOLO_WEIGHTS = resolve_yolo_weights()
TRACKER_CONFIG = "bytetrack.yaml"

# Examination Behavior Classes (5 Classes)
CLASSES = [
    "normal",
    "hand_signal",
    "passing_of_notes",
    "side_glancing",
    "use_of_unauthorized_object",
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Class-specific raw video folders
RAW_CLASS_DIRS = {cls: RAW_VIDEOS_DIR / cls for cls in CLASSES}

# Ensure all folders exist
for directory in [DATA_DIR, RAW_VIDEOS_DIR, PROCESSED_DIR, WEIGHTS_DIR, *RAW_CLASS_DIRS.values()]:
    directory.mkdir(parents=True, exist_ok=True)

# Pose & Temporal Sequence Settings
# COCO 17 keypoints: Nose, L/R Eye, L/R Ear, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Hip, L/R Knee, L/R Ankle
NUM_KEYPOINTS = 17
# 2 normalized coordinates (x, y) per keypoint -> 34 features per frame
COORDINATES_PER_KEYPOINT = 2
FEATURE_DIM = NUM_KEYPOINTS * COORDINATES_PER_KEYPOINT  # 34

# Temporal sliding window length (e.g. 30 frames ~= 1 second at 30 FPS)
SEQUENCE_LENGTH = 30
# Stride between consecutive extracted windows during training dataset generation
WINDOW_STRIDE = 5

# LSTM Architecture Hyperparameters
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = True
LSTM_DROPOUT = 0.3

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Dataset Split (70% Train, 15% Validation, 15% Held-out Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Real-time Visualization & Alert Thresholds
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.60
ALERT_DEBOUNCE_FRAMES = 5
