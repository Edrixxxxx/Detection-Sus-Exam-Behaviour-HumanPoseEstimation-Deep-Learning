"""
AI Examination Behavior Proctoring System - Modern PyQt6 Desktop UI
====================================================================
Provides an intuitive, graphical interface for real-time exam surveillance.
Users can seamlessly select between live webcams or recorded video clips,
view real-time pose tracking overlays and behavior classifications, adjust
confidence thresholds, view live suspicious incident feeds, and export audit reports
without typing command-line arguments in the terminal.

Architecture:
- PyQt6 Event Loop on Main GUI Thread
- GPU/CPU Accelerated Computer Vision & Deep Learning Pipeline on Dedicated QThread
- Supports PyTorch LSTM Classifier, Hybrid Ensemble, Heuristics, and Simulation
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any

import cv2
import numpy as np
import torch

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QIcon, QPainter, QBrush, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QLineEdit,
    QSlider,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QProgressBar,
    QFrame,
    QSizePolicy,
)

# Project Module Imports
try:
    from config import (
        YOLO_WEIGHTS,
        LSTM_MODEL_PATH,
        CLASSES,
        IDX_TO_CLASS,
        SUSPICIOUS_CONFIDENCE_THRESHOLD,
        SEQUENCE_LENGTH,
        ALERT_DEBOUNCE_FRAMES,
        resolve_yolo_weights,
        DATA_DIR,
        RAW_VIDEOS_DIR,
        REPORTS_DIR,
    )
    from tracker import PoseTrackerManager
    from model_adapter import (
        ModelRegistry,
        BaseAntiCheatingModel,
        LSTMModelAdapter,
        HeuristicRuleModelAdapter,
        HybridEnsembleModelAdapter,
        MockSimulationModelAdapter,
    )
    from inference import (
        CLASS_COLORS,
        SKELETON_PAIRS,
        draw_proctoring_overlay,
        draw_hud,
    )
except ImportError:
    # If launched from another directory
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from config import (
        YOLO_WEIGHTS,
        LSTM_MODEL_PATH,
        CLASSES,
        IDX_TO_CLASS,
        SUSPICIOUS_CONFIDENCE_THRESHOLD,
        SEQUENCE_LENGTH,
        ALERT_DEBOUNCE_FRAMES,
        resolve_yolo_weights,
        DATA_DIR,
        RAW_VIDEOS_DIR,
        REPORTS_DIR,
    )
    from tracker import PoseTrackerManager
    from model_adapter import (
        ModelRegistry,
        BaseAntiCheatingModel,
        LSTMModelAdapter,
        HeuristicRuleModelAdapter,
        HybridEnsembleModelAdapter,
        MockSimulationModelAdapter,
    )
    from inference import (
        CLASS_COLORS,
        SKELETON_PAIRS,
        draw_proctoring_overlay,
        draw_hud,
    )

# Evidence Snapshots Directory
SNAPSHOTS_DIR = DATA_DIR / "evidence_snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def scan_available_cameras(max_test: int = 4) -> List[int]:
    """Probes video capture devices to detect available webcams without hanging."""
    available = []
    for idx in range(max_test):
        try:
            # Test with DirectShow on Windows for fastest probe
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        except Exception:
            pass
    # If DirectShow probe failed to find any, fallback to standard probe for index 0
    if not available:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            available.append(0)
            cap.release()
    return available if available else [0]


def play_chime():
    """Plays a quick system chime for high-severity alerts in a non-blocking background thread."""
    def _beep():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


class VideoInferenceWorker(QThread):
    """
    Background worker thread running YOLO26s-Pose tracking, sequential LSTM
    classification, and HUD drawing. Ensures the PyQt6 GUI remains silky smooth.
    """
    frame_ready = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(float, int, int, str)  # fps, active_examinees, total_alerts, threat_level
    alert_detected = pyqtSignal(dict)  # Alert payload dictionary
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        source: str,
        yolo_weights: str,
        lstm_weights: Path,
        model_type: str = "lstm",
        threshold: float = SUSPICIOUS_CONFIDENCE_THRESHOLD,
        loop_video: bool = False,
        save_video_path: Optional[str] = None,
        show_bbox: bool = True,
        show_skeleton: bool = True,
        show_labels: bool = True,
        show_banners: bool = True,
    ):
        super().__init__()
        self.source = source
        self.yolo_weights = yolo_weights
        self.lstm_weights = lstm_weights
        self.model_type = model_type
        self.threshold = threshold
        self.loop_video = loop_video
        self.save_video_path = save_video_path

        self.show_bbox = show_bbox
        self.show_skeleton = show_skeleton
        self.show_labels = show_labels
        self.show_banners = show_banners

        self._is_running = True
        self._is_paused = False
        self._pending_snapshot: Optional[str] = None

    def update_settings(
        self,
        threshold: Optional[float] = None,
        show_bbox: Optional[bool] = None,
        show_skeleton: Optional[bool] = None,
        show_labels: Optional[bool] = None,
        show_banners: Optional[bool] = None,
    ):
        """Allows dynamic adjustment of settings while the stream is active."""
        if threshold is not None:
            self.threshold = threshold
        if show_bbox is not None:
            self.show_bbox = show_bbox
        if show_skeleton is not None:
            self.show_skeleton = show_skeleton
        if show_labels is not None:
            self.show_labels = show_labels
        if show_banners is not None:
            self.show_banners = show_banners

    def request_snapshot(self, output_path: str):
        """Thread-safe request to save the current annotated frame as evidence."""
        self._pending_snapshot = output_path

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status_changed.emit(f"Initializing CV models on {device}...")

        # Parse source (camera index or video path)
        video_source = int(self.source) if self.source.isdigit() else self.source

        # Open video capture
        if isinstance(video_source, int) and sys.platform == "win32":
            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            self.error_occurred.emit(f"Failed to open video source: {self.source}")
            self.finished.emit()
            return

        # Initialize Video Writer if requested
        writer = None
        if self.save_video_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.save_video_path, fourcc, fps_in, (width, height))

        # Load Pose Tracker
        try:
            self.status_changed.emit("Loading YOLO pose tracker...")
            tracker_manager = PoseTrackerManager(
                weights_path=self.yolo_weights,
                sequence_length=SEQUENCE_LENGTH,
            )
        except Exception as e:
            self.error_occurred.emit(f"Error loading YOLO weights ({self.yolo_weights}): {e}")
            cap.release()
            self.finished.emit()
            return

        # Load Behavior Classification Model Adapter
        try:
            self.status_changed.emit(f"Loading {self.model_type.upper()} classifier...")
            if self.model_type == "heuristic":
                classifier_adapter = HeuristicRuleModelAdapter()
            elif self.model_type == "hybrid":
                classifier_adapter = HybridEnsembleModelAdapter(
                    weights_path=self.lstm_weights,
                    device=device,
                )
            elif self.model_type == "simulation":
                classifier_adapter = MockSimulationModelAdapter()
            else:
                # Default: LSTM
                classifier_adapter = LSTMModelAdapter(
                    weights_path=self.lstm_weights,
                    device=device,
                )
        except Exception as e:
            self.error_occurred.emit(f"Error initializing classifier: {e}")
            classifier_adapter = None

        self.status_changed.emit("Proctoring surveillance active.")

        prev_time = time.time()
        total_suspicious_incidents = 0
        frame_idx = 0

        while self._is_running:
            if self._is_paused:
                self.msleep(40)
                continue

            ret, frame = cap.read()
            if not ret:
                if self.loop_video and not str(self.source).isdigit():
                    # Rewind video clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.status_changed.emit("End of video stream reached.")
                    break

            frame_idx += 1
            curr_time = time.time()
            elapsed = curr_time - prev_time
            fps = (1.0 / elapsed) if elapsed > 0 else 30.0
            prev_time = curr_time

            # 1. YOLO Pose Tracking with ByteTrack
            try:
                detections = tracker_manager.process_frame(frame)
            except Exception as e:
                print(f"[Worker] Pose tracking error on frame {frame_idx}: {e}")
                detections = []

            active_examinees = len(detections)
            current_frame_suspicious = 0

            # 2. Sequential Behavior Classification per tracked examinee
            for det in detections:
                person = det["tracked_person"]
                is_ready = det["sequence_ready"]
                seq = det["sequence"]

                if classifier_adapter is not None and is_ready and seq is not None:
                    # Run model adapter inference
                    prediction = classifier_adapter.predict_sequence(
                        sequence=seq,
                        raw_keypoints=det["raw_keypoints"],
                        bbox=det["bbox"],
                        context={"alert_duration_frames": person.alert_counter},
                    )

                    pred_class = prediction.predicted_class
                    conf = prediction.confidence
                    severity = prediction.severity_level

                    # Debounce and threshold check
                    if pred_class != "normal" and conf >= self.threshold:
                        person.alert_counter = min(ALERT_DEBOUNCE_FRAMES * 2, person.alert_counter + 2)
                        current_frame_suspicious += 1
                        person.current_prediction = pred_class
                        person.current_confidence = conf

                        # Trigger alert event once debounce threshold is hit
                        if person.alert_counter == ALERT_DEBOUNCE_FRAMES:
                            total_suspicious_incidents += 1
                            alert_payload = {
                                "timestamp": time.strftime("%H:%M:%S"),
                                "track_id": det["track_id"],
                                "behavior": pred_class,
                                "confidence": conf,
                                "severity": severity,
                                "explanation": prediction.explanation,
                            }
                            self.alert_detected.emit(alert_payload)
                    else:
                        person.alert_counter = max(0, person.alert_counter - 1)
                        if person.alert_counter == 0:
                            person.current_prediction = "normal"
                            person.current_confidence = conf if pred_class == "normal" else (1.0 - conf)

                    # Custom overlay rendering respecting user toggles
                    self._draw_overlay(
                        frame=frame,
                        detection=det,
                        pred_label=person.current_prediction,
                        confidence=person.current_confidence,
                        is_warming_up=False,
                        warmup_count=SEQUENCE_LENGTH,
                    )
                else:
                    # Buffer warming up
                    warmup_len = len(person.buffer)
                    self._draw_overlay(
                        frame=frame,
                        detection=det,
                        pred_label="normal",
                        confidence=0.0,
                        is_warming_up=True,
                        warmup_count=warmup_len,
                    )

            # 3. Threat Level Determination
            if current_frame_suspicious >= 2:
                threat_level = "CRITICAL"
            elif current_frame_suspicious == 1:
                threat_level = "ALERT"
            else:
                threat_level = "NORMAL"

            # 4. Draw Top HUD Bar
            draw_hud(frame, fps, active_examinees, total_suspicious_incidents)

            # 5. Handle snapshot request if pending
            if self._pending_snapshot:
                try:
                    cv2.imwrite(self._pending_snapshot, frame)
                    self.status_changed.emit(f"Snapshot saved: {Path(self._pending_snapshot).name}")
                except Exception as e:
                    self.error_occurred.emit(f"Snapshot failed: {e}")
                self._pending_snapshot = None

            # 6. Record frame if enabled
            if writer is not None:
                writer.write(frame)

            # 7. Emit frame and statistics to GUI thread
            self.frame_ready.emit(frame)
            self.stats_updated.emit(fps, active_examinees, total_suspicious_incidents, threat_level)

        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        self.finished.emit()

    def _draw_overlay(
        self,
        frame: np.ndarray,
        detection: dict,
        pred_label: str,
        confidence: float,
        is_warming_up: bool,
        warmup_count: int,
    ):
        """Custom overlay drawing that respects UI toggle settings."""
        bbox = detection["bbox"].astype(int)
        kpts = detection["raw_keypoints"]
        tid = detection["track_id"]
        x1, y1, x2, y2 = bbox

        if is_warming_up:
            box_color = (180, 180, 180)
            status_text = f"ID: {tid} | WARMING UP ({warmup_count}/{SEQUENCE_LENGTH})"
        else:
            box_color = CLASS_COLORS.get(pred_label, (0, 0, 245))
            label_display = pred_label.upper().replace("_", " ")
            status_text = f"ID: {tid} | {label_display} {confidence*100:.0f}%"

        # Bounding Box
        if self.show_bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Label Header Badge
        if self.show_labels:
            (w, h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            badge_y1 = max(0, y1 - h - 10)
            badge_y2 = max(h + 10, y1)
            badge_x2 = min(frame.shape[1], x1 + w + 14)
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
            cv2.putText(
                frame, status_text, (x1 + 6, badge_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Skeleton Bones & Joints
        if self.show_skeleton:
            for p1, p2 in SKELETON_PAIRS:
                pt1, pt2 = kpts[p1], kpts[p2]
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(
                        frame,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        box_color, 2, cv2.LINE_AA
                    )
            for pt in kpts:
                if pt[0] > 0 and pt[1] > 0:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

        # Alert Banner
        if self.show_banners and pred_label != "normal" and not is_warming_up:
            banner_text = f"! ALERT: {pred_label.upper().replace('_', ' ')} !"
            cv2.putText(
                frame, banner_text, (x1, min(frame.shape[0] - 10, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2, cv2.LINE_AA
            )


# ==============================================================================
# Modern Dark UI Stylesheet (QSS)
# ==============================================================================
DARK_THEME_QSS = """
QMainWindow {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

QWidget {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    color: #e2e8f0;
}

QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-top: 14px;
    font-weight: 600;
    font-size: 13px;
    color: #38bdf8;
    background-color: #111827;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #111827;
}

QFrame.card {
    background-color: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 8px;
}

QFrame.kpi-card {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e293b, stop:1 #0f172a);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px;
}

QLabel {
    color: #cbd5e1;
    font-size: 13px;
}

QLabel.header-title {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
}

QLabel.header-subtitle {
    font-size: 12px;
    color: #94a3b8;
}

QLabel.kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #38bdf8;
}

QLabel.kpi-title {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
}

QPushButton {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}

QPushButton:pressed {
    background-color: #0f172a;
}

QPushButton:disabled {
    background-color: #0f172a;
    color: #64748b;
    border-color: #1e293b;
}

QPushButton.btn-primary {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10b981, stop:1 #059669);
    border: 1px solid #10b981;
    color: #ffffff;
    font-size: 14px;
    padding: 10px 20px;
}

QPushButton.btn-primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34d399, stop:1 #10b981);
}

QPushButton.btn-pause {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f59e0b, stop:1 #d97706);
    border: 1px solid #f59e0b;
    color: #ffffff;
    font-size: 14px;
    padding: 10px 18px;
}

QPushButton.btn-pause:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fbbf24, stop:1 #f59e0b);
}

QPushButton.btn-danger {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ef4444, stop:1 #dc2626);
    border: 1px solid #ef4444;
    color: #ffffff;
    font-size: 14px;
    padding: 10px 18px;
}

QPushButton.btn-danger:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f87171, stop:1 #ef4444);
}

QPushButton.btn-action {
    background-color: #0284c7;
    border: 1px solid #38bdf8;
    color: #ffffff;
}

QPushButton.btn-action:hover {
    background-color: #0369a1;
}

QLineEdit, QComboBox {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
    font-size: 13px;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #38bdf8;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    selection-background-color: #0284c7;
    color: #f8fafc;
}

QRadioButton, QCheckBox {
    font-size: 13px;
    color: #e2e8f0;
    spacing: 8px;
}

QRadioButton::indicator, QCheckBox::indicator {
    width: 18px;
    height: 18px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #f8fafc;
    border: 2px solid #38bdf8;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #38bdf8;
}

QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #111827;
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background: #0f172a;
    border: 1px solid #1e293b;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #94a3b8;
    font-weight: 600;
}

QTabBar::tab:selected {
    background: #111827;
    border-bottom: 2px solid #38bdf8;
    color: #38bdf8;
}

QTabBar::tab:hover:!selected {
    background: #1e293b;
    color: #e2e8f0;
}

QTableWidget {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    gridline-color: #1e293b;
    border-radius: 6px;
    color: #f1f5f9;
}

QTableWidget::item {
    padding: 4px 8px;
}

QTableWidget::item:selected {
    background-color: #1e293b;
}

QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    font-weight: 600;
    padding: 6px;
    border: none;
    border-right: 1px solid #0f172a;
    border-bottom: 1px solid #334155;
}

QStatusBar {
    background-color: #0f172a;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
}
"""


class ExamProctorWindow(QMainWindow):
    """
    Primary Desktop Application Window for the AI Exam Proctoring Surveillance System.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Examination Behavior Proctoring System | YOLO26s-Pose + ByteTrack + LSTM")
        self.resize(1340, 860)
        self.setMinimumSize(1080, 680)

        # Worker instance
        self.worker: Optional[VideoInferenceWorker] = None
        self.selected_video_path: Optional[str] = None
        self.is_paused = False

        # Alert audio alarm toggle
        self.audio_alert_enabled = True

        # Build Interface
        self._init_ui()
        self.apply_stylesheet()

        # Update initial device badge
        device_str = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        self.badge_device.setText(f"Device: {device_str}")
        if torch.cuda.is_available():
            self.badge_device.setStyleSheet("background-color: #065f46; color: #34d399; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 11px;")
        else:
            self.badge_device.setStyleSheet("background-color: #374151; color: #9ca3af; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 11px;")

        # Populate Webcams on startup
        self.refresh_cameras()

    def apply_stylesheet(self):
        self.setStyleSheet(DARK_THEME_QSS)

    def _init_ui(self):
        """Constructs layout, widgets, and signal bindings."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. Top Header Bar
        # -------------------------------------------------------------
        header_layout = QHBoxLayout()

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        lbl_title = QLabel("AI Examination Behavior Proctoring System")
        lbl_title.setObjectName("lbl_title")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f8fafc;")

        lbl_subtitle = QLabel("Real-Time Suspicious Behavior Surveillance | Dual Pose & Temporal LSTM Engine")
        lbl_subtitle.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)

        header_layout.addStretch()

        # Status Badges
        self.badge_device = QLabel("Device: Probing...")
        header_layout.addWidget(self.badge_device)

        self.badge_status = QLabel("STATUS: IDLE")
        self.badge_status.setStyleSheet("background-color: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 11px;")
        header_layout.addWidget(self.badge_status)

        main_layout.addLayout(header_layout)

        # -------------------------------------------------------------
        # 2. Main Content Splitter (Left: Video/Controls, Right: Dashboard/Alerts)
        # -------------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # =============================================================
        # Left Panel (Source Selection, Action Bar, Video Canvas)
        # =============================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        # A. Video Source Selection Card
        source_group = QGroupBox("1. Video Source Selection (Webcam or Clip)")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)

        # Mode Selection Radio Buttons
        mode_layout = QHBoxLayout()
        self.rb_webcam = QRadioButton("📹 Live Webcam")
        self.rb_clip = QRadioButton("📁 Recorded Video Clip (.mp4 / .avi)")
        self.rb_webcam.setChecked(True)

        self.btn_group_source = QButtonGroup(self)
        self.btn_group_source.addButton(self.rb_webcam, 1)
        self.btn_group_source.addButton(self.rb_clip, 2)
        self.btn_group_source.idClicked.connect(self._on_source_mode_changed)

        mode_layout.addWidget(self.rb_webcam)
        mode_layout.addWidget(self.rb_clip)
        mode_layout.addStretch()
        source_layout.addLayout(mode_layout)

        # Webcam Selector Controls (Container)
        self.webcam_container = QWidget()
        webcam_layout = QHBoxLayout(self.webcam_container)
        webcam_layout.setContentsMargins(0, 4, 0, 4)

        lbl_cam = QLabel("Camera Device:")
        self.combo_cameras = QComboBox()
        self.combo_cameras.setMinimumWidth(220)

        self.btn_refresh_cams = QPushButton("🔄 Scan Cameras")
        self.btn_refresh_cams.clicked.connect(self.refresh_cameras)

        self.edit_custom_cam = QLineEdit()
        self.edit_custom_cam.setPlaceholderText("Or enter index / RTSP URL (e.g. 0, rtsp://...)")
        self.edit_custom_cam.setMaximumWidth(280)

        webcam_layout.addWidget(lbl_cam)
        webcam_layout.addWidget(self.combo_cameras)
        webcam_layout.addWidget(self.btn_refresh_cams)
        webcam_layout.addWidget(self.edit_custom_cam)
        webcam_layout.addStretch()
        source_layout.addWidget(self.webcam_container)

        # Video Clip Selector Controls (Container)
        self.clip_container = QWidget()
        clip_layout = QHBoxLayout(self.clip_container)
        clip_layout.setContentsMargins(0, 4, 0, 4)

        self.btn_browse_clip = QPushButton("📂 Browse Video File...")
        self.btn_browse_clip.clicked.connect(self._browse_video_file)

        self.lbl_selected_clip = QLineEdit()
        self.lbl_selected_clip.setReadOnly(True)
        self.lbl_selected_clip.setPlaceholderText("No video file selected yet. Click 'Browse Video File...'")

        self.chk_loop = QCheckBox("🔁 Loop Playback")
        self.chk_loop.setChecked(True)

        clip_layout.addWidget(self.btn_browse_clip)
        clip_layout.addWidget(self.lbl_selected_clip)
        clip_layout.addWidget(self.chk_loop)
        source_layout.addWidget(self.clip_container)
        self.clip_container.setVisible(False)  # Initially hidden since webcam is selected

        left_layout.addWidget(source_group)

        # B. Real-Time Action Control Buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.btn_start = QPushButton("▶  Start Proctoring")
        self.btn_start.setProperty("class", "btn-primary")
        self.btn_start.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10b981, stop:1 #059669); border: 1px solid #10b981; color: white; font-weight: 700; font-size: 14px; padding: 10px 22px; border-radius: 6px;")
        self.btn_start.clicked.connect(self.start_proctoring)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setStyleSheet("background: #f59e0b; border: 1px solid #f59e0b; color: white; font-weight: 700; font-size: 14px; padding: 10px 18px; border-radius: 6px;")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setStyleSheet("background: #ef4444; border: 1px solid #ef4444; color: white; font-weight: 700; font-size: 14px; padding: 10px 18px; border-radius: 6px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_proctoring)

        self.btn_snapshot = QPushButton("📸  Capture Snapshot")
        self.btn_snapshot.setStyleSheet("background: #0284c7; border: 1px solid #38bdf8; color: white; font-weight: 600; font-size: 13px; padding: 10px 16px; border-radius: 6px;")
        self.btn_snapshot.setEnabled(False)
        self.btn_snapshot.clicked.connect(self.capture_snapshot)

        action_layout.addWidget(self.btn_start)
        action_layout.addWidget(self.btn_pause)
        action_layout.addWidget(self.btn_stop)
        action_layout.addWidget(self.btn_snapshot)
        action_layout.addStretch()

        # Record video checkbox
        self.chk_record = QCheckBox("Record Annotated Video (.mp4)")
        action_layout.addWidget(self.chk_record)

        left_layout.addLayout(action_layout)

        # C. Video Display Canvas
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: #050811; border: 1px solid #1e293b; border-radius: 8px;")
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.setContentsMargins(4, 4, 4, 4)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(640, 420)
        self._set_placeholder_view()

        video_layout.addWidget(self.video_label)
        left_layout.addWidget(self.video_frame, stretch=1)

        splitter.addWidget(left_widget)

        # =============================================================
        # Right Panel (KPI Dashboard, Alert Feed Table, Model Config)
        # =============================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # A. KPI Metric Cards
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(8)

        # Card 1: Active Examinees
        card_students = QFrame()
        card_students.setProperty("class", "kpi-card")
        card_students.setStyleSheet("background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;")
        c1_layout = QVBoxLayout(card_students)
        c1_layout.setContentsMargins(8, 6, 8, 6)
        lbl_c1_title = QLabel("ACTIVE EXAMINEES")
        lbl_c1_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #94a3b8;")
        self.val_active_students = QLabel("0")
        self.val_active_students.setStyleSheet("font-size: 26px; font-weight: 800; color: #38bdf8;")
        c1_layout.addWidget(lbl_c1_title)
        c1_layout.addWidget(self.val_active_students)
        kpi_grid.addWidget(card_students, 0, 0)

        # Card 2: Alerts Count
        card_alerts = QFrame()
        card_alerts.setStyleSheet("background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;")
        c2_layout = QVBoxLayout(card_alerts)
        c2_layout.setContentsMargins(8, 6, 8, 6)
        lbl_c2_title = QLabel("TOTAL ALERTS")
        lbl_c2_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #94a3b8;")
        self.val_total_alerts = QLabel("0")
        self.val_total_alerts.setStyleSheet("font-size: 26px; font-weight: 800; color: #f87171;")
        c2_layout.addWidget(lbl_c2_title)
        c2_layout.addWidget(self.val_total_alerts)
        kpi_grid.addWidget(card_alerts, 0, 1)

        # Card 3: FPS
        card_fps = QFrame()
        card_fps.setStyleSheet("background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;")
        c3_layout = QVBoxLayout(card_fps)
        c3_layout.setContentsMargins(8, 6, 8, 6)
        lbl_c3_title = QLabel("PIPELINE FPS")
        lbl_c3_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #94a3b8;")
        self.val_fps = QLabel("0.0")
        self.val_fps.setStyleSheet("font-size: 26px; font-weight: 800; color: #34d399;")
        c3_layout.addWidget(lbl_c3_title)
        c3_layout.addWidget(self.val_fps)
        kpi_grid.addWidget(card_fps, 1, 0)

        # Card 4: Threat Level
        card_status = QFrame()
        card_status.setStyleSheet("background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;")
        c4_layout = QVBoxLayout(card_status)
        c4_layout.setContentsMargins(8, 6, 8, 6)
        lbl_c4_title = QLabel("CLASSROOM STATUS")
        lbl_c4_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #94a3b8;")
        self.val_threat = QLabel("NORMAL")
        self.val_threat.setStyleSheet("font-size: 20px; font-weight: 800; color: #10b981;")
        c4_layout.addWidget(lbl_c4_title)
        c4_layout.addWidget(self.val_threat)
        kpi_grid.addWidget(card_status, 1, 1)

        right_layout.addLayout(kpi_grid)

        # B. Tabbed Dashboard
        self.tabs = QTabWidget()

        # Tab 1: Live Alert Feed
        tab_alerts = QWidget()
        tab_alerts_layout = QVBoxLayout(tab_alerts)
        tab_alerts_layout.setContentsMargins(8, 10, 8, 8)
        tab_alerts_layout.setSpacing(8)

        # Alert Feed Tools
        tools_layout = QHBoxLayout()
        self.chk_auto_scroll = QCheckBox("Auto-scroll")
        self.chk_auto_scroll.setChecked(True)

        self.chk_audio = QCheckBox("🔊 Chime on Red Alert")
        self.chk_audio.setChecked(True)
        self.chk_audio.toggled.connect(self._toggle_audio_alert)

        self.btn_export_csv = QPushButton("💾 Export CSV")
        self.btn_export_csv.clicked.connect(self._export_alerts_csv)

        self.btn_clear_alerts = QPushButton("🗑️ Clear Log")
        self.btn_clear_alerts.clicked.connect(self._clear_alerts)

        tools_layout.addWidget(self.chk_auto_scroll)
        tools_layout.addWidget(self.chk_audio)
        tools_layout.addStretch()
        tools_layout.addWidget(self.btn_export_csv)
        tools_layout.addWidget(self.btn_clear_alerts)
        tab_alerts_layout.addLayout(tools_layout)

        # Table of Alerts
        self.table_alerts = QTableWidget()
        self.table_alerts.setColumnCount(5)
        self.table_alerts.setHorizontalHeaderLabels(["Time", "ID", "Suspicious Behavior", "Conf.", "Severity"])
        self.table_alerts.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_alerts.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_alerts.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_alerts.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_alerts.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_alerts.verticalHeader().setVisible(False)
        self.table_alerts.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tab_alerts_layout.addWidget(self.table_alerts)

        self.tabs.addTab(tab_alerts, "🚨 Real-Time Alerts")

        # Tab 2: Settings & Thresholds
        tab_settings = QWidget()
        tab_settings_layout = QVBoxLayout(tab_settings)
        tab_settings_layout.setContentsMargins(10, 10, 10, 10)
        tab_settings_layout.setSpacing(12)

        # Classification Engine
        lbl_model_engine = QLabel("Classification Engine Adapter:")
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("Deep Learning: PyTorch LSTM (Trained Weights)", "lstm")
        self.combo_engine.addItem("Hybrid Ensemble: LSTM + Geometric Heuristics", "hybrid")
        self.combo_engine.addItem("Deterministic Geometric Heuristics (Zero-GPU)", "heuristic")
        self.combo_engine.addItem("Mock Simulation Stream (Test Pattern)", "simulation")
        tab_settings_layout.addWidget(lbl_model_engine)
        tab_settings_layout.addWidget(self.combo_engine)

        # Threshold Slider
        thresh_layout = QHBoxLayout()
        lbl_thresh = QLabel("Suspicious Alert Threshold:")
        self.lbl_thresh_val = QLabel(f"{int(SUSPICIOUS_CONFIDENCE_THRESHOLD * 100)}%")
        self.lbl_thresh_val.setStyleSheet("font-weight: 700; color: #38bdf8;")
        thresh_layout.addWidget(lbl_thresh)
        thresh_layout.addStretch()
        thresh_layout.addWidget(self.lbl_thresh_val)
        tab_settings_layout.addLayout(thresh_layout)

        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(20, 95)
        self.slider_thresh.setValue(int(SUSPICIOUS_CONFIDENCE_THRESHOLD * 100))
        self.slider_thresh.valueChanged.connect(self._on_threshold_changed)
        tab_settings_layout.addWidget(self.slider_thresh)

        # YOLO Weights Path
        lbl_yolo = QLabel("YOLO Pose Model Weights:")
        self.edit_yolo_weights = QLineEdit(YOLO_WEIGHTS)
        btn_browse_yolo = QPushButton("Browse...")
        btn_browse_yolo.clicked.connect(self._browse_yolo_weights)
        yolo_box = QHBoxLayout()
        yolo_box.addWidget(self.edit_yolo_weights)
        yolo_box.addWidget(btn_browse_yolo)
        tab_settings_layout.addWidget(lbl_yolo)
        tab_settings_layout.addLayout(yolo_box)

        # LSTM Weights Path
        lbl_lstm = QLabel("LSTM Classifier Checkpoint (.pt):")
        self.edit_lstm_weights = QLineEdit(str(LSTM_MODEL_PATH))
        btn_browse_lstm = QPushButton("Browse...")
        btn_browse_lstm.clicked.connect(self._browse_lstm_weights)
        lstm_box = QHBoxLayout()
        lstm_box.addWidget(self.edit_lstm_weights)
        lstm_box.addWidget(btn_browse_lstm)
        tab_settings_layout.addWidget(lbl_lstm)
        tab_settings_layout.addLayout(lstm_box)

        # Visual Overlay Checkboxes
        lbl_overlay = QLabel("Visual Overlay Elements:")
        tab_settings_layout.addWidget(lbl_overlay)

        self.chk_show_bbox = QCheckBox("Show Examinee Bounding Boxes")
        self.chk_show_bbox.setChecked(True)
        self.chk_show_bbox.toggled.connect(self._on_display_toggled)

        self.chk_show_skeleton = QCheckBox("Show Pose Skeletons & Keypoint Nodes")
        self.chk_show_skeleton.setChecked(True)
        self.chk_show_skeleton.toggled.connect(self._on_display_toggled)

        self.chk_show_labels = QCheckBox("Show Examinee ID & Behavior Confidence Badges")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.toggled.connect(self._on_display_toggled)

        self.chk_show_banners = QCheckBox("Show High-Alert Glowing Incidents Banner")
        self.chk_show_banners.setChecked(True)
        self.chk_show_banners.toggled.connect(self._on_display_toggled)

        tab_settings_layout.addWidget(self.chk_show_bbox)
        tab_settings_layout.addWidget(self.chk_show_skeleton)
        tab_settings_layout.addWidget(self.chk_show_labels)
        tab_settings_layout.addWidget(self.chk_show_banners)

        tab_settings_layout.addStretch()
        self.tabs.addTab(tab_settings, "⚙️ Detection Settings")

        # Tab 3: Evidence Snapshots Gallery
        tab_evidence = QWidget()
        tab_evidence_layout = QVBoxLayout(tab_evidence)
        tab_evidence_layout.setContentsMargins(10, 10, 10, 10)
        tab_evidence_layout.setSpacing(8)

        lbl_ev_info = QLabel("Captured Snapshots Directory:")
        lbl_ev_path = QLabel(str(SNAPSHOTS_DIR))
        lbl_ev_path.setStyleSheet("color: #94a3b8; font-size: 11px;")

        btn_open_folder = QPushButton("📁 Open Snapshots Folder in Explorer")
        btn_open_folder.clicked.connect(self._open_snapshots_folder)

        tab_evidence_layout.addWidget(lbl_ev_info)
        tab_evidence_layout.addWidget(lbl_ev_path)
        tab_evidence_layout.addWidget(btn_open_folder)
        tab_evidence_layout.addStretch()

        self.tabs.addTab(tab_evidence, "🖼️ Evidence Snapshots")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_widget)

        # Set splitter proportions (65% Left, 35% Right)
        splitter.setSizes([850, 450])
        main_layout.addWidget(splitter, stretch=1)

        # -------------------------------------------------------------
        # 3. Bottom Status Bar
        # -------------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select video source and press 'Start Proctoring'.")

    # =========================================================================
    # User Interaction Slots
    # =========================================================================

    def _set_placeholder_view(self):
        """Displays a clean placeholder graphic when surveillance is stopped."""
        pixmap = QPixmap(800, 450)
        pixmap.fill(QColor("#080c14"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw framing border
        painter.setPen(QPen(QColor("#1e293b"), 2, Qt.PenStyle.DashLine))
        painter.drawRect(20, 20, 760, 410)

        # Central text
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "SURVEILLANCE FEED INACTIVE\n\n1. Select Live Webcam or Browse Video Clip\n2. Click 'Start Proctoring' to begin real-time analysis")
        painter.end()

        self.video_label.setPixmap(pixmap)

    def _on_source_mode_changed(self, btn_id: int):
        """Switches between Webcam and Video Clip selector widgets."""
        if btn_id == 1:  # Webcam
            self.webcam_container.setVisible(True)
            self.clip_container.setVisible(False)
            self.status_bar.showMessage("Live webcam source selected.")
        else:  # Recorded Clip
            self.webcam_container.setVisible(False)
            self.clip_container.setVisible(True)
            self.status_bar.showMessage("Video recording clip source selected.")

    def refresh_cameras(self):
        """Scans for plugged-in webcams and populates the device dropdown."""
        self.combo_cameras.clear()
        self.status_bar.showMessage("Scanning video capture devices...")
        cams = scan_available_cameras()
        for idx in cams:
            self.combo_cameras.addItem(f"Camera Device {idx} (Built-in / USB)", idx)
        self.status_bar.showMessage(f"Camera scan complete. Detected {len(cams)} device(s).")

    def _browse_video_file(self):
        """Opens native file dialog to choose a recorded video clip."""
        initial_dir = str(RAW_VIDEOS_DIR) if RAW_VIDEOS_DIR.exists() else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Exam Video Recording",
            initial_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*.*)",
        )
        if file_path:
            self.selected_video_path = file_path
            self.lbl_selected_clip.setText(Path(file_path).name)
            self.lbl_selected_clip.setToolTip(file_path)
            self.status_bar.showMessage(f"Selected clip: {Path(file_path).name}")

    def _browse_yolo_weights(self):
        weights_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Pose Model Weights", str(Path.cwd()), "PyTorch Checkpoints (*.pt *.engine);;All Files (*.*)"
        )
        if weights_path:
            self.edit_yolo_weights.setText(weights_path)

    def _browse_lstm_weights(self):
        weights_path, _ = QFileDialog.getOpenFileName(
            self, "Select Trained LSTM Classifier Checkpoint", str(Path.cwd()), "PyTorch Checkpoint (*.pt);;All Files (*.*)"
        )
        if weights_path:
            self.edit_lstm_weights.setText(weights_path)

    def _on_threshold_changed(self, value: int):
        thresh = value / 100.0
        self.lbl_thresh_val.setText(f"{value}%")
        if self.worker and self.worker.isRunning():
            self.worker.update_settings(threshold=thresh)

    def _on_display_toggled(self):
        if self.worker and self.worker.isRunning():
            self.worker.update_settings(
                show_bbox=self.chk_show_bbox.isChecked(),
                show_skeleton=self.chk_show_skeleton.isChecked(),
                show_labels=self.chk_show_labels.isChecked(),
                show_banners=self.chk_show_banners.isChecked(),
            )

    def _toggle_audio_alert(self, checked: bool):
        self.audio_alert_enabled = checked

    # =========================================================================
    # Pipeline Execution Lifecycle
    # =========================================================================

    def start_proctoring(self):
        """Validates inputs, initializes worker thread, and starts live proctoring."""
        # Determine source
        if self.rb_webcam.isChecked():
            custom_input = self.edit_custom_cam.text().strip()
            if custom_input:
                source = custom_input
            else:
                cam_idx = self.combo_cameras.currentData()
                source = str(cam_idx if cam_idx is not None else 0)
        else:
            if not self.selected_video_path or not os.path.exists(self.selected_video_path):
                QMessageBox.warning(
                    self,
                    "No Video File Selected",
                    "Please click 'Browse Video File...' and choose a valid video clip (.mp4, .avi, etc.) first.",
                )
                return
            source = self.selected_video_path

        # Determine save video path if checked
        save_video_path = None
        if self.chk_record.isChecked():
            rec_filename = f"proctor_session_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            save_video_path = str(SNAPSHOTS_DIR / rec_filename)

        yolo_path = self.edit_yolo_weights.text().strip()
        lstm_path = Path(self.edit_lstm_weights.text().strip())
        model_mode = self.combo_engine.currentData()
        threshold = self.slider_thresh.value() / 100.0
        loop = self.chk_loop.isChecked()

        # Update UI Controls state
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(True)
        self.btn_snapshot.setEnabled(True)
        self.rb_webcam.setEnabled(False)
        self.rb_clip.setEnabled(False)
        self.combo_cameras.setEnabled(False)
        self.btn_browse_clip.setEnabled(False)
        self.badge_status.setText("STATUS: RUNNING")
        self.badge_status.setStyleSheet("background-color: #065f46; color: #34d399; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 11px;")

        # Spawn background worker thread
        self.worker = VideoInferenceWorker(
            source=source,
            yolo_weights=yolo_path,
            lstm_weights=lstm_path,
            model_type=model_mode,
            threshold=threshold,
            loop_video=loop,
            save_video_path=save_video_path,
            show_bbox=self.chk_show_bbox.isChecked(),
            show_skeleton=self.chk_show_skeleton.isChecked(),
            show_labels=self.chk_show_labels.isChecked(),
            show_banners=self.chk_show_banners.isChecked(),
        )

        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.stats_updated.connect(self._on_stats_updated)
        self.worker.alert_detected.connect(self._on_alert_detected)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.error_occurred.connect(self._on_error_occurred)
        self.worker.finished.connect(self._on_worker_finished)

        self.worker.start()

    def toggle_pause(self):
        if not self.worker:
            return
        if not self.is_paused:
            self.worker.pause()
            self.is_paused = True
            self.btn_pause.setText("▶  Resume")
            self.badge_status.setText("STATUS: PAUSED")
            self.badge_status.setStyleSheet("background-color: #78350f; color: #fbbf24; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 11px;")
        else:
            self.worker.resume()
            self.is_paused = False
            self.btn_pause.setText("⏸  Pause")
            self.badge_status.setText("STATUS: RUNNING")
            self.badge_status.setStyleSheet("background-color: #065f46; color: #34d399; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 11px;")

    def stop_proctoring(self):
        """Stops the proctoring stream and releases resources."""
        if self.worker and self.worker.isRunning():
            self.status_bar.showMessage("Stopping proctoring stream...")
            self.worker.stop()
            self.worker.wait(1500)
        self._on_worker_finished()

    def capture_snapshot(self):
        """Saves current frame as timestamped evidence snapshot."""
        if not self.worker or not self.worker.isRunning():
            return
        filename = f"evidence_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = str(SNAPSHOTS_DIR / filename)
        self.worker.request_snapshot(filepath)

    def _open_snapshots_folder(self):
        """Opens evidence folder in Windows Explorer."""
        if sys.platform == "win32":
            os.startfile(str(SNAPSHOTS_DIR))
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(SNAPSHOTS_DIR)])

    # =========================================================================
    # Worker Thread Event Handlers
    # =========================================================================

    @pyqtSlot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray):
        """Renders incoming BGR frame smoothly to the video label widget."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pix = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pix)

    @pyqtSlot(float, int, int, str)
    def _on_stats_updated(self, fps: float, active_students: int, total_alerts: int, threat: str):
        """Updates top KPI cards with latest surveillance metrics."""
        self.val_fps.setText(f"{fps:.1f}")
        self.val_active_students.setText(str(active_students))
        self.val_total_alerts.setText(str(total_alerts))

        self.val_threat.setText(threat)
        if threat == "CRITICAL":
            self.val_threat.setStyleSheet("font-size: 20px; font-weight: 800; color: #ef4444;")
        elif threat == "ALERT":
            self.val_threat.setStyleSheet("font-size: 20px; font-weight: 800; color: #f59e0b;")
        else:
            self.val_threat.setStyleSheet("font-size: 20px; font-weight: 800; color: #10b981;")

    @pyqtSlot(dict)
    def _on_alert_detected(self, alert: dict):
        """Appends incident row to live alerts table and triggers alert chime."""
        row_pos = self.table_alerts.rowCount()
        self.table_alerts.insertRow(row_pos)

        item_time = QTableWidgetItem(alert["timestamp"])
        item_id = QTableWidgetItem(f"Student #{alert['track_id']}")
        item_behavior = QTableWidgetItem(alert["behavior"].upper().replace("_", " "))
        item_conf = QTableWidgetItem(f"{alert['confidence']*100:.1f}%")
        item_severity = QTableWidgetItem(alert["severity"])

        # Format severity color badges
        sev = alert["severity"]
        if sev == "RED":
            item_severity.setForeground(QColor("#f87171"))
            item_behavior.setForeground(QColor("#f87171"))
            if self.audio_alert_enabled:
                play_chime()
        elif sev == "ORANGE":
            item_severity.setForeground(QColor("#fbbf24"))
            item_behavior.setForeground(QColor("#fbbf24"))
        elif sev == "YELLOW":
            item_severity.setForeground(QColor("#fde047"))
        else:
            item_severity.setForeground(QColor("#94a3b8"))

        self.table_alerts.setItem(row_pos, 0, item_time)
        self.table_alerts.setItem(row_pos, 1, item_id)
        self.table_alerts.setItem(row_pos, 2, item_behavior)
        self.table_alerts.setItem(row_pos, 3, item_conf)
        self.table_alerts.setItem(row_pos, 4, item_severity)

        if self.chk_auto_scroll.isChecked():
            self.table_alerts.scrollToBottom()

    @pyqtSlot(str)
    def _on_status_changed(self, message: str):
        self.status_bar.showMessage(message)

    @pyqtSlot(str)
    def _on_error_occurred(self, err_message: str):
        self.status_bar.showMessage(f"Error: {err_message}")
        QMessageBox.critical(self, "Proctoring Pipeline Error", err_message)

    @pyqtSlot()
    def _on_worker_finished(self):
        """Resets controls when stream ends or is stopped."""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.rb_webcam.setEnabled(True)
        self.rb_clip.setEnabled(True)
        self.combo_cameras.setEnabled(True)
        self.btn_browse_clip.setEnabled(True)
        self.is_paused = False

        self.badge_status.setText("STATUS: IDLE")
        self.badge_status.setStyleSheet("background-color: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 11px;")
        self._set_placeholder_view()
        self.status_bar.showMessage("Surveillance stream terminated.")

    def _export_alerts_csv(self):
        """Exports the active table of alerts to a CSV report."""
        if self.table_alerts.rowCount() == 0:
            QMessageBox.information(self, "No Alerts", "There are no recorded alerts to export yet.")
            return

        default_name = str(REPORTS_DIR / f"exam_alerts_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        export_path, _ = QFileDialog.getSaveFileName(
            self, "Export Proctoring Audit Log", default_name, "CSV Files (*.csv);;All Files (*.*)"
        )
        if not export_path:
            return

        import csv
        try:
            with open(export_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Examinee ID", "Suspicious Behavior", "Confidence", "Severity Level"])
                for row in range(self.table_alerts.rowCount()):
                    row_data = [self.table_alerts.item(row, col).text() if self.table_alerts.item(row, col) else "" for col in range(5)]
                    writer.writerow(row_data)
            QMessageBox.information(self, "Export Successful", f"Audit report successfully exported to:\n{export_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save CSV file: {e}")

    def _clear_alerts(self):
        self.table_alerts.setRowCount(0)
        self.val_total_alerts.setText("0")
        self.status_bar.showMessage("Alert logs cleared.")

    def closeEvent(self, event):
        """Ensures worker thread is gracefully terminated when window closes."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ExamProctorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
