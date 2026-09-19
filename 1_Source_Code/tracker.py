import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_WEIGHTS,
    TRACKER_CONFIG,
    SEQUENCE_LENGTH,
    FEATURE_DIM,
)
from dataset import normalize_keypoints


class TrackedPerson:
    """Represents a single examinee tracked across frames."""

    def __init__(self, track_id: int, max_history: int = SEQUENCE_LENGTH):
        self.track_id = track_id
        self.max_history = max_history
        self.buffer = deque(maxlen=max_history)
        self.raw_keypoints_buffer = deque(maxlen=max_history)
        self.last_bbox: Optional[np.ndarray] = None
        self.last_seen_frame: int = 0
        self.current_prediction: str = "normal"
        self.current_confidence: float = 0.0
        self.alert_counter: int = 0

    def update(self, norm_kpts: np.ndarray, raw_kpts: np.ndarray, bbox: np.ndarray, frame_idx: int):
        self.buffer.append(norm_kpts)
        self.raw_keypoints_buffer.append(raw_kpts)
        self.last_bbox = bbox
        self.last_seen_frame = frame_idx

    def is_sequence_ready(self) -> bool:
        return len(self.buffer) == self.max_history

    def get_sequence_array(self) -> Optional[np.ndarray]:
        """Returns array of shape (sequence_length, feature_dim) if buffer is full."""
        if not self.is_sequence_ready():
            return None
        return np.array(self.buffer, dtype=np.float32)


class PoseTrackerManager:
    """
    Manages YOLO26s-pose and ByteTrack tracking.
    Maintains per-student sliding window buffers for sequential LSTM classification.
    """

    def __init__(
        self,
        weights_path: str = YOLO_WEIGHTS,
        tracker_config: str = TRACKER_CONFIG,
        sequence_length: int = SEQUENCE_LENGTH,
        max_stale_frames: int = 90,
    ):
        print(f"-> Loading YOLO Pose Model from: {weights_path}")
        self.model = YOLO(weights_path)
        self.tracker_config = tracker_config
        self.sequence_length = sequence_length
        self.max_stale_frames = max_stale_frames

        # Map track_id -> TrackedPerson
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.frame_idx = 0

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO pose detection and ByteTrack tracking on a single frame.

        Returns a list of dicts for detected/tracked people in the frame:
        [
            {
                "track_id": int,
                "bbox": np.ndarray [x1, y1, x2, y2],
                "raw_keypoints": np.ndarray (17, 2),
                "norm_keypoints": np.ndarray (34,),
                "sequence_ready": bool,
                "sequence": Optional[np.ndarray] (seq_len, 34),
                "tracked_person": TrackedPerson
            }, ...
        ]
        """
        self.frame_idx += 1
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        frame_detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            self._prune_stale_tracks()
            return frame_detections

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is None or len(boxes) == 0 or keypoints is None or len(keypoints) == 0:
            self._prune_stale_tracks()
            return frame_detections

        # Check track IDs
        if boxes.id is None:
            # Detections without assigned track IDs yet
            return frame_detections

        track_ids = boxes.id.int().cpu().numpy()
        bboxes = boxes.xyxy.cpu().numpy()
        kpts_xy = keypoints.xy.cpu().numpy()  # (N, 17, 2)

        for i, tid in enumerate(track_ids):
            bbox = bboxes[i]
            kpts = kpts_xy[i]

            if tid not in self.tracked_people:
                self.tracked_people[tid] = TrackedPerson(tid, self.sequence_length)

            person = self.tracked_people[tid]
            norm_kpts = normalize_keypoints(kpts, bbox)

            person.update(norm_kpts, kpts, bbox, self.frame_idx)

            frame_detections.append({
                "track_id": int(tid),
                "bbox": bbox,
                "raw_keypoints": kpts,
                "norm_keypoints": norm_kpts,
                "sequence_ready": person.is_sequence_ready(),
                "sequence": person.get_sequence_array(),
                "tracked_person": person,
            })

        self._prune_stale_tracks()
        return frame_detections

    def _prune_stale_tracks(self):
        """Removes tracks that have disappeared from the video feed."""
        stale_ids = [
            tid for tid, person in self.tracked_people.items()
            if (self.frame_idx - person.last_seen_frame) > self.max_stale_frames
        ]
        for tid in stale_ids:
            del self.tracked_people[tid]

    def reset(self):
        """Resets all tracking histories (e.g. when starting a new video)."""
        self.tracked_people.clear()
        self.frame_idx = 0
