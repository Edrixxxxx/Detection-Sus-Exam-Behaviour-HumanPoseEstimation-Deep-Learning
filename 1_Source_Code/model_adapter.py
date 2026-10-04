"""
Anti-Cheating Model Adapter Framework
======================================
Provides a modular, extensible, and pluggable architecture for adapting different
behavior classification models (Deep Learning LSTM, Heuristic Rules, Hybrid Ensembles,
or custom ONNX/PyTorch models) into the real-time proctoring surveillance system.

Complies with the 28-feature extraction and Platt-scaling calibration specifications
outlined in the academic thesis documentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Any
import numpy as np
import torch
import torch.nn as nn

try:
    from config import (
        LSTM_MODEL_PATH,
        CLASSES,
        CLASS_TO_IDX,
        IDX_TO_CLASS,
        NUM_CLASSES,
        FEATURE_DIM,
        SEQUENCE_LENGTH,
        SUSPICIOUS_CONFIDENCE_THRESHOLD,
    )
    from model import ExamBehaviorLSTM
except ImportError:
    from .config import (
        LSTM_MODEL_PATH,
        CLASSES,
        CLASS_TO_IDX,
        IDX_TO_CLASS,
        NUM_CLASSES,
        FEATURE_DIM,
        SEQUENCE_LENGTH,
        SUSPICIOUS_CONFIDENCE_THRESHOLD,
    )
    from .model import ExamBehaviorLSTM


@dataclass
class ModelPrediction:
    """Standardized prediction output payload returned by any Anti-Cheating Model."""
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    calibrated_probabilities: Dict[str, float]
    severity_level: str  # "NONE", "YELLOW", "ORANGE", "RED"
    heuristics_info: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    model_name: str = ""

    def is_suspicious(self) -> bool:
        return self.predicted_class != "normal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "class_index": self.class_index,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "calibrated_probabilities": {k: round(float(v), 4) for k, v in self.calibrated_probabilities.items()},
            "severity_level": self.severity_level,
            "heuristics_info": self.heuristics_info,
            "explanation": self.explanation,
            "model_name": self.model_name,
        }


class BaseAntiCheatingModel(ABC):
    """
    Abstract Base Class for adapting any anti-cheating model into the proctoring pipeline.
    To adapt a new model (e.g. Vision Transformer, TCN, XGBoost, or Custom Neural Network),
    inherit from this class and implement `predict_sequence` and `get_metadata`.
    """

    def __init__(self, name: str, version: str = "1.0", classes: Optional[List[str]] = None):
        self.name = name
        self.version = version
        self.classes = classes or CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Platt Scaling Parameters per class: P_calibrated = 1 / (1 + exp(A * p + B))
        # Fitted to calibrate raw neural and heuristic scores to genuine posteriors
        self.platt_params: Dict[str, Tuple[float, float]] = {
            "normal": (-2.5, 0.5),
            "hand_signal": (-3.0, 1.2),
            "passing_of_notes": (-3.2, 1.3),
            "side_glancing": (-3.0, 1.1),
            "use_of_unauthorized_object": (-3.5, 1.5),
        }

    @abstractmethod
    def predict_sequence(
        self,
        sequence: np.ndarray,
        raw_keypoints: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelPrediction:
        """
        Processes a sequence of normalized pose keypoints and optional spatial context.
        
        Args:
            sequence: np.ndarray of shape (T, FEATURE_DIM) where T is sequence length (e.g. 30)
            raw_keypoints: Optional (17, 2) array of unnormalized COCO keypoints for current frame
            bbox: Optional [x1, y1, x2, y2] bounding box
            context: Additional contextual signals (e.g., neighbor tracks, sustained duration)
            
        Returns:
            ModelPrediction: Standardized prediction object
        """
        pass

    def apply_platt_scaling(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Applies Platt logistic transformation to calibrate raw classifier probabilities."""
        calibrated = {}
        for cls_name, p in probs.items():
            a, b = self.platt_params.get(cls_name, (-3.0, 1.0))
            # Platt scaling sigmoid transformation
            cal_p = 1.0 / (1.0 + np.exp(a * p + b))
            calibrated[cls_name] = float(np.clip(cal_p, 0.0, 1.0))
            
        # Re-normalize to sum to 1.0
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {k: v / total for k, v in calibrated.items()}
        return calibrated

    def calculate_severity(
        self,
        predicted_class: str,
        confidence: float,
        duration_frames: int = 1,
    ) -> str:
        """
        Assigns Graduated Alert Severity (Yellow, Orange, Red) based on class and confidence.
        - Yellow (Low): Single anomaly or 0.40 <= conf < 0.60
        - Orange (Medium): Sustained anomaly or 0.60 <= conf < 0.80
        - Red (High): Critical violation or conf >= 0.80 sustained
        """
        if predicted_class == "normal":
            return "NONE"

        if confidence >= 0.80 or duration_frames >= 20:
            return "RED"
        elif confidence >= 0.60 or duration_frames >= 8:
            return "ORANGE"
        elif confidence >= 0.40:
            return "YELLOW"
        return "NONE"

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata, configuration, and capabilities."""
        return {
            "name": self.name,
            "version": self.version,
            "classes": self.classes,
            "num_classes": self.num_classes,
        }


class LSTMModelAdapter(BaseAntiCheatingModel):
    """
    Adapter for PyTorch ExamBehaviorLSTM neural network.
    Loads checkpoint weights (best_lstm_model.pt) and executes recurrent temporal classification.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(name="PyTorch LSTM Classifier", version="2.0")
        self.weights_path = Path(weights_path) if weights_path else LSTM_MODEL_PATH
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[ExamBehaviorLSTM] = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Loads or reloads model weights from checkpoint."""
        if not self.weights_path.exists():
            print(f"[LSTMModelAdapter] Checkpoint not found at: {self.weights_path}")
            self.is_loaded = False
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            input_dim = checkpoint.get("input_dim", FEATURE_DIM)
            hidden_dim = checkpoint.get("hidden_dim", 64)
            num_layers = checkpoint.get("num_layers", 2)
            num_classes = checkpoint.get("num_classes", len(self.classes))
            bidirectional = checkpoint.get("bidirectional", True)

            self.model = ExamBehaviorLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                bidirectional=bidirectional,
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_loaded = True
            print(f"[LSTMModelAdapter] Loaded checkpoint from: {self.weights_path} ({self.device})")
        except Exception as e:
            print(f"[LSTMModelAdapter] Error loading checkpoint: {e}")
            self.is_loaded = False

    def reload_weights(self, new_weights_path: Path) -> bool:
        """Dynamically hot-swaps model weights at runtime."""
        self.weights_path = Path(new_weights_path)
        self._load_model()
        return self.is_loaded

    def predict_sequence(
        self,
        sequence: np.ndarray,
        raw_keypoints: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelPrediction:
        if not self.is_loaded or self.model is None:
            # Fallback if weights not loaded
            return ModelPrediction(
                predicted_class="normal",
                class_index=0,
                confidence=0.5,
                probabilities={cls: 1.0 / len(self.classes) for cls in self.classes},
                calibrated_probabilities={cls: 1.0 / len(self.classes) for cls in self.classes},
                severity_level="NONE",
                explanation="Model checkpoint not loaded; returning neutral prediction.",
                model_name=self.name,
            )

        # Prepare tensor
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs_tensor = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Build class probability mapping
        raw_probs = {self.classes[i]: float(probs_tensor[i]) for i in range(len(self.classes))}
        calibrated_probs = self.apply_platt_scaling(raw_probs)

        # Predicted class and confidence
        pred_idx = int(np.argmax(probs_tensor))
        pred_class = self.classes[pred_idx]
        confidence = float(probs_tensor[pred_idx])

        duration = (context or {}).get("alert_duration_frames", 1)
        severity = self.calculate_severity(pred_class, confidence, duration)

        explanation = f"Neural LSTM classified sequence as '{pred_class}' with {confidence*100:.1f}% confidence."

        return ModelPrediction(
            predicted_class=pred_class,
            class_index=pred_idx,
            confidence=confidence,
            probabilities=raw_probs,
            calibrated_probabilities=calibrated_probs,
            severity_level=severity,
            explanation=explanation,
            model_name=self.name,
        )

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "type": "Deep Learning (PyTorch Recurrent LSTM)",
            "weights_path": str(self.weights_path),
            "is_loaded": self.is_loaded,
            "device": str(self.device),
        })
        return meta


class HeuristicRuleModelAdapter(BaseAntiCheatingModel):
    """
    Deterministic Heuristic Anti-Cheating Classifier.
    Evaluates geometric posture rules (Head Yaw, Pitch, Wrist Reach, Shoulder Alignment)
    from the 28-feature specifications without requiring GPU or neural training weights.
    """

    def __init__(self):
        super().__init__(name="Deterministic Heuristic Engine", version="1.5")

    def _extract_geometric_features(
        self,
        raw_kpts: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Computes geometric angles and normalized distances from 17 COCO keypoints:
        0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear,
        5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow,
        9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip
        """
        feats: Dict[str, float] = {
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "shoulder_width": 1.0,
            "l_wrist_reach": 0.0,
            "r_wrist_reach": 0.0,
            "hands_elevated": 0.0,
            "head_down_ratio": 0.0,
        }

        if raw_kpts is None or len(raw_kpts) < 17:
            return feats

        nose = raw_kpts[0]
        l_eye, r_eye = raw_kpts[1], raw_kpts[2]
        l_ear, r_ear = raw_kpts[3], raw_kpts[4]
        l_sh, r_sh = raw_kpts[5], raw_kpts[6]
        l_wr, r_wr = raw_kpts[9], raw_kpts[10]

        # Inter-shoulder width for scale normalization
        sh_dist = np.linalg.norm(l_sh - r_sh)
        sh_width = max(sh_dist, 20.0)
        feats["shoulder_width"] = float(sh_width)

        # 1. Head Yaw (lateral turning peek):
        # Ratio of (Nose - L_Ear) vs (R_Ear - Nose)
        if l_ear[0] > 0 and r_ear[0] > 0 and nose[0] > 0:
            d_left = abs(nose[0] - l_ear[0])
            d_right = abs(nose[0] - r_ear[0])
            yaw_ratio = (d_right - d_left) / (d_right + d_left + 1e-5)
            # Map ratio to approximate degrees [-60, +60]
            feats["head_yaw"] = float(yaw_ratio * 60.0)
        elif l_eye[0] > 0 and r_eye[0] > 0 and nose[0] > 0:
            mid_eyes = (l_eye[0] + r_eye[0]) / 2.0
            feats["head_yaw"] = float((nose[0] - mid_eyes) / (sh_width + 1e-5) * 90.0)

        # 2. Head Pitch (peeking down into lap or concealed phone):
        if nose[1] > 0 and l_sh[1] > 0 and r_sh[1] > 0:
            mid_sh_y = (l_sh[1] + r_sh[1]) / 2.0
            # Distance from nose downward towards shoulder line
            pitch_disp = (nose[1] - mid_sh_y) / (sh_width + 1e-5)
            feats["head_down_ratio"] = float(pitch_disp)
            feats["head_pitch"] = float(pitch_disp * 45.0)  # Positive = deeply bowed down

        # 3. Wrist Lateral Reach (passing notes across desks):
        if l_wr[0] > 0 and l_sh[0] > 0:
            feats["l_wrist_reach"] = float(np.linalg.norm(l_wr - l_sh) / sh_width)
        if r_wr[0] > 0 and r_sh[0] > 0:
            feats["r_wrist_reach"] = float(np.linalg.norm(r_wr - r_sh) / sh_width)

        # 4. Elevated Hand Signals (wrist raised above shoulder line):
        mid_sh_y = (l_sh[1] + r_sh[1]) / 2.0 if (l_sh[1] > 0 and r_sh[1] > 0) else 1000.0
        elevated_count = 0
        if l_wr[1] > 0 and l_wr[1] < (mid_sh_y - 15):
            elevated_count += 1
        if r_wr[1] > 0 and r_wr[1] < (mid_sh_y - 15):
            elevated_count += 1
        feats["hands_elevated"] = float(elevated_count)

        return feats

    def predict_sequence(
        self,
        sequence: np.ndarray,
        raw_keypoints: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelPrediction:
        # Default probabilities
        probs = {cls: 0.05 for cls in self.classes}
        probs["normal"] = 0.80

        heuristics = self._extract_geometric_features(raw_keypoints, bbox)
        triggered_rules = []

        yaw = abs(heuristics["head_yaw"])
        pitch_drop = heuristics["head_down_ratio"]
        max_reach = max(heuristics["l_wrist_reach"], heuristics["r_wrist_reach"])
        elevated = heuristics["hands_elevated"]

        # Rule 1: Side Glancing (lateral head turn > 25 degrees)
        if yaw > 25.0:
            severity_factor = min(1.0, (yaw - 25.0) / 25.0)
            score = 0.65 + 0.30 * severity_factor
            probs["side_glancing"] = max(probs["side_glancing"], score)
            probs["normal"] = max(0.05, 1.0 - score)
            triggered_rules.append(f"Head Yaw angle ({yaw:.1f}deg > 25deg) peeking sideways")

        # Rule 2: Use of Unauthorized Object (head bowed deeply down into lap)
        if pitch_drop > 0.15:
            severity_factor = min(1.0, (pitch_drop - 0.15) / 0.30)
            score = 0.65 + 0.30 * severity_factor
            probs["use_of_unauthorized_object"] = max(probs["use_of_unauthorized_object"], score)
            probs["normal"] = max(0.05, 1.0 - score)
            triggered_rules.append(f"Head pitched downward (ratio {pitch_drop:.2f} > 0.15) towards lap/desk")

        # Rule 3: Passing of Notes (lateral arm extension across desk)
        if max_reach > 1.30:
            severity_factor = min(1.0, (max_reach - 1.30) / 0.50)
            score = 0.65 + 0.28 * severity_factor
            probs["passing_of_notes"] = max(probs["passing_of_notes"], score)
            probs["normal"] = max(0.05, 1.0 - score)
            triggered_rules.append(f"Wrist lateral reach ({max_reach:.2f}x shoulder width > 1.30x)")

        # Rule 4: Hand Signaling (hand raised above shoulder height)
        if elevated > 0:
            score = 0.70 + 0.20 * min(elevated, 2)
            probs["hand_signal"] = max(probs["hand_signal"], score)
            probs["normal"] = max(0.05, 1.0 - score)
            triggered_rules.append(f"Elevated wrist gesturing detected above shoulder level")

        # Normalize probabilities
        tot = sum(probs.values())
        norm_probs = {k: v / tot for k, v in probs.items()}
        calibrated_probs = self.apply_platt_scaling(norm_probs)

        # Selected class
        pred_class = max(norm_probs.keys(), key=lambda k: norm_probs[k])
        pred_idx = self.class_to_idx[pred_class]
        confidence = norm_probs[pred_class]

        duration = (context or {}).get("alert_duration_frames", 1)
        severity = self.calculate_severity(pred_class, confidence, duration)

        if triggered_rules:
            explanation = "Heuristic triggers: " + "; ".join(triggered_rules)
        else:
            explanation = "Examinee posture remains within normal bounds."

        return ModelPrediction(
            predicted_class=pred_class,
            class_index=pred_idx,
            confidence=confidence,
            probabilities=norm_probs,
            calibrated_probabilities=calibrated_probs,
            severity_level=severity,
            heuristics_info=heuristics,
            explanation=explanation,
            model_name=self.name,
        )

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "type": "Geometric-Temporal Heuristic Rules",
            "thresholds": {
                "head_yaw_deg": 25.0,
                "head_pitch_drop_ratio": 0.15,
                "wrist_reach_factor": 1.30,
            }
        })
        return meta


class HybridEnsembleModelAdapter(BaseAntiCheatingModel):
    """
    Hybrid Ensemble Model: Fuses Deep Learning (LSTM) predictions with
    Deterministic Geometric Heuristics and Platt-calibrated posterior probabilities.
    Reduces false alarms by requiring geometric plausibility for flagged gestures.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        lstm_weight: float = 0.60,
        heuristic_weight: float = 0.40,
    ):
        super().__init__(name="Hybrid Neural-Heuristic Ensemble", version="2.5")
        self.lstm_adapter = LSTMModelAdapter(weights_path=weights_path, device=device)
        self.heuristic_adapter = HeuristicRuleModelAdapter()
        self.lstm_weight = lstm_weight
        self.heuristic_weight = heuristic_weight

    def predict_sequence(
        self,
        sequence: np.ndarray,
        raw_keypoints: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelPrediction:
        # 1. Run LSTM inference
        lstm_pred = self.lstm_adapter.predict_sequence(sequence, raw_keypoints, bbox, context)
        # 2. Run Heuristic rule evaluation
        heur_pred = self.heuristic_adapter.predict_sequence(sequence, raw_keypoints, bbox, context)

        # 3. Fuse probabilities
        fused_probs = {}
        for cls in self.classes:
            p_lstm = lstm_pred.probabilities.get(cls, 0.0)
            p_heur = heur_pred.probabilities.get(cls, 0.0)
            fused_probs[cls] = (self.lstm_weight * p_lstm) + (self.heuristic_weight * p_heur)

        # Normalize
        tot = sum(fused_probs.values())
        if tot > 0:
            fused_probs = {k: v / tot for k, v in fused_probs.items()}

        # Platt scaling on ensemble
        calibrated_probs = self.apply_platt_scaling(fused_probs)

        # Winner
        pred_class = max(fused_probs.keys(), key=lambda k: fused_probs[k])
        pred_idx = self.class_to_idx[pred_class]
        confidence = fused_probs[pred_class]

        duration = (context or {}).get("alert_duration_frames", 1)
        severity = self.calculate_severity(pred_class, confidence, duration)

        # Combined explanation
        explanation = (
            f"Ensemble (LSTM: {lstm_pred.predicted_class} [{lstm_pred.confidence*100:.0f}%], "
            f"Heuristic: {heur_pred.predicted_class} [{heur_pred.confidence*100:.0f}%]) "
            f"-> {pred_class} ({confidence*100:.1f}%)"
        )

        return ModelPrediction(
            predicted_class=pred_class,
            class_index=pred_idx,
            confidence=confidence,
            probabilities=fused_probs,
            calibrated_probabilities=calibrated_probs,
            severity_level=severity,
            heuristics_info=heur_pred.heuristics_info,
            explanation=explanation,
            model_name=self.name,
        )

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "type": "Hybrid Ensemble (LSTM + Heuristics)",
            "weights": {
                "lstm_weight": self.lstm_weight,
                "heuristic_weight": self.heuristic_weight,
            },
            "lstm_info": self.lstm_adapter.get_metadata(),
            "heuristic_info": self.heuristic_adapter.get_metadata(),
        })
        return meta


class MockSimulationModelAdapter(BaseAntiCheatingModel):
    """
    Simulation Model Adapter for testing, demonstration, and CI/CD pipelines.
    Generates realistic examinee behaviors with configurable periodic anomalies.
    """

    def __init__(self):
        super().__init__(name="Simulation Model Adapter", version="1.0")
        self.step = 0

    def predict_sequence(
        self,
        sequence: np.ndarray,
        raw_keypoints: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelPrediction:
        self.step += 1
        probs = {cls: 0.05 for cls in self.classes}

        # Cycle behaviors every 60 steps for demonstration
        cycle = (self.step // 45) % 6
        if cycle == 0:
            pred_class = "normal"
            probs["normal"] = 0.92
        elif cycle == 1:
            pred_class = "side_glancing"
            probs["side_glancing"] = 0.88
            probs["normal"] = 0.08
        elif cycle == 2:
            pred_class = "normal"
            probs["normal"] = 0.95
        elif cycle == 3:
            pred_class = "use_of_unauthorized_object"
            probs["use_of_unauthorized_object"] = 0.85
            probs["normal"] = 0.10
        elif cycle == 4:
            pred_class = "passing_of_notes"
            probs["passing_of_notes"] = 0.82
            probs["normal"] = 0.12
        else:
            pred_class = "hand_signal"
            probs["hand_signal"] = 0.84
            probs["normal"] = 0.11

        tot = sum(probs.values())
        norm_probs = {k: v / tot for k, v in probs.items()}
        calibrated_probs = self.apply_platt_scaling(norm_probs)
        conf = norm_probs[pred_class]
        severity = self.calculate_severity(pred_class, conf, duration_frames=5)

        return ModelPrediction(
            predicted_class=pred_class,
            class_index=self.class_to_idx[pred_class],
            confidence=conf,
            probabilities=norm_probs,
            calibrated_probabilities=calibrated_probs,
            severity_level=severity,
            explanation=f"Simulated test pattern: {pred_class} active.",
            model_name=self.name,
        )

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({"type": "Mock Simulation Stream"})
        return meta


class ModelRegistry:
    """
    Global Model Registry allowing researchers and developers to register,
    retrieve, and hot-swap anti-cheating models dynamically in the system.
    """

    _registry: Dict[str, Type[BaseAntiCheatingModel]] = {}
    _instances: Dict[str, BaseAntiCheatingModel] = {}

    @classmethod
    def register(cls, model_id: str, adapter_class: Type[BaseAntiCheatingModel]):
        """Registers a model adapter class."""
        cls._registry[model_id.lower()] = adapter_class
        print(f"[ModelRegistry] Registered model: '{model_id}' -> {adapter_class.__name__}")

    @classmethod
    def get_model(cls, model_id: str, **kwargs) -> BaseAntiCheatingModel:
        """Instantiates or retrieves a cached instance of the requested model."""
        key = model_id.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Model '{model_id}' not found. Available models: {available}")

        # Instantiate
        adapter_class = cls._registry[key]
        return adapter_class(**kwargs)

    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        """Lists all registered models with their metadata."""
        model_list = []
        for model_id, adapter_class in cls._registry.items():
            # Create a lightweight metadata descriptor
            doc = adapter_class.__doc__ or ""
            model_list.append({
                "id": model_id,
                "class_name": adapter_class.__name__,
                "description": doc.strip().split("\n")[0],
            })
        return model_list


# Auto-register default built-in model adapters
ModelRegistry.register("lstm", LSTMModelAdapter)
ModelRegistry.register("heuristic", HeuristicRuleModelAdapter)
ModelRegistry.register("hybrid", HybridEnsembleModelAdapter)
ModelRegistry.register("simulation", MockSimulationModelAdapter)
