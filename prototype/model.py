import torch
import torch.nn as nn
from typing import Tuple

from config import (
    FEATURE_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_BIDIRECTIONAL,
    LSTM_DROPOUT,
    NUM_CLASSES,
)


class ExamBehaviorLSTM(nn.Module):
    """
    Recurrent neural network classifier for examination behavior proctoring.
    Classifies normalized sequences of 17-keypoint human poses into
    'normal' vs 'suspicious' classes.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        bidirectional: bool = LSTM_BIDIRECTIONAL,
        dropout: float = LSTM_DROPOUT,
    ):
        super(ExamBehaviorLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization across features
        self.input_norm = nn.LayerNorm(input_dim)

        # LSTM Temporal Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output feature dimension from LSTM
        lstm_out_dim = hidden_dim * self.num_directions

        # Temporal representation: combine last hidden step + temporal average pooling
        classifier_input_dim = lstm_out_dim * 2

        # Classification MLP Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
        """
        # Apply layer normalization across input feature dimension
        x = self.input_norm(x)

        # LSTM pass: out shape -> (batch_size, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)

        # 1. Final time step output
        last_step = lstm_out[:, -1, :]

        # 2. Mean temporal pooling over all sequence frames
        mean_pooled = torch.mean(lstm_out, dim=1)

        # Concatenate temporal features for rich motion representation
        combined = torch.cat([last_step, mean_pooled], dim=-1)

        # Classify
        logits = self.classifier(combined)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates class probabilities via Softmax.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            torch.Tensor: Probabilities of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def predict_single(self, sequence_tensor: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        Helper for single-sequence real-time inference.
        Args:
            sequence_tensor (torch.Tensor): (seq_len, input_dim) or (1, seq_len, input_dim)
        Returns:
            Tuple[int, float, torch.Tensor]:
                - predicted_class_idx (int)
                - confidence (float)
                - probabilities (torch.Tensor of shape (num_classes,))
        """
        if sequence_tensor.dim() == 2:
            sequence_tensor = sequence_tensor.unsqueeze(0)

        probs = self.predict_proba(sequence_tensor).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)
        return int(pred_idx.item()), float(conf.item()), probs
