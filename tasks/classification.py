"""
Binary Classification Task Handler (AD vs CN and MCI vs CN).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class ClassificationTask:
    def __init__(self, task_name: str = "AD_vs_CN", pos_weight: float = 1.0):
        self.task_name = task_name
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]) if pos_weight != 1.0 else None)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss from logits and targets."""
        return self.loss_fn(logits, targets)

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to get class 1 probability."""
        return torch.sigmoid(logits)

    def get_predictions(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions 0 or 1."""
        probs = self.get_probabilities(logits)
        return (probs >= threshold).float()
