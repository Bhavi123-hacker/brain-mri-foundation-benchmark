"""
Continuous Regression Task Handler (Brain-Age Prediction).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class RegressionTask:
    def __init__(self, task_name: str = "Brain_Age", loss_type: str = "MSELoss"):
        self.task_name = task_name
        if loss_type == "L1Loss":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "SmoothL1Loss":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute regression loss."""
        return self.loss_fn(predictions, targets)

    def get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Get continuous scalar predictions."""
        return outputs
