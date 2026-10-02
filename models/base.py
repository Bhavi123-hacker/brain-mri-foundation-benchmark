"""
Abstract Base Model Interface for all Benchmark Architectures.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class BaseBenchModel(nn.Module, ABC):
    """
    Standard interface for all benchmark models (Baselines and Foundation Adapters).
    Guarantees consistent API across forward pass, feature extraction, parameter counts,
    and compute profiling.
    """
    def __init__(self, model_name: str, task_type: str, output_dim: int = 1):
        super().__init__()
        self.model_name = model_name
        self.task_type = task_type
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input 3D volume tensor of shape (B, C, D, H, W).
        Returns:
            Output logits or continuous scalar tensor of shape (B, output_dim).
        """
        pass

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation before final task head.
        Args:
            x: Input 3D volume tensor (B, C, D, H, W).
        Returns:
            Feature embedding tensor of shape (B, D_latent).
        """
        pass

    def get_parameter_count(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
        }
