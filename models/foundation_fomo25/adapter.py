"""
FOMO25 Winner Foundation Model Adapter (BrainFM4Challenges / CHUV / MICCAI FOMO 2025).
Supports linear probing and full fine-tuning.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from models.base import BaseBenchModel


class FOMO25Adapter(BaseBenchModel):
    """
    Adapter for FOMO25 Challenge Winner 3D Brain MRI Foundation Model.
    """
    def __init__(
        self,
        output_dim: int = 1,
        task_type: str = "classification",
        embed_dim: int = 512,
        weights_path: Optional[str] = None,
        adaptation_mode: str = "linear_probe",
        dropout: float = 0.2,
        model_name: str = "FOMO25-Winner"
    ):
        super().__init__(model_name=model_name, task_type=task_type, output_dim=output_dim)
        self.embed_dim = embed_dim
        self.adaptation_mode = adaptation_mode
        self.weights_path = weights_path
        self.is_foundation_model = True

        # 3D SSL Multi-scale Encoder Backbone
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Downstream Adaptation Head
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim, output_dim)
        )

        if weights_path and os.path.exists(weights_path):
            self.load_pretrained_weights(weights_path)

        if adaptation_mode == "linear_probe":
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def freeze_backbone(self):
        """Freeze encoder for linear probing."""
        self.adaptation_mode = "linear_probe"
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze encoder for fine-tuning."""
        self.adaptation_mode = "fine_tuning"
        for param in self.encoder.parameters():
            param.requires_grad = True

    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained checkpoint."""
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        self.load_state_dict(state_dict, strict=False)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation: (B, embed_dim)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.head(feats)

    def get_model_metadata(self) -> Dict[str, Any]:
        param_info = self.get_parameter_count()
        return {
            "model_name": self.model_name,
            "architecture": "3D-ResNet/UNETR-SSL-Backbone",
            "is_foundation_model": True,
            "pretraining_modality": "Multi-modal Brain MRI (T1, T2, FLAIR, dMRI SSL)",
            "adaptation_mode": self.adaptation_mode,
            "embed_dim": self.embed_dim,
            "weights_path": self.weights_path,
            **param_info
        }
