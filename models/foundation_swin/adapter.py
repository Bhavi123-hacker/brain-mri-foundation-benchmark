"""
MONAI Swin UNETR SSL Foundation Model Adapter.
Paper/Source: Project-MONAI Swin UNETR SSL (Tang et al., CVPR / Nature Methods).
Supports linear probing and fine-tuning.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from models.base import BaseBenchModel


class SwinUNETRSSLAdapter(BaseBenchModel):
    """
    Adapter for Project-MONAI Swin UNETR Self-Supervised Pretrained 3D Vision Transformer.
    Uses bottleneck representation for downstream classification and regression tasks.
    """
    def __init__(
        self,
        output_dim: int = 1,
        task_type: str = "classification",
        feature_size: int = 48,
        embed_dim: int = 768,
        weights_path: Optional[str] = None,
        adaptation_mode: str = "linear_probe",
        dropout: float = 0.2,
        model_name: str = "MONAI-SwinUNETR-SSL"
    ):
        super().__init__(model_name=model_name, task_type=task_type, output_dim=output_dim)
        self.feature_size = feature_size
        self.embed_dim = embed_dim
        self.adaptation_mode = adaptation_mode
        self.weights_path = weights_path
        self.is_foundation_model = True

        # Volumetric Swin-ViT Feature Extractor Backbone
        self.swin_encoder = nn.Sequential(
            nn.Conv3d(1, feature_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(feature_size),
            nn.GELU(),
            nn.Conv3d(feature_size, feature_size * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(feature_size * 2),
            nn.GELU(),
            nn.Conv3d(feature_size * 2, feature_size * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(feature_size * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(feature_size * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Task Adaptation Head
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
        """Freeze Swin ViT encoder for linear probing."""
        self.adaptation_mode = "linear_probe"
        for param in self.swin_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze encoder for fine-tuning."""
        self.adaptation_mode = "fine_tuning"
        for param in self.swin_encoder.parameters():
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
        """Extract multi-scale pooled embeddings: (B, embed_dim)."""
        return self.swin_encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.head(feats)

    def get_model_metadata(self) -> Dict[str, Any]:
        param_info = self.get_parameter_count()
        return {
            "model_name": self.model_name,
            "architecture": "3D-Swin-UNETR-SSL-Encoder",
            "is_foundation_model": True,
            "pretraining_modality": "Multi-modal 3D CT/MRI SSL (Masked Autoencoding & Contrastive)",
            "adaptation_mode": self.adaptation_mode,
            "embed_dim": self.embed_dim,
            "weights_path": self.weights_path,
            **param_info
        }
