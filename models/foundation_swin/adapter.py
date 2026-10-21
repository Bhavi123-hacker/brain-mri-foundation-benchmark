"""
MONAI Swin UNETR SSL Foundation Model Adapter.
Paper/Source: Project-MONAI Swin UNETR SSL (Tang et al., CVPR 2022).
Pretrained on large-scale volumetric 3D medical images via masked autoencoding and contrastive learning.
Supports linear probing and fine-tuning.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from monai.networks.nets import SwinUNETR
from models.base import BaseBenchModel


class SwinUNETRSSLAdapter(BaseBenchModel):
    """
    Adapter for Project-MONAI Swin UNETR Self-Supervised Pretrained 3D Vision Transformer.
    Uses SwinViT backbone representations for downstream classification tasks.
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

        # Official MONAI SwinViT 3D Transformer Backbone
        swin_net = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            spatial_dims=3
        )
        self.swin_encoder = swin_net.swinViT
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

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
        """Load pretrained checkpoint with robust key remapping."""
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        # Remap module prefix and fc1/fc2 naming differences across MONAI versions
        remapped_sd = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            clean_k = clean_k.replace(".mlp.fc1.", ".mlp.linear1.")
            clean_k = clean_k.replace(".mlp.fc2.", ".mlp.linear2.")
            remapped_sd[clean_k] = v

        load_res = self.swin_encoder.load_state_dict(remapped_sd, strict=False)
        self.weights_path = weights_path
        return load_res

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck embeddings: (B, embed_dim)."""
        multi_scale_features = self.swin_encoder(x)
        # multi_scale_features is a list; the deepest bottleneck feature is the last entry (index 4)
        bottleneck = multi_scale_features[4]  # shape (B, 768, D', H', W')
        pooled = self.pool(bottleneck)       # shape (B, 768, 1, 1, 1)
        return pooled.view(pooled.size(0), -1)

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
