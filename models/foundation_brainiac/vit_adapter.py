"""
models/foundation_brainiac/vit_adapter.py
Official BrainIAC 3D Vision Transformer (ViT-B/16³) Foundation Model Adapter.

Source: AIM-KannLab BrainIAC (Harvard Medical School / Mass General Brigham)
Backbone: 3D Vision Transformer (ViT-B/16³, 88.3M parameters, 12 layers, 12 heads, 768-d hidden)
Pretraining: SimCLR contrastive learning on large-scale multi-cohort brain MRI
"""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT
from safetensors.torch import load_file
from models.base import BaseBenchModel

DEFAULT_WEIGHTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "weights", "brainiac_vit_backbone.safetensors")
)


class BrainIACViTAdapter(BaseBenchModel):
    """
    Official BrainIAC 3D ViT Foundation Model Adapter.
    Supports authentic weights loading, feature extraction, linear probing,
    partial fine-tuning, and full fine-tuning.
    """
    def __init__(
        self,
        output_dim: int = 1,
        task_type: str = "classification",
        weights_path: Optional[str] = None,
        adaptation_mode: str = "linear_probe",
        dropout: float = 0.2,
        pretrained: bool = True,
        model_name: str = "BrainIAC-ViT-B16"
    ):
        if not pretrained:
            model_name = "Random-ViT-B16-Control"
        super().__init__(model_name=model_name, task_type=task_type, output_dim=output_dim)
        self.adaptation_mode = adaptation_mode
        self.pretrained = pretrained
        self.weights_path = weights_path or DEFAULT_WEIGHTS_PATH
        self.is_foundation_model = pretrained
        self.embed_dim = 768

        # Instantiate official MONAI 3D ViT-B/16 architecture (88.3M params)
        self.vit = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            spatial_dims=3,
            classification=False
        )

        # Downstream task classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.embed_dim, output_dim)
        )

        # Load authentic pretrained weights if pretrained is True
        if self.pretrained:
            if self.weights_path and os.path.exists(self.weights_path):
                self.load_pretrained_weights(self.weights_path)
            else:
                raise FileNotFoundError(
                    f"Authentic BrainIAC checkpoint not found at {self.weights_path}! "
                    "Random initialization fallback is strictly prohibited in SHIFT-MRI V4."
                )

        # Apply adaptation mode
        self.set_adaptation_mode(adaptation_mode)

    def load_pretrained_weights(self, weights_path: str):
        """Loads authentic BrainIAC safetensors checkpoint into MONAI ViT."""
        state_dict = load_file(weights_path)
        load_res = self.vit.load_state_dict(state_dict, strict=False)
        assert len(load_res.unexpected_keys) == 0, f"Unexpected keys in BrainIAC: {load_res.unexpected_keys}"
        for k in load_res.missing_keys:
            assert "norm_cross_attn" in k, f"Critical missing key: {k}"

    def set_adaptation_mode(self, mode: str):
        """Configure frozen/trainable parameters according to adaptation strategy."""
        self.adaptation_mode = mode
        if mode == "linear_probe":
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = True
        elif mode == "partial_ft":
            # Freeze early layers (blocks 0-9), fine-tune top 2 transformer blocks (10-11)
            for param in self.vit.parameters():
                param.requires_grad = False
            for block in self.vit.blocks[10:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
        elif mode == "full_ft":
            for param in self.vit.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown adaptation mode: {mode}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representations: (B, 768).
        Automatically interpolates to (96, 96, 96) if input is 128^3.
        """
        if x.shape[-3:] != (96, 96, 96):
            x = F.interpolate(x, size=(96, 96, 96), mode="trilinear", align_corners=False)
        
        # ViT forward returns (tokens, hidden_states)
        tokens, _ = self.vit(x) # (B, 216, 768)
        # Pool across the 216 spatial patches
        pooled = tokens.mean(dim=1) # (B, 768)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.head(feats)
