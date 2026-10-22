"""
Official FOMO25 AMAES ResEnc-B Foundation Model Adapter.

Implements verified checkpoint loading, 3D feature extraction,
and linear probe / classification interfacing for the official
FOMO25 AMAES ResEnc-B backbone pretrained on FOMO260K.

Model: ResEnc-B (Residual Encoder UNet Backbone, 102M parameters total;
       Encoder backbone: 90,296,576 parameters, 6 stages: 32, 64, 128, 256, 320, 320).
Pretext Task: Augmented Masked Autoencoder (AMAES, Munk et al., arXiv:2408.00640).
Pretraining Dataset: FOMO260K (260,000 multi-sequence 3D brain MRIs).
Feature Representation: Stage 5 bottleneck representation (320 channels),
                        pooled via 3D Adaptive Average Pooling to [B, 320].
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from asparagus.modules.networks.resenc_unet import resenc_unet_b


class AMAESResEncBAdapter(nn.Module):
    """
    Adapter for the official FOMO25 AMAES ResEnc-B 3D Foundation Model.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        num_classes: int = 1,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 320

        # Instantiate official ResEnc-B network (dimensions='3D', input_channels=1, output_channels=1)
        self.base_net = resenc_unet_b(
            dimensions="3D",
            input_channels=1,
            output_channels=1,
            deep_supervision=False,
            use_skip_connections=True,
        )
        self.encoder = self.base_net.encoder

        # Load weights if checkpoint path is provided or exists in default location
        if checkpoint_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "weights", "resenc_unet_b.ckpt"
            )
            if os.path.exists(default_path):
                checkpoint_path = os.path.abspath(default_path)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load_pretrained_weights(checkpoint_path)

        # We only need the encoder backbone for feature extraction
        del self.base_net.decoder

        # Classification head for downstream tasks
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

        if freeze_backbone:
            self.freeze_encoder()

        if device is not None:
            self.to(device)

    def load_pretrained_weights(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Loads the 448 encoder keys from the official Lightning checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = checkpoint.get("state_dict", checkpoint)

        # Extract strictly encoder keys
        encoder_weights = {}
        for k, v in sd.items():
            if "encoder." in k:
                # Strip prefix down to submodule name
                clean_key = k[k.index("encoder.") + len("encoder."):]
                encoder_weights[clean_key] = v

        load_result = self.encoder.load_state_dict(encoder_weights, strict=True)
        return {
            "checkpoint_path": checkpoint_path,
            "keys_loaded": len(encoder_weights),
            "status": "SUCCESS",
        }

    def freeze_encoder(self):
        """Freezes all encoder weights for linear probing / frozen feature extraction."""
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def set_partial_ft(self):
        """Freezes stem and early stages, unfreezes stage 5 bottleneck + classifier."""
        self.freeze_encoder()
        for p in self.encoder.stages[-1].parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreezes encoder weights for partial or full fine-tuning."""
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled bottleneck embedding from 3D MRI volume.
        Input: [B, 1, D, H, W]
        Output: [B, 320]
        """
        # Encoder returns a list of 6 skip stages
        skips = self.encoder(x)
        # Bottleneck feature map is skips[-1] (Stage 5, 320 channels)
        bottleneck = skips[-1]
        pooled = F.adaptive_avg_pool3d(bottleneck, (1, 1, 1)).flatten(1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        Input: [B, 1, D, H, W]
        Output: [B, num_classes]
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "encoder_parameters": encoder_total,
        }
