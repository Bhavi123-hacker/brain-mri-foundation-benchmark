"""
Foundation Model Downstream Task Adaptation Engine.
Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines.

Implements structured layer unfreezing and discriminative parameter group builders for:
1. BrainIAC (AIM-KannLab / Harvard BWH)
2. FOMO25 Winner (BrainFM4Challenges / CHUV / MICCAI 2025)
3. MONAI Swin UNETR SSL (Project-MONAI / Tang et al.)

Strategies:
- Strategy A: Linear Probe (Head-only control)
- Strategy B: Partial Fine-Tuning (Final representation/projection block + LayerNorm + head)
- Strategy C: Deeper Fine-Tuning (Final 25–50% of backbone + head)
- Strategy D: Full Fine-Tuning (Complete backbone + head)
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn

from models.foundation_brainiac.adapter import BrainIACAdapter
from models.foundation_fomo25.adapter import FOMO25Adapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter


WEIGHT_PATHS = {
    "BrainIAC": "weights/brainiac_vit_base.pt",
    "FOMO25": "weights/fomo25_winner_encoder.pt",
    "SwinUNETR": "weights/model_swinvit.pt",
}


def build_adapted_foundation_model(
    model_name: str,
    strategy: str = "linear_probe",
    weights_path: Optional[str] = None,
    dropout: float = 0.2,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Instantiate foundation model, load official weights, and apply adaptation unfreezing strategy.
    """
    if weights_path is None:
        weights_path = WEIGHT_PATHS.get(model_name)
    if weights_path and not os.path.exists(weights_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cand = os.path.join(project_root, weights_path)
        if os.path.exists(cand):
            weights_path = cand

    if model_name == "BrainIAC":
        model = BrainIACAdapter(
            output_dim=1,
            weights_path=weights_path,
            adaptation_mode="linear_probe",
            dropout=dropout,
        )
    elif model_name == "FOMO25":
        model = FOMO25Adapter(
            output_dim=1,
            weights_path=weights_path,
            adaptation_mode="linear_probe",
            dropout=dropout,
        )
    elif model_name in ("SwinUNETR", "Swin"):
        model = SwinUNETRSSLAdapter(
            output_dim=1,
            weights_path=weights_path,
            adaptation_mode="linear_probe",
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown foundation model: {model_name}")

    meta = configure_adaptation_strategy(model, model_name, strategy)
    model.to(device)
    return model, meta


def configure_adaptation_strategy(
    model: nn.Module, model_name: str, strategy: str
) -> Dict[str, Any]:
    """
    Configure requires_grad for layers based on the requested adaptation strategy.
    """
    strategy = strategy.lower().strip()
    valid_strategies = ["linear_probe", "partial_ft", "deeper_ft", "full_ft"]
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Head is ALWAYS trainable
    for param in model.head.parameters():
        param.requires_grad = True

    unfrozen_layers = ["head"]

    if strategy == "linear_probe":
        # Head only; backbone stays frozen
        pass

    elif strategy == "partial_ft":
        # Unfreeze the final projection / bottleneck layer & LayerNorm
        if model_name == "BrainIAC":
            for param in model.patch_embed[9].parameters():  # Linear 8192->768
                param.requires_grad = True
            for param in model.patch_embed[10].parameters(): # LayerNorm 768
                param.requires_grad = True
            unfrozen_layers.extend(["patch_embed.9 (Linear)", "patch_embed.10 (LayerNorm)"])

        elif model_name == "FOMO25":
            for param in model.encoder[14].parameters():     # Linear 256->512
                param.requires_grad = True
            for param in model.encoder[15].parameters():     # LayerNorm 512
                param.requires_grad = True
            unfrozen_layers.extend(["encoder.14 (Linear)", "encoder.15 (LayerNorm)"])

        elif model_name in ("SwinUNETR", "Swin"):
            for param in model.swin_encoder[11].parameters(): # Linear 192->768
                param.requires_grad = True
            for param in model.swin_encoder[12].parameters(): # LayerNorm 768
                param.requires_grad = True
            unfrozen_layers.extend(["swin_encoder.11 (Linear)", "swin_encoder.12 (LayerNorm)"])

    elif strategy == "deeper_ft":
        # Unfreeze final conv stage + final projection layer
        if model_name == "BrainIAC":
            for param in model.patch_embed[4].parameters():  # Conv3d 64->128
                param.requires_grad = True
            for param in model.patch_embed[5].parameters():  # BatchNorm 128
                param.requires_grad = True
            for param in model.patch_embed[9].parameters():  # Linear 8192->768
                param.requires_grad = True
            for param in model.patch_embed[10].parameters(): # LayerNorm 768
                param.requires_grad = True
            unfrozen_layers.extend(["patch_embed.4 (Conv3d)", "patch_embed.5 (BN)", "patch_embed.9 (Linear)", "patch_embed.10 (LN)"])

        elif model_name == "FOMO25":
            for param in model.encoder[9].parameters():      # Conv3d 128->256
                param.requires_grad = True
            for param in model.encoder[10].parameters():     # BatchNorm 256
                param.requires_grad = True
            for param in model.encoder[14].parameters():     # Linear 256->512
                param.requires_grad = True
            for param in model.encoder[15].parameters():     # LayerNorm 512
                param.requires_grad = True
            unfrozen_layers.extend(["encoder.9 (Conv3d)", "encoder.10 (BN)", "encoder.14 (Linear)", "encoder.15 (LN)"])

        elif model_name in ("SwinUNETR", "Swin"):
            for param in model.swin_encoder[6].parameters(): # Conv3d 96->192
                param.requires_grad = True
            for param in model.swin_encoder[7].parameters(): # BatchNorm 192
                param.requires_grad = True
            for param in model.swin_encoder[11].parameters(): # Linear 192->768
                param.requires_grad = True
            for param in model.swin_encoder[12].parameters(): # LayerNorm 768
                param.requires_grad = True
            unfrozen_layers.extend(["swin_encoder.6 (Conv3d)", "swin_encoder.7 (BN)", "swin_encoder.11 (Linear)", "swin_encoder.12 (LN)"])

    elif strategy == "full_ft":
        # Unfreeze complete backbone
        for param in model.parameters():
            param.requires_grad = True
        unfrozen_layers = ["entire backbone + head"]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "model_name": model_name,
        "strategy": strategy,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
        "unfrozen_layers": unfrozen_layers,
    }


def build_discriminative_optimizer(
    model: nn.Module,
    strategy: str,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Construct AdamW optimizer with discriminative learning rates:
    - classification head: head_lr
    - unfreezed backbone layers: backbone_lr
    """
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(param_groups)
