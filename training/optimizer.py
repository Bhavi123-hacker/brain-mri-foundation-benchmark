"""
Fair Optimizer and Learning Rate Scheduler Builder.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine",
    total_epochs: int = 50,
    warmup_epochs: int = 3,
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    Build optimizer and LR scheduler enforcing identical optimization policy across models.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1.0e-6
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        scheduler = None

    return optimizer, scheduler
