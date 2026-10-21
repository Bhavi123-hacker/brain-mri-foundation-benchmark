"""
Phase 5: Foundation Model Training & Adaptation Pipeline.

Evaluates MONAI Swin UNETR SSL (Tang et al., CVPR 2022) with official pretrained weights.
Supports:
1. Linear Probe: Frozen 3D SwinViT foundation backbone, training only the 768->1 classification head.
2. Fine-Tuning: End-to-end adaptation with low learning rate.
Fitted strictly on OASIS-1 training split (N=138) and tuned on validation split (N=30).
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from evaluation.metrics import compute_classification_metrics
from scripts.run_master_benchmark_campaign import OASISPreprocessedDataset

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
SWIN_WEIGHTS_PATH = r"D:\ml project\weights\model_swinvit.pt"

RUNS_DIR = r"D:\brain_mri_data\benchmark_runs"
CHECKPOINTS_DIR = os.path.join(RUNS_DIR, "checkpoints")
TABLES_DIR = os.path.join(RUNS_DIR, "tables")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_swin_model(model, loader, device=DEVICE):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets, all_ids = [], [], []

    use_cuda = (device.type == "cuda")
    for images, targets, sids in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        total_loss += loss.item() * len(images)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_targets.extend(targets.cpu().numpy().flatten())
        all_ids.extend(sids)

    avg_loss = total_loss / len(loader.dataset)
    m = compute_classification_metrics(np.array(all_targets), np.array(all_probs), threshold=0.5)
    m["loss"] = round(avg_loss, 4)
    return m, np.array(all_probs), np.array(all_targets), all_ids


def train_swin_adaptation(adaptation_mode="linear_probe", epochs=15, lr=1e-3, weight_decay=1e-3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    splits_df = pd.read_csv(OASIS_SPLITS_CSV)
    train_df = splits_df[splits_df["split"] == "train"]
    val_df = splits_df[splits_df["split"] == "val"]

    train_ds = OASISPreprocessedDataset(train_df, OASIS_TENSORS_DIR, is_train=True)
    val_ds = OASISPreprocessedDataset(val_df, OASIS_TENSORS_DIR, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = SwinUNETRSSLAdapter(
        weights_path=SWIN_WEIGHTS_PATH,
        adaptation_mode=adaptation_mode,
        dropout=0.3
    ).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_cuda = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_cuda)

    best_val_auc = 0.0
    best_metrics = None
    best_state = None
    best_epoch = 0

    print(f"\n--- Training MONAI Swin UNETR SSL ({adaptation_mode}) ---")
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, targets, _ in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)

        scheduler.step()
        train_loss /= len(train_ds)

        val_m, _, _, _ = evaluate_swin_model(model, val_loader, device=DEVICE)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val AUC: {val_m['roc_auc']:.4f} | Val BalAcc: {val_m['balanced_accuracy']:.4f}")

        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            best_metrics = val_m
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"swin_unetr_{adaptation_mode}_best.pt")
    torch.save(best_state, ckpt_path)
    print(f"Best checkpoint saved to {ckpt_path} (Val AUC: {best_val_auc:.4f} at epoch {best_epoch})")

    return model, best_metrics, ckpt_path


def run_phase5():
    print("=" * 70)
    print("Phase 5: Foundation Model Adaptation Benchmark (MONAI Swin UNETR SSL)")
    print("=" * 70)

    # 1. Linear Probe
    lp_model, lp_val_m, lp_ckpt = train_swin_adaptation(
        adaptation_mode="linear_probe",
        epochs=15,
        lr=2e-3,
        weight_decay=1e-3
    )

    # 2. Fine-Tuning
    ft_model, ft_val_m, ft_ckpt = train_swin_adaptation(
        adaptation_mode="fine_tuning",
        epochs=15,
        lr=2e-5,
        weight_decay=1e-4
    )

    print("\n--- Validation Selection (Phase 5) ---")
    print(f"Linear Probe Val AUC: {lp_val_m['roc_auc']:.4f} (BalAcc: {lp_val_m['balanced_accuracy']:.4f})")
    print(f"Fine-Tuning  Val AUC: {ft_val_m['roc_auc']:.4f} (BalAcc: {ft_val_m['balanced_accuracy']:.4f})")

    if ft_val_m["roc_auc"] >= lp_val_m["roc_auc"]:
        selected_mode = "fine_tuning"
        best_val_m = ft_val_m
    else:
        selected_mode = "linear_probe"
        best_val_m = lp_val_m

    print(f"--> Selected Swin UNETR Mode: {selected_mode} (Val AUC: {best_val_m['roc_auc']:.4f})")

    summary = {
        "linear_probe": lp_val_m,
        "fine_tuning": ft_val_m,
        "selected_mode": selected_mode,
        "validation_metrics": best_val_m
    }
    out_path = os.path.join(TABLES_DIR, "phase5_swin_unetr_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_phase5()
