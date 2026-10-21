"""
Phase 2: Systematic Investigation of 3D ResNet-18 and ResNet-34 Baselines.
Explores optimization, regularization, data augmentation, class handling, and overfitting
using strictly OASIS-1 training (N=138) and validation (N=30) sets.
Never evaluates or peeks at the locked OASIS test set or ADNI external set.
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline.resnet3d import build_resnet18_3d, build_resnet34_3d
from evaluation.metrics import compute_classification_metrics
from scripts.run_master_benchmark_campaign import OASISPreprocessedDataset

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
RUNS_DIR = r"D:\brain_mri_data\benchmark_runs"
OUT_JSON = os.path.join(RUNS_DIR, "tables", "phase2_3d_baseline_investigation.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AugmentedOASISDataset(Dataset):
    """OASIS dataset with regularized 3D spatial augmentations."""
    def __init__(self, df, tensors_dir, is_train=False, augment_level="standard"):
        self.df = df.copy().reset_index(drop=True)
        self.tensors_dir = tensors_dir
        self.is_train = is_train
        self.augment_level = augment_level

        if "binary_label" in self.df.columns:
            self.df["binary_label"] = self.df["binary_label"].astype(float)
        elif "cdr" in self.df.columns:
            self.df["binary_label"] = (self.df["cdr"] > 0.0).astype(float)
        else:
            raise KeyError("Missing label column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row["session_id"]).strip()
        label = float(row["binary_label"])
        npy_path = os.path.join(self.tensors_dir, f"{sid}.npy")
        arr = np.load(npy_path)
        tensor = torch.from_numpy(arr).float()

        if self.is_train:
            # 1. Random spatial axis flips (D, H, W)
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[1])
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[3])

            # 2. Advanced regularized augmentations
            if self.augment_level == "strong":
                # Small intensity jitter (gain +/- 5%)
                gain = 1.0 + (random.random() - 0.5) * 0.10
                tensor = tensor * gain
                # Small Gaussian noise
                if random.random() > 0.5:
                    noise = torch.randn_like(tensor) * 0.02
                    tensor = tensor + noise

        return tensor, torch.tensor([label], dtype=torch.float32), sid


def train_and_evaluate_config(
    model_fn,
    train_df,
    val_df,
    config_name,
    lr=1e-3,
    weight_decay=1e-4,
    dropout=0.2,
    augment_level="standard",
    label_smoothing=0.0,
    epochs=25,
    patience=8,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = AugmentedOASISDataset(train_df, OASIS_TENSORS_DIR, is_train=True, augment_level=augment_level)
    val_ds = AugmentedOASISDataset(val_df, OASIS_TENSORS_DIR, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = model_fn(output_dim=1, dropout=dropout).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    def smooth_bce(logits, targets, smoothing=0.0):
        if smoothing > 0:
            smoothed = targets * (1.0 - smoothing) + 0.5 * smoothing
            return F.binary_cross_entropy_with_logits(logits, smoothed)
        return F.binary_cross_entropy_with_logits(logits, targets)

    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == "cuda"))
    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    best_metrics = None
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_probs, train_targets = [], []

        for images, targets, _ in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(images)
                loss = smooth_bce(logits, targets, smoothing=label_smoothing)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(images)
            train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
            train_targets.extend(targets.cpu().numpy().flatten())

        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        train_m = compute_classification_metrics(np.array(train_targets), np.array(train_probs))

        # Validation
        model.eval()
        val_loss = 0.0
        val_probs, val_targets = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    logits = model(images)
                    loss = F.binary_cross_entropy_with_logits(logits, targets)
                val_loss += loss.item() * len(images)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())

        val_loss = val_loss / len(val_loader.dataset)
        val_m = compute_classification_metrics(np.array(val_targets), np.array(val_probs))
        val_m["loss"] = round(val_loss, 4)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_auc": train_m["roc_auc"],
            "val_loss": val_m["loss"],
            "val_auc": val_m["roc_auc"],
            "val_bal_acc": val_m["balanced_accuracy"],
            "overfit_gap": round(train_m["roc_auc"] - val_m["roc_auc"], 4)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_m
            best_val_auc = val_m["roc_auc"]

        if epoch - best_epoch >= patience:
            break

    elapsed = time.time() - start_time
    return {
        "config_name": config_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "augment_level": augment_level,
        "label_smoothing": label_smoothing,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_auc": best_val_auc,
        "best_val_metrics": best_metrics,
        "final_train_auc": history[-1]["train_auc"],
        "final_overfit_gap": round(history[-1]["train_auc"] - best_val_auc, 4),
        "elapsed_seconds": round(elapsed, 1),
        "history": history
    }


def run_investigation():
    print("=" * 70)
    print("Phase 2: Systematic Investigation of 3D Baseline Tuning & Regularization")
    print("=" * 70)

    df = pd.read_csv(OASIS_SPLITS_CSV)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    print(f"Dataset: Train N={len(train_df)}, Val N={len(val_df)}")

    configs = [
        # ResNet-18 Variations
        {
            "name": "ResNet18_Original_Baseline",
            "model_fn": build_resnet18_3d,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.2,
            "augment_level": "standard",
            "label_smoothing": 0.0
        },
        {
            "name": "ResNet18_Conservative_LR_Warmup",
            "model_fn": build_resnet18_3d,
            "lr": 2e-4,
            "weight_decay": 1e-3,
            "dropout": 0.3,
            "augment_level": "standard",
            "label_smoothing": 0.0
        },
        {
            "name": "ResNet18_HighReg_Augmented",
            "model_fn": build_resnet18_3d,
            "lr": 2e-4,
            "weight_decay": 1e-2,
            "dropout": 0.5,
            "augment_level": "strong",
            "label_smoothing": 0.1
        },
        # ResNet-34 Variations
        {
            "name": "ResNet34_Original_Baseline",
            "model_fn": build_resnet34_3d,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.2,
            "augment_level": "standard",
            "label_smoothing": 0.0
        },
        {
            "name": "ResNet34_HighReg_Augmented",
            "model_fn": build_resnet34_3d,
            "lr": 2e-4,
            "weight_decay": 1e-2,
            "dropout": 0.5,
            "augment_level": "strong",
            "label_smoothing": 0.1
        }
    ]

    results = []
    for cfg in configs:
        print(f"\nEvaluating Config: {cfg['name']} (LR={cfg['lr']}, WD={cfg['weight_decay']}, Drop={cfg['dropout']})...")
        res = train_and_evaluate_config(
            model_fn=cfg["model_fn"],
            train_df=train_df,
            val_df=val_df,
            config_name=cfg["name"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            dropout=cfg["dropout"],
            augment_level=cfg["augment_level"],
            label_smoothing=cfg["label_smoothing"]
        )
        print(f"  Result: Best Val AUC = {res['best_val_auc']:.4f}, Val Loss = {res['best_val_loss']:.4f}, Overfit Gap = {res['final_overfit_gap']:.4f}")
        results.append(res)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll Phase 2 investigations completed. Results saved to {OUT_JSON}")
    return results


if __name__ == "__main__":
    run_investigation()
