"""
Phase 3: 2D Pretrained Transfer Learning for Volumetric Brain MRI.

Scientific Methodology:
1. Slice Extraction: 16 anatomically informative 2D slices per participant (8 axial, 8 coronal)
   sampling central ventricular and hippocampal subfields (slices 52-80 of 128^3 grid).
2. Participant Isolation: Partitioning is performed strictly at the participant level.
   Never are slices from one participant shared across partitions (0% leakage).
3. ImageNet Pretraining Provenance: Official torchvision ResNet-18 ImageNet-1K V1 weights.
   Note: Pretraining provides generic visual feature representations (textures, edges),
   not domain-specific brain MRI biological semantics.
4. Adaptation Regimes:
   - Frozen Feature Extraction (Linear Head): Only classifier weights are learned.
   - Partial Fine-Tuning: Deepest residual block (Layer 4) + classification head.
5. Participant-Level Aggregation:
   Slice-level sigmoid probabilities are averaged across slices per participant.
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
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_classification_metrics

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
ADNI_MANIFEST_CSV = r"D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv"
ADNI_TENSORS_DIR = r"D:\brain_mri_data\ADNI\processed\tensors"

RUNS_DIR = r"D:\brain_mri_data\benchmark_runs"
CHECKPOINTS_DIR = os.path.join(RUNS_DIR, "checkpoints")
TABLES_DIR = os.path.join(RUNS_DIR, "tables")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Informative central slice indices (out of 128)
AXIAL_SLICES = [52, 56, 60, 64, 68, 72, 76, 80]
CORONAL_SLICES = [52, 56, 60, 64, 68, 72, 76, 80]


class ParticipantSliceDataset(Dataset):
    """
    Extracts 2D slices from preprocessed 3D MRI volumes.
    Guarantees participant-level isolation.
    """
    def __init__(self, df, tensors_dir, is_train=False, is_adni=False):
        self.df = df.copy().reset_index(drop=True)
        self.tensors_dir = tensors_dir
        self.is_train = is_train
        self.is_adni = is_adni

        # Resolve binary labels
        if "binary_label" in self.df.columns:
            self.df["target"] = self.df["binary_label"].astype(float)
        elif "group" in self.df.columns:
            self.df["target"] = (self.df["group"] == "AD").astype(float)
        elif "cdr" in self.df.columns:
            self.df["target"] = (self.df["cdr"] > 0.0).astype(float)
        else:
            raise KeyError("No valid label column found")

        # Build list of (subject_idx, plane, slice_idx)
        self.slice_items = []
        for row_idx, row in self.df.iterrows():
            sub_id = row["subject_id"] if "subject_id" in row else row["session_id"]
            img_id = row.get("image_data_id", "")
            for s_idx in AXIAL_SLICES:
                self.slice_items.append((row_idx, "axial", s_idx, sub_id, img_id))
            for s_idx in CORONAL_SLICES:
                self.slice_items.append((row_idx, "coronal", s_idx, sub_id, img_id))

        # Cache volume tensors in memory or load on demand
        self._volume_cache = {}

    def _load_volume(self, row_idx):
        if row_idx in self._volume_cache:
            return self._volume_cache[row_idx]

        row = self.df.iloc[row_idx]
        if self.is_adni:
            sid = str(row["subject_id"]).strip()
            iid = str(row["image_data_id"]).strip()
            npy_path = os.path.join(self.tensors_dir, f"{sid}_{iid}.npy")
        else:
            sid = str(row["session_id"]).strip()
            npy_path = os.path.join(self.tensors_dir, f"{sid}.npy")

        arr = np.load(npy_path) # (1, 128, 128, 128)
        # Store compact float16 or keep as float32
        vol = arr[0] # (128, 128, 128)
        self._volume_cache[row_idx] = vol
        return vol

    def __len__(self):
        return len(self.slice_items)

    def __getitem__(self, idx):
        row_idx, plane, s_idx, sub_id, img_id = self.slice_items[idx]
        vol = self._load_volume(row_idx)
        target = self.df.iloc[row_idx]["target"]

        if plane == "axial":
            # slice along axis 0 (D): shape (128, 128)
            slice_2d = vol[s_idx, :, :]
        else:
            # slice along axis 1 (H): shape (128, 128)
            slice_2d = vol[:, s_idx, :]

        # Normalize slice to [0, 1] range for ImageNet backbone
        s_min, s_max = slice_2d.min(), slice_2d.max()
        if s_max - s_min > 1e-6:
            norm_slice = (slice_2d - s_min) / (s_max - s_min)
        else:
            norm_slice = np.zeros_like(slice_2d)

        # Standard ImageNet standardization (using grayscale repeat for 3 channels)
        # Repeat across 3 channels: (3, 128, 128)
        t = torch.from_numpy(norm_slice).float().unsqueeze(0).repeat(3, 1, 1)
        # ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                t = torch.flip(t, dims=[2])

        return t, torch.tensor([target], dtype=torch.float32), sub_id, img_id


class TransferResNet2D(nn.Module):
    """Pretrained 2D ResNet-18 with configurable adaptation mode."""
    def __init__(self, adaptation_mode="frozen", dropout=0.3):
        super().__init__()
        self.adaptation_mode = adaptation_mode
        self.base_model = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace classification head
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

        if adaptation_mode == "frozen":
            # Freeze all parameters except the new fc head
            for name, param in self.base_model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif adaptation_mode == "partial_finetune":
            # Freeze layers 1, 2, 3; unfreeze layer 4 + fc head
            for name, param in self.base_model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)


@torch.no_grad()
def evaluate_participant_level(model, dataset, batch_size=64):
    """
    Evaluates slice-level predictions and aggregates to participant-level.
    Returns participant-level classification metrics and predictions dataframe.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    slice_records = []
    for images, targets, sub_ids, img_ids in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets_np = targets.numpy().flatten()

        for p, t, s, i in zip(probs, targets_np, sub_ids, img_ids):
            slice_records.append({
                "subject_id": s,
                "image_data_id": i,
                "slice_prob": float(p),
                "target": float(t)
            })

    slice_df = pd.DataFrame(slice_records)
    # Aggregate per participant
    group_cols = ["subject_id", "target"]
    if dataset.is_adni:
        group_cols.append("image_data_id")

    agg_df = slice_df.groupby(group_cols).agg(
        prob_mean=("slice_prob", "mean"),
        prob_median=("slice_prob", "median")
    ).reset_index()

    y_true = agg_df["target"].values
    y_prob = agg_df["prob_mean"].values
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    return metrics, agg_df


def train_2d_transfer_model(
    adaptation_mode="frozen",
    epochs=15,
    lr=1e-3,
    weight_decay=1e-3,
    dropout=0.3,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    splits_df = pd.read_csv(OASIS_SPLITS_CSV)
    train_df = splits_df[splits_df["split"] == "train"]
    val_df = splits_df[splits_df["split"] == "val"]

    train_ds = ParticipantSliceDataset(train_df, OASIS_TENSORS_DIR, is_train=True)
    val_ds = ParticipantSliceDataset(val_df, OASIS_TENSORS_DIR, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)

    model = TransferResNet2D(adaptation_mode=adaptation_mode, dropout=dropout).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == "cuda"))

    best_val_auc = 0.0
    best_metrics = None
    best_state = None
    best_epoch = 0

    print(f"\n--- Training 2D Transfer Learning ({adaptation_mode}) ---")
    print(f"Train slices: {len(train_ds)}, Val slices: {len(val_ds)}, Trainable params: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets, _, _ in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)

        scheduler.step()
        train_loss /= len(train_ds)

        # Participant-level validation
        val_m, _ = evaluate_participant_level(model, val_ds)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Part AUC: {val_m['roc_auc']:.4f} | Val BalAcc: {val_m['balanced_accuracy']:.4f}")

        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            best_metrics = val_m
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"resnet2d_{adaptation_mode}_best.pt")
    torch.save(best_state, ckpt_path)
    print(f"Best checkpoint saved to {ckpt_path} (Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch})")

    return model, best_metrics, ckpt_path


def run_phase3():
    print("=" * 70)
    print("Phase 3: 2D Pretrained Transfer Learning Benchmark")
    print("=" * 70)

    # 1. Compare Frozen Features vs Partial Fine-Tuning
    frozen_model, frozen_val_m, frozen_ckpt = train_2d_transfer_model(
        adaptation_mode="frozen",
        epochs=15,
        lr=2e-3,
        weight_decay=1e-3,
        dropout=0.3
    )

    finetune_model, finetune_val_m, finetune_ckpt = train_2d_transfer_model(
        adaptation_mode="partial_finetune",
        epochs=15,
        lr=2e-4,
        weight_decay=1e-3,
        dropout=0.3
    )

    # Select best adaptation mode based strictly on Validation AUC
    print("\n--- Validation Selection (Phase 3) ---")
    print(f"Frozen Linear Probe Val AUC: {frozen_val_m['roc_auc']:.4f} (BalAcc: {frozen_val_m['balanced_accuracy']:.4f})")
    print(f"Partial Fine-Tuning Val AUC:  {finetune_val_m['roc_auc']:.4f} (BalAcc: {finetune_val_m['balanced_accuracy']:.4f})")

    if finetune_val_m["roc_auc"] >= frozen_val_m["roc_auc"]:
        best_mode = "partial_finetune"
        best_model = finetune_model
        best_val_m = finetune_val_m
    else:
        best_mode = "frozen"
        best_model = frozen_model
        best_val_m = frozen_val_m

    print(f"--> Selected Mode: {best_mode} (Val AUC: {best_val_m['roc_auc']:.4f})")

    # Save summary json
    summary = {
        "frozen_linear_probe": frozen_val_m,
        "partial_finetuning": finetune_val_m,
        "selected_mode": best_mode,
        "validation_metrics": best_val_m
    }
    out_path = os.path.join(TABLES_DIR, "phase3_2d_transfer_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Phase 3 validation results saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_phase3()
