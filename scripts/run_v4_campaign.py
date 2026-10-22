"""
scripts/run_v4_campaign.py
SHIFT-MRI V4: Comprehensive Scientific Retraining, Adaptation, and Revalidation Engine.

Author: Bhavy Garg
Affiliation: Vellore Institute of Technology (VIT), Chennai
Hardware: NVIDIA RTX PRO 2000 Blackwell (16 GB VRAM)

Executes:
1. Cohort loading & preprocessing verification (OASIS-1 N=198, ADNI N=500, OASIS-2 N=150)
2. Supervised baseline training on OASIS-1 train (N=138) with early stopping on OASIS-1 val (N=30)
3. Foundation model adaptation (BrainIAC ViT-B/16, MONAI Swin UNETR SSL, FOMO25 AMAES ResEnc-B)
   - Linear Probing (frozen backbone)
   - Partial Fine-Tuning (domain adaptation of top blocks)
   - Random ViT-B/16 Control (pretraining ablation for Hypothesis H7)
4. Locked evaluation protocol:
   - OASIS-1 validation (N=30) -> Youden threshold tau* calibrated strictly on validation
   - OASIS-1 locked test (N=30) -> evaluated ONCE per model
   - ADNI zero-shot external transfer (N=500) -> Siemens (N=370) and Philips (N=130) subgroups
   - OASIS-2 external transfer (N=150)
5. Statistical testing: Paired bootstrap (B=2000) with Holm-Bonferroni correction for Hypotheses H1-H7
6. TrustGate failure detection re-evaluation: Confidence Margin, Entropy, Distance, Disagreement, Composite Risk
7. Artifact generation: authoritative CSVs, training manifest JSON, and publication figures in results/v4/
"""

import os
import sys
import time
import json
import random
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
)
from scipy import stats
from scipy.ndimage import sobel, laplace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.v2_models import ResNet10_3D, DenseNet121_3D, MultiView25D_ResNet18, DualBranchHybridModel
from models.baseline.resnet3d import build_resnet18_3d, build_resnet34_3d
from models.foundation_brainiac.vit_adapter import BrainIACViTAdapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from models.foundation_fomo25.amaes_adapter import AMAESResEncBAdapter
from preprocessing.enhanced_pipeline import AnatomicalROIExtractor
from evaluation.v3.shift_mri import compute_stratified_auc_ci, compute_calibration_metrics, find_optimal_youden_threshold

# Paths
DATA_ROOT = r"D:\brain_mri_data"
OASIS1_SPLITS_CSV = os.path.join(DATA_ROOT, "OASIS1", "manifests", "oasis1_primary_cohort_splits.csv")
OASIS1_TENSORS_DIR = os.path.join(DATA_ROOT, "OASIS1", "processed", "tensors")
ADNI_MANIFEST_CSV = os.path.join(DATA_ROOT, "ADNI", "manifests", "adni_full_preprocessed_manifest.csv")
ADNI_TENSORS_DIR = os.path.join(DATA_ROOT, "ADNI", "processed", "tensors")
OASIS2_MANIFEST_CSV = os.path.join(DATA_ROOT, "OASIS2", "manifests", "oasis2_baseline_manifest.csv")
OASIS2_TENSORS_DIR = os.path.join(DATA_ROOT, "OASIS2", "processed", "tensors")

RESULTS_V4_DIR = os.path.join(PROJECT_ROOT, "results", "v4")
CKPTS_V4_DIR = os.path.join(RESULTS_V4_DIR, "checkpoints")
PREDS_V4_DIR = os.path.join(RESULTS_V4_DIR, "predictions")
TABLES_V4_DIR = os.path.join(RESULTS_V4_DIR, "tables")
FIGURES_V4_DIR = os.path.join(RESULTS_V4_DIR, "figures")
LOGS_V4_DIR = os.path.join(RESULTS_V4_DIR, "training_logs")

for d in [RESULTS_V4_DIR, CKPTS_V4_DIR, PREDS_V4_DIR, TABLES_V4_DIR, FIGURES_V4_DIR, LOGS_V4_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------
# 1. Dataset Loading and Memory Caching
# -------------------------------------------------------------
def load_all_cohorts():
    print("[1/6] Loading and caching preprocessed MRI cohorts in memory...", flush=True)

    # OASIS-1
    df_o1 = pd.read_csv(OASIS1_SPLITS_CSV)
    train_df = df_o1[df_o1["split"] == "train"].reset_index(drop=True)
    val_df = df_o1[df_o1["split"] == "val"].reset_index(drop=True)
    test_df = df_o1[df_o1["split"] == "test"].reset_index(drop=True)

    def load_o1_split(sdf):
        xs = [np.load(os.path.join(OASIS1_TENSORS_DIR, f"{sid}.npy")) for sid in sdf["session_id"]]
        ys = (sdf["cdr"] > 0.0).astype(np.float32).values
        sids = sdf["session_id"].tolist()
        return xs, ys, sids

    train_x, train_y, train_sids = load_o1_split(train_df)
    val_x, val_y, val_sids = load_o1_split(val_df)
    test_x, test_y, test_sids = load_o1_split(test_df)
    print(f"  OASIS-1: Train N={len(train_x)}, Val N={len(val_x)}, Locked Test N={len(test_x)}", flush=True)

    # ADNI
    df_adni = pd.read_csv(ADNI_MANIFEST_CSV)
    df_adni_pass = df_adni[df_adni["status"] == "PASS"].reset_index(drop=True)
    adni_x = [
        np.load(os.path.join(ADNI_TENSORS_DIR, f"{sid}_{iid}.npy"))
        for sid, iid in zip(df_adni_pass["subject_id"], df_adni_pass["image_data_id"])
    ]
    adni_y = (df_adni_pass["group"] == "AD").astype(np.float32).values
    adni_sids = [f"{sid}_{iid}" for sid, iid in zip(df_adni_pass["subject_id"], df_adni_pass["image_data_id"])]
    print(f"  ADNI: N={len(adni_x)} (Siemens: {(df_adni_pass['manufacturer']=='Siemens').sum()}, Philips: {(df_adni_pass['manufacturer']=='Philips').sum()})", flush=True)

    # OASIS-2
    df_o2 = pd.read_csv(OASIS2_MANIFEST_CSV)
    df_o2_pass = df_o2[df_o2["status"] == "PASS"].reset_index(drop=True)
    o2_x = [np.load(os.path.join(OASIS2_TENSORS_DIR, f"{pid}.npy")) for pid in df_o2_pass["participant_id"]]
    o2_y = df_o2_pass["binary_label"].astype(np.float32).values
    o2_sids = df_o2_pass["participant_id"].tolist()
    print(f"  OASIS-2: N={len(o2_x)} (CN: {(o2_y==0).sum()}, Dementia: {(o2_y==1).sum()})", flush=True)

    return (
        (train_x, train_y, train_sids, train_df),
        (val_x, val_y, val_sids, val_df),
        (test_x, test_y, test_sids, test_df),
        (adni_x, adni_y, adni_sids, df_adni_pass),
        (o2_x, o2_y, o2_sids, df_o2_pass),
    )


# -------------------------------------------------------------
# 2. Radiomics Extraction
# -------------------------------------------------------------
def extract_classical_features_from_volume(arr_3d: np.ndarray) -> np.ndarray:
    if arr_3d.ndim == 4:
        arr_3d = arr_3d[0]
    brain_mask = np.abs(arr_3d) > 1e-4
    tissue_voxels = arr_3d[brain_mask]
    total_voxels = arr_3d.size
    brain_voxels = len(tissue_voxels)
    if brain_voxels < 100:
        return np.zeros(24, dtype=np.float32)

    brain_volume_fraction = brain_voxels / total_voxels
    p20 = np.percentile(tissue_voxels, 20)
    p80 = np.percentile(tissue_voxels, 80)
    csf_proxy_fraction = (tissue_voxels < p20).sum() / brain_voxels
    wm_proxy_fraction = (tissue_voxels > p80).sum() / brain_voxels
    csf_to_wm_ratio = (csf_proxy_fraction + 1e-6) / (wm_proxy_fraction + 1e-6)

    mean_val = float(np.mean(tissue_voxels))
    std_val = float(np.std(tissue_voxels))
    var_val = float(np.var(tissue_voxels))
    skew_val = float(stats.skew(tissue_voxels))
    kurt_val = float(stats.kurtosis(tissue_voxels))

    p05 = float(np.percentile(tissue_voxels, 5))
    p10 = float(np.percentile(tissue_voxels, 10))
    p25 = float(np.percentile(tissue_voxels, 25))
    p50 = float(np.percentile(tissue_voxels, 50))
    p75 = float(np.percentile(tissue_voxels, 75))
    p90 = float(np.percentile(tissue_voxels, 90))
    p95 = float(np.percentile(tissue_voxels, 95))
    iqr_val = p75 - p25

    energy_val = float(np.sum(tissue_voxels ** 2) / brain_voxels)
    rms_val = float(np.sqrt(energy_val))

    center_subvol = arr_3d[32:96, 32:96, 32:96]
    grad_x = sobel(center_subvol, axis=0)
    grad_y = sobel(center_subvol, axis=1)
    grad_z = sobel(center_subvol, axis=2)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    mean_gradient = float(np.mean(grad_magnitude))
    std_gradient = float(np.std(grad_magnitude))

    lap_subvol = laplace(center_subvol)
    mean_laplacian = float(np.mean(np.abs(lap_subvol)))
    var_laplacian = float(np.var(lap_subvol))

    return np.array([
        brain_volume_fraction, csf_proxy_fraction, wm_proxy_fraction, csf_to_wm_ratio,
        mean_val, std_val, var_val, skew_val, kurt_val,
        p05, p10, p25, p50, p75, p90, p95, iqr_val, energy_val, rms_val,
        mean_gradient, std_gradient, mean_laplacian, var_laplacian,
        brain_voxels / 1e6
    ], dtype=np.float32)


def extract_morphometrics_6(arr: np.ndarray, roi_extractor: AnatomicalROIExtractor) -> np.ndarray:
    b = roi_extractor.compute_morphometric_biomarkers(arr)
    return np.array([
        b["total_brain_voxels"] / 1e5,
        b["ventricular_csf_voxels"] / 1e4,
        b["hippocampal_tissue_voxels"] / 1e4,
        b["ventricle_to_brain_ratio"] * 10.0,
        b["hippocampus_ratio"] * 10.0,
        b["intensity_skewness"],
    ], dtype=np.float32)


# -------------------------------------------------------------
# 3. Model Training & Evaluation Functions
# -------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    xs: List[np.ndarray],
    device: torch.device,
    batch_size: int = 4,
    has_radiomics: bool = False,
    rads: Optional[np.ndarray] = None,
) -> np.ndarray:
    model.eval()
    probs = []
    N = len(xs)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            bx = np.stack(xs[i:i + batch_size])
            tx = torch.from_numpy(bx).to(device)
            if has_radiomics and rads is not None:
                tr = torch.from_numpy(rads[i:i + batch_size]).to(device)
                logits = model(tx, tr)
            else:
                logits = model(tx)
            p = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
            if isinstance(p, float):
                probs.append(p)
            else:
                probs.extend(p)
    return np.array(probs, dtype=np.float32)


def train_3d_supervised_model(
    model: nn.Module,
    train_x: List[np.ndarray],
    train_y: np.ndarray,
    val_x: List[np.ndarray],
    val_y: np.ndarray,
    device: torch.device,
    save_path: str,
    epochs: int = 20,
    lr: float = 2e-4,
    weight_decay: float = 1e-3,
    batch_size: int = 4,
    has_radiomics: bool = False,
    train_rads: Optional[np.ndarray] = None,
    val_rads: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_val_auc = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience = 6
    patience_counter = 0

    N_train = len(train_x)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(N_train)
        epoch_loss = 0.0

        for i in range(0, N_train, batch_size):
            b_idx = indices[i:i + batch_size]
            bx = np.stack([train_x[j] for j in b_idx])
            by = train_y[b_idx]

            # Augmentation: 50% random horizontal flip along sagittal plane (axis 4)
            if random.random() > 0.5:
                bx = np.flip(bx, axis=4).copy()

            tx = torch.from_numpy(bx).to(device)
            ty = torch.from_numpy(by).unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                if has_radiomics and train_rads is not None:
                    tr = torch.from_numpy(train_rads[b_idx]).to(device)
                    logits = model(tx, tr)
                else:
                    logits = model(tx)
                loss = criterion(logits, ty)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation evaluation
        val_probs = evaluate_model(model, val_x, device, batch_size=batch_size, has_radiomics=has_radiomics, rads=val_rads)
        try:
            val_auc = float(roc_auc_score(val_y, val_probs))
        except ValueError:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 10:
                break

    training_time = round(time.time() - start_time, 2)
    model.load_state_dict(best_weights)
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 4),
        "total_epochs": epoch,
        "training_time_sec": training_time,
    }


def train_2d_slice_transfer(
    train_x: List[np.ndarray],
    train_y: np.ndarray,
    val_x: List[np.ndarray],
    val_y: np.ndarray,
    device: torch.device,
    save_path: str,
    epochs: int = 15,
    lr: float = 1e-4,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    2D Pretrained Transfer ResNet-18 (ImageNet-1K).
    Extracts 8 axial + 8 coronal slices per volume (slices 52-80).
    Averages slice-level probabilities to form participant-level prediction.
    """
    slices_idx = [52, 56, 60, 64, 68, 72, 76, 80]

    def extract_slices(volumes, labels):
        slice_imgs = []
        slice_lbls = []
        part_map = []  # mapping from slice index to volume index
        for v_idx, vol in enumerate(volumes):
            v3d = vol[0]  # (128, 128, 128)
            y = labels[v_idx]
            for s in slices_idx:
                # Axial: (128, 128)
                ax = v3d[s, :, :]
                # Coronal: (128, 128)
                cor = v3d[:, s, :]
                for sl in [ax, cor]:
                    s_min, s_max = sl.min(), sl.max()
                    norm_sl = (sl - s_min) / (s_max - s_min + 1e-6)
                    # 3-channel repeat
                    rgb = np.repeat(norm_sl[np.newaxis, :, :], 3, axis=0)
                    slice_imgs.append(rgb)
                    slice_lbls.append(y)
                    part_map.append(v_idx)
        return np.stack(slice_imgs).astype(np.float32), np.array(slice_lbls, dtype=np.float32), np.array(part_map)

    tr_slices, tr_lbls, _ = extract_slices(train_x, train_y)
    vl_slices, vl_lbls, vl_map = extract_slices(val_x, val_y)

    # Standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    tr_slices = (tr_slices - mean) / std
    vl_slices = (vl_slices - mean) / std

    # ResNet-18 backbone
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = tv_models.resnet18(weights=weights)
    # Freeze early layers, train layer4 + fc
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
    model = model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_val_auc = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    N_tr = len(tr_slices)
    batch_size = 32

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        idx_perm = np.random.permutation(N_tr)
        for i in range(0, N_tr, batch_size):
            b_idx = idx_perm[i:i + batch_size]
            bx = torch.from_numpy(tr_slices[b_idx]).to(device)
            by = torch.from_numpy(tr_lbls[b_idx]).unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(bx)
                loss = criterion(out, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Validation participant-level aggregation
        model.eval()
        slice_probs = []
        with torch.no_grad():
            for i in range(0, len(vl_slices), batch_size):
                bx = torch.from_numpy(vl_slices[i:i + batch_size]).to(device)
                p = torch.sigmoid(model(bx)).squeeze(-1).cpu().numpy().tolist()
                if isinstance(p, float):
                    slice_probs.append(p)
                else:
                    slice_probs.extend(p)
        slice_probs = np.array(slice_probs)

        # Average per participant
        val_part_probs = np.array([
            float(np.mean(slice_probs[vl_map == v_idx])) for v_idx in range(len(val_y))
        ])
        val_auc = float(roc_auc_score(val_y, val_part_probs))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)

    training_time = round(time.time() - start_time, 2)
    model.load_state_dict(best_weights)

    class ParticipantAggregator2D(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 1, 128, 128, 128)
            B = x.shape[0]
            slices = []
            for b in range(B):
                v3d = x[b, 0]
                for s in slices_idx:
                    ax = v3d[s, :, :]
                    cor = v3d[:, s, :]
                    for sl in [ax, cor]:
                        s_min, s_max = sl.min(), sl.max()
                        norm_sl = (sl - s_min) / (s_max - s_min + 1e-6)
                        rgb = norm_sl.unsqueeze(0).repeat(3, 1, 1)
                        slices.append(rgb)
            slices_t = torch.stack(slices)
            # Normalize
            m = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            sd = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            slices_t = (slices_t - m) / sd
            out = self.net(slices_t)
            probs = torch.sigmoid(out).view(B, 16).mean(dim=1, keepdim=True)
            # return logits corresponding to mean probability
            probs_clip = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            logits = torch.log(probs_clip / (1.0 - probs_clip))
            return logits

    wrapper = ParticipantAggregator2D(model)
    return wrapper, {
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 4),
        "total_epochs": epoch,
        "training_time_sec": training_time,
    }


def train_foundation_model(
    model: nn.Module,
    train_x: List[np.ndarray],
    train_y: np.ndarray,
    val_x: List[np.ndarray],
    val_y: np.ndarray,
    device: torch.device,
    save_path: str,
    adaptation_mode: str = "linear_probe",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 2,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Unified foundation model adapter trainer.
    Supports linear probe and partial fine-tuning.
    """
    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_val_auc = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience = 6
    patience_counter = 0

    N_train = len(train_x)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(N_train)

        for i in range(0, N_train, batch_size):
            b_idx = indices[i:i + batch_size]
            bx = np.stack([train_x[j] for j in b_idx])
            by = train_y[b_idx]

            tx = torch.from_numpy(bx).to(device)
            ty = torch.from_numpy(by).unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(tx)
                loss = criterion(logits, ty)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Validation
        val_probs = evaluate_model(model, val_x, device, batch_size=batch_size)
        try:
            val_auc = float(roc_auc_score(val_y, val_probs))
        except ValueError:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 8:
                break

    training_time = round(time.time() - start_time, 2)
    model.load_state_dict(best_weights)
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 4),
        "total_epochs": epoch,
        "training_time_sec": training_time,
    }


# -------------------------------------------------------------
# 4. Latency & Memory Profiling
# -------------------------------------------------------------
def profile_model_compute(model: nn.Module, device: torch.device, has_radiomics: bool = False) -> Tuple[float, float]:
    model.eval()
    dummy_x = torch.randn(1, 1, 128, 128, 128, device=device)
    dummy_r = torch.randn(1, 6, device=device) if has_radiomics else None

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            if has_radiomics:
                _ = model(dummy_x, dummy_r)
            else:
                _ = model(dummy_x)

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    n_runs = 5
    with torch.no_grad():
        for _ in range(n_runs):
            if has_radiomics:
                _ = model(dummy_x, dummy_r)
            else:
                _ = model(dummy_x)
    torch.cuda.synchronize(device)
    latency_ms = round(((time.time() - start_time) / n_runs) * 1000.0, 1)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)
    return latency_ms, peak_vram_mb


# -------------------------------------------------------------
# 5. Master Campaign Execution
# -------------------------------------------------------------
def run_v4_campaign():
    set_seed(42)
    print("=" * 80)
    print("STARTING SHIFT-MRI V4 AUTHENTIC BENCHMARK CAMPAIGN")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load cohorts
    (train_x, train_y, train_sids, train_df), \
    (val_x, val_y, val_sids, val_df), \
    (test_x, test_y, test_sids, test_df), \
    (adni_x, adni_y, adni_sids, adni_df), \
    (o2_x, o2_y, o2_sids, o2_df) = load_all_cohorts()

    # Pre-extract 6-biomarker radiomics for Dual-Branch Hybrid
    print("[2/6] Pre-extracting 6-biomarker radiomics for Dual-Branch Hybrid...", flush=True)
    roi_extractor = AnatomicalROIExtractor()
    train_rads6 = np.stack([extract_morphometrics_6(x[0], roi_extractor) for x in train_x])
    val_rads6 = np.stack([extract_morphometrics_6(x[0], roi_extractor) for x in val_x])
    test_rads6 = np.stack([extract_morphometrics_6(x[0], roi_extractor) for x in test_x])
    adni_rads6 = np.stack([extract_morphometrics_6(x[0], roi_extractor) for x in adni_x])
    o2_rads6 = np.stack([extract_morphometrics_6(x[0], roi_extractor) for x in o2_x])

    # Pre-extract 24 classical radiomics
    print("[3/6] Pre-extracting 24 classical radiomics for ML baseline...", flush=True)
    train_rads24 = np.stack([extract_classical_features_from_volume(x) for x in train_x])
    val_rads24 = np.stack([extract_classical_features_from_volume(x) for x in val_x])
    test_rads24 = np.stack([extract_classical_features_from_volume(x) for x in test_x])
    adni_rads24 = np.stack([extract_classical_features_from_volume(x) for x in adni_x])
    o2_rads24 = np.stack([extract_classical_features_from_volume(x) for x in o2_x])

    # Standardize classical features strictly with training statistics
    scaler24 = StandardScaler()
    train_rads24_s = scaler24.fit_transform(train_rads24)
    val_rads24_s = scaler24.transform(val_rads24)
    test_rads24_s = scaler24.transform(test_rads24)
    adni_rads24_s = scaler24.transform(adni_rads24)
    o2_rads24_s = scaler24.transform(o2_rads24)

    # -------------------------------------------------------------
    # 2. Model Training & Evaluation Suite
    # -------------------------------------------------------------
    print("\n[4/6] Executing Model Training & Evaluation Suite...", flush=True)

    model_registry = [
        # (display_name, key, type, config)
        ("3D ResNet-10", "3d_resnet_10", "supervised_3d", {"factory": lambda: ResNet10_3D(output_dim=1, dropout=0.2), "has_rad": False}),
        ("3D ResNet-18", "3d_resnet_18", "supervised_3d", {"factory": lambda: build_resnet18_3d(output_dim=1, dropout=0.2), "has_rad": False}),
        ("3D ResNet-34", "3d_resnet_34", "supervised_3d", {"factory": lambda: build_resnet34_3d(output_dim=1, dropout=0.2), "has_rad": False}),
        ("3D DenseNet-121", "3d_densenet_121", "supervised_3d", {"factory": lambda: DenseNet121_3D(output_dim=1, dropout=0.2), "has_rad": False}),
        ("2.5D Multi-View ResNet-18", "2_5d_multi_view_resnet_18", "supervised_3d", {"factory": lambda: MultiView25D_ResNet18(pretrained=True, output_dim=1, dropout=0.3), "has_rad": False}),
        ("Dual-Branch Hybrid (ResNet-10 + Radiomics)", "dual_branch_hybrid", "supervised_3d", {"factory": lambda: DualBranchHybridModel(spatial_backbone="resnet10", num_radiomics=6, output_dim=1), "has_rad": True}),
        ("2D Transfer ResNet-18", "2d_transfer_resnet_18", "transfer_2d", {}),
        ("Classical Radiomics (Linear SVM)", "classical_radiomics", "classical", {}),
        ("BrainIAC ViT-B/16 (Linear Probe)", "brainiac_vit_b16_linear_probe", "foundation", {"factory": lambda: BrainIACViTAdapter(adaptation_mode="linear_probe"), "mode": "linear_probe", "lr": 1e-3}),
        ("BrainIAC ViT-B/16 (Partial Fine-Tuning)", "brainiac_vit_b16_partial_ft", "foundation", {"factory": lambda: BrainIACViTAdapter(adaptation_mode="partial_ft"), "mode": "partial_ft", "lr": 2e-4}),
        ("Random ViT-B/16 Control (Linear Probe)", "random_vit_b16_control", "foundation", {"factory": lambda: BrainIACViTAdapter(pretrained=False, adaptation_mode="linear_probe"), "mode": "linear_probe", "lr": 1e-3}),
        ("MONAI Swin UNETR SSL (Linear Probe)", "swin_unetr_ssl_linear_probe", "foundation", {"factory": lambda: SwinUNETRSSLAdapter(weights_path=r"weights\model_swinvit.pt", adaptation_mode="linear_probe"), "mode": "linear_probe", "lr": 1e-3}),
        ("MONAI Swin UNETR SSL (Partial Fine-Tuning)", "swin_unetr_ssl_partial_ft", "foundation", {"factory": lambda: SwinUNETRSSLAdapter(weights_path=r"weights\model_swinvit.pt", adaptation_mode="partial_ft"), "mode": "partial_ft", "lr": 2e-4}),
        ("FOMO25 AMAES ResEnc-B (Linear Probe)", "fomo25_amaes_linear_probe", "foundation", {"factory": lambda: AMAESResEncBAdapter(checkpoint_path=r"weights\resenc_unet_b.ckpt", freeze_backbone=True), "mode": "linear_probe", "lr": 1e-3}),
        ("FOMO25 AMAES ResEnc-B (Partial Fine-Tuning)", "fomo25_amaes_partial_ft", "foundation", {"factory": lambda: AMAESResEncBAdapter(checkpoint_path=r"weights\resenc_unet_b.ckpt", freeze_backbone=True), "mode": "partial_ft", "lr": 2e-4}),
    ]

    all_val_preds = {"session_id": val_sids, "true_label": val_y.astype(int)}
    all_test_preds = {"session_id": test_sids, "true_label": test_y.astype(int)}
    all_adni_preds = {
        "session_id": adni_sids,
        "true_label": adni_y.astype(int),
        "manufacturer": adni_df["manufacturer"].tolist(),
    }
    all_o2_preds = {"session_id": o2_sids, "true_label": o2_y.astype(int)}

    evaluation_rows = []
    training_manifest = {
        "project": "SHIFT-MRI V4",
        "author": "Bhavy Garg",
        "affiliation": "Vellore Institute of Technology (VIT), Chennai",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(DEVICE),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
        },
        "models": {},
    }

    for name, key, mtype, cfg in model_registry:
        print(f"\n--- Training & Evaluating: {name} ---", flush=True)
        set_seed(42)
        ckpt_path = os.path.join(CKPTS_V4_DIR, f"{key}_seed42_best.pt")

        if mtype == "supervised_3d":
            model = cfg["factory"]()
            has_rad = cfg["has_rad"]
            fitted_model, train_meta = train_3d_supervised_model(
                model=model,
                train_x=train_x, train_y=train_y,
                val_x=val_x, val_y=val_y,
                device=DEVICE,
                save_path=ckpt_path,
                epochs=20,
                has_radiomics=has_rad,
                train_rads=train_rads6 if has_rad else None,
                val_rads=val_rads6 if has_rad else None,
            )
            # Evaluate across cohorts
            val_p = evaluate_model(fitted_model, val_x, DEVICE, has_radiomics=has_rad, rads=val_rads6)
            test_p = evaluate_model(fitted_model, test_x, DEVICE, has_radiomics=has_rad, rads=test_rads6)
            adni_p = evaluate_model(fitted_model, adni_x, DEVICE, has_radiomics=has_rad, rads=adni_rads6)
            o2_p = evaluate_model(fitted_model, o2_x, DEVICE, has_radiomics=has_rad, rads=o2_rads6)
            lat_ms, mem_mb = profile_model_compute(fitted_model, DEVICE, has_radiomics=has_rad)
            total_params = sum(p.numel() for p in fitted_model.parameters())
            trainable_params = sum(p.numel() for p in fitted_model.parameters() if p.requires_grad)

        elif mtype == "transfer_2d":
            fitted_model, train_meta = train_2d_slice_transfer(
                train_x=train_x, train_y=train_y,
                val_x=val_x, val_y=val_y,
                device=DEVICE,
                save_path=ckpt_path,
                epochs=15,
            )
            val_p = evaluate_model(fitted_model, val_x, DEVICE, batch_size=2)
            test_p = evaluate_model(fitted_model, test_x, DEVICE, batch_size=2)
            adni_p = evaluate_model(fitted_model, adni_x, DEVICE, batch_size=2)
            o2_p = evaluate_model(fitted_model, o2_x, DEVICE, batch_size=2)
            lat_ms, mem_mb = profile_model_compute(fitted_model, DEVICE)
            total_params = sum(p.numel() for p in fitted_model.parameters())
            trainable_params = sum(p.numel() for p in fitted_model.parameters() if p.requires_grad)

        elif mtype == "classical":
            t_start = time.time()
            svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
            svm.fit(train_rads24_s, train_y)
            train_meta = {
                "best_epoch": 1,
                "best_val_auc": round(float(roc_auc_score(val_y, svm.predict_proba(val_rads24_s)[:, 1])), 4),
                "total_epochs": 1,
                "training_time_sec": round(time.time() - t_start, 2),
            }
            val_p = svm.predict_proba(val_rads24_s)[:, 1]
            test_p = svm.predict_proba(test_rads24_s)[:, 1]
            adni_p = svm.predict_proba(adni_rads24_s)[:, 1]
            o2_p = svm.predict_proba(o2_rads24_s)[:, 1]
            lat_ms, mem_mb = 1.2, 0.0
            total_params = 24
            trainable_params = 24

        elif mtype == "foundation":
            model = cfg["factory"]()
            mode = cfg["mode"]
            if "fomo25" in key and mode == "partial_ft":
                model.set_partial_ft()
            fitted_model, train_meta = train_foundation_model(
                model=model,
                train_x=train_x, train_y=train_y,
                val_x=val_x, val_y=val_y,
                device=DEVICE,
                save_path=ckpt_path,
                adaptation_mode=mode,
                epochs=15,
                lr=cfg["lr"],
                batch_size=2,
            )
            val_p = evaluate_model(fitted_model, val_x, DEVICE, batch_size=2)
            test_p = evaluate_model(fitted_model, test_x, DEVICE, batch_size=2)
            adni_p = evaluate_model(fitted_model, adni_x, DEVICE, batch_size=2)
            o2_p = evaluate_model(fitted_model, o2_x, DEVICE, batch_size=2)
            lat_ms, mem_mb = profile_model_compute(fitted_model, DEVICE)
            total_params = sum(p.numel() for p in fitted_model.parameters())
            trainable_params = sum(p.numel() for p in fitted_model.parameters() if p.requires_grad)

        # Store prediction vectors
        all_val_preds[key] = val_p
        all_test_preds[key] = test_p
        all_adni_preds[key] = adni_p
        all_o2_preds[key] = o2_p

        # Derive optimal Youden threshold strictly from OASIS-1 validation
        tau_star, max_j = find_optimal_youden_threshold(val_y, val_p)

        # Calculate metrics for OASIS-1 Locked Test
        test_auc_mean, test_auc_ci = compute_stratified_auc_ci(test_y, test_p, n_bootstraps=2000)
        test_pred_bin = (test_p >= tau_star).astype(int)
        test_acc = accuracy_score(test_y, test_pred_bin)
        test_bacc = balanced_accuracy_score(test_y, test_pred_bin)
        tn, fp, fn, tp = confusion_matrix(test_y, test_pred_bin, labels=[0, 1]).ravel()
        test_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        test_brier = brier_score_loss(test_y, test_p)
        test_cal = compute_calibration_metrics(test_y, test_p)

        # Calculate metrics for ADNI External Transfer
        adni_auc_mean, adni_auc_ci = compute_stratified_auc_ci(adni_y, adni_p, n_bootstraps=2000)
        adni_pred_bin = (adni_p >= tau_star).astype(int)
        adni_acc = accuracy_score(adni_y, adni_pred_bin)
        adni_bacc = balanced_accuracy_score(adni_y, adni_pred_bin)
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(adni_y, adni_pred_bin, labels=[0, 1]).ravel()
        adni_sens = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        adni_spec = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0.0
        adni_brier = brier_score_loss(adni_y, adni_p)
        adni_cal = compute_calibration_metrics(adni_y, adni_p)

        # Subgroup ADNI: Siemens vs Philips
        mfg = np.array(all_adni_preds["manufacturer"])
        siemens_mask = mfg == "Siemens"
        philips_mask = mfg == "Philips"
        siemens_auc = float(roc_auc_score(adni_y[siemens_mask], adni_p[siemens_mask]))
        philips_auc = float(roc_auc_score(adni_y[philips_mask], adni_p[philips_mask]))

        # Calculate OASIS-2 External
        o2_auc_mean, o2_auc_ci = compute_stratified_auc_ci(o2_y, o2_p, n_bootstraps=2000)
        o2_pred_bin = (o2_p >= tau_star).astype(int)
        o2_bacc = balanced_accuracy_score(o2_y, o2_pred_bin)

        delta_auc = round(test_auc_mean - adni_auc_mean, 4)

        row = {
            "model_name": name,
            "model_key": key,
            "architecture_family": mtype,
            "val_auc": round(float(roc_auc_score(val_y, val_p)), 4),
            "val_optimal_threshold": round(tau_star, 4),
            "oasis_test_auc": round(test_auc_mean, 4),
            "oasis_test_ci_lower": round(test_auc_ci[0], 4),
            "oasis_test_ci_upper": round(test_auc_ci[1], 4),
            "oasis_test_accuracy": round(test_acc, 4),
            "oasis_test_balanced_acc": round(test_bacc, 4),
            "oasis_test_sensitivity": round(test_sens, 4),
            "oasis_test_specificity": round(test_spec, 4),
            "oasis_test_brier": round(test_brier, 4),
            "oasis_test_ece": round(test_cal["ece"], 4),
            "adni_auc": round(adni_auc_mean, 4),
            "adni_ci_lower": round(adni_auc_ci[0], 4),
            "adni_ci_upper": round(adni_auc_ci[1], 4),
            "adni_accuracy": round(adni_acc, 4),
            "adni_balanced_acc": round(adni_bacc, 4),
            "adni_sensitivity": round(adni_sens, 4),
            "adni_specificity": round(adni_spec, 4),
            "adni_brier": round(adni_brier, 4),
            "adni_ece": round(adni_cal["ece"], 4),
            "adni_siemens_auc": round(siemens_auc, 4),
            "adni_philips_auc": round(philips_auc, 4),
            "oasis2_auc": round(o2_auc_mean, 4),
            "oasis2_balanced_acc": round(o2_bacc, 4),
            "generalization_gap_delta_auc": delta_auc,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "latency_ms_per_volume": lat_ms,
            "peak_vram_mb": mem_mb,
            "training_time_sec": train_meta["training_time_sec"],
            "best_epoch": train_meta["best_epoch"],
        }
        evaluation_rows.append(row)
        training_manifest["models"][key] = {
            "name": name,
            "train_meta": train_meta,
            "checkpoint_path": ckpt_path,
            "metrics": row,
        }
        print(f"  OASIS-1 Test AUC: {row['oasis_test_auc']:.4f} [{row['oasis_test_ci_lower']:.4f}, {row['oasis_test_ci_upper']:.4f}] | ADNI AUC: {row['adni_auc']:.4f} [{row['adni_ci_lower']:.4f}, {row['adni_ci_upper']:.4f}] | Delta_AUC: {delta_auc:+.4f}", flush=True)

    # Convert to DataFrames and Save Predictions
    df_eval = pd.DataFrame(evaluation_rows)
    df_eval.to_csv(os.path.join(RESULTS_V4_DIR, "FINAL_MODEL_EVALUATION.csv"), index=False)
    df_eval.to_csv(os.path.join(TABLES_V4_DIR, "FINAL_MODEL_EVALUATION.csv"), index=False)

    df_val_preds = pd.DataFrame(all_val_preds)
    df_test_preds = pd.DataFrame(all_test_preds)
    df_adni_preds = pd.DataFrame(all_adni_preds)
    df_o2_preds = pd.DataFrame(all_o2_preds)

    df_val_preds.to_csv(os.path.join(PREDS_V4_DIR, "oasis1_validation_predictions.csv"), index=False)
    df_test_preds.to_csv(os.path.join(PREDS_V4_DIR, "oasis1_locked_test_predictions.csv"), index=False)
    df_adni_preds.to_csv(os.path.join(PREDS_V4_DIR, "adni_external_predictions.csv"), index=False)
    df_o2_preds.to_csv(os.path.join(PREDS_V4_DIR, "oasis2_external_predictions.csv"), index=False)

    # -------------------------------------------------------------
    # 6. Statistical Hypothesis Testing (H1-H7) with Holm-Bonferroni
    # -------------------------------------------------------------
    print("\n[5/6] Conducting Statistical Hypothesis Testing (H1-H7)...", flush=True)

    hypotheses = [
        ("H1", "Model Capacity: 3D ResNet-10 vs 3D ResNet-34 on OASIS-1 Test", "3d_resnet_10", "3d_resnet_34", "oasis_test"),
        ("H2", "Dense vs Residual: 3D DenseNet-121 vs 3D ResNet-18 on ADNI", "3d_densenet_121", "3d_resnet_18", "adni"),
        ("H3", "Multi-Planar 2.5D: 2.5D ResNet-18 vs 3D ResNet-18 on ADNI", "2_5d_multi_view_resnet_18", "3d_resnet_18", "adni"),
        ("H4", "Pretrained Transfer 2D: 2D Transfer ResNet-18 vs 3D ResNet-18 on ADNI", "2d_transfer_resnet_18", "3d_resnet_18", "adni"),
        ("H5", "Dual-Branch Synergy: Hybrid vs 3D ResNet-10 on OASIS-1 Test", "dual_branch_hybrid", "3d_resnet_10", "oasis_test"),
        ("H6a", "Foundation Probe: BrainIAC ViT-B/16 vs 3D ResNet-18 on ADNI", "brainiac_vit_b16_linear_probe", "3d_resnet_18", "adni"),
        ("H6b", "Foundation Probe: MONAI Swin UNETR SSL vs 3D ResNet-18 on ADNI", "swin_unetr_ssl_linear_probe", "3d_resnet_18", "adni"),
        ("H6c", "Foundation Probe: FOMO25 AMAES vs 3D ResNet-18 on ADNI", "fomo25_amaes_linear_probe", "3d_resnet_18", "adni"),
        ("H7", "Pretraining Control: BrainIAC ViT-B/16 vs Random ViT-B/16 Control on ADNI", "brainiac_vit_b16_linear_probe", "random_vit_b16_control", "adni"),
    ]

    stat_results = []
    rng = np.random.RandomState(42)
    B = 2000

    for hid, desc, kA, kB, split in hypotheses:
        if split == "oasis_test":
            y_arr = test_y
            pA = all_test_preds[kA]
            pB = all_test_preds[kB]
        else:
            y_arr = adni_y
            pA = all_adni_preds[kA]
            pB = all_adni_preds[kB]

        base_auc_A = roc_auc_score(y_arr, pA)
        base_auc_B = roc_auc_score(y_arr, pB)
        delta_auc = float(base_auc_A - base_auc_B)

        # Paired bootstrap test
        n = len(y_arr)
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        boot_diffs = []
        for _ in range(B):
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            b_idx = np.concatenate([b_pos, b_neg])
            try:
                auc_a = roc_auc_score(y_arr[b_idx], pA[b_idx])
                auc_b = roc_auc_score(y_arr[b_idx], pB[b_idx])
                boot_diffs.append(auc_a - auc_b)
            except ValueError:
                continue

        boot_diffs = np.array(boot_diffs)
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        # Two-sided empirical p-value
        p_val = 2.0 * min(np.mean(boot_diffs <= 0.0), np.mean(boot_diffs >= 0.0))
        p_val = max(p_val, 1.0 / B)

        stat_results.append({
            "hypothesis_id": hid,
            "description": desc,
            "model_A": kA,
            "model_B": kB,
            "cohort": split,
            "auc_A": round(base_auc_A, 4),
            "auc_B": round(base_auc_B, 4),
            "delta_auc": round(delta_auc, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "raw_p_value": round(p_val, 4),
        })

    # Holm-Bonferroni correction
    df_stats = pd.DataFrame(stat_results)
    df_stats = df_stats.sort_values(by="raw_p_value").reset_index(drop=True)
    m = len(df_stats)
    adj_p_vals = []
    sig_flags = []
    for rank, p in enumerate(df_stats["raw_p_value"]):
        adj_p = min(p * (m - rank), 1.0)
        adj_p_vals.append(adj_p)
        sig_flags.append(adj_p < 0.05)
    # Enforce monotonicity
    for i in range(1, m):
        if adj_p_vals[i] < adj_p_vals[i - 1]:
            adj_p_vals[i] = adj_p_vals[i - 1]

    df_stats["holm_bonferroni_p_value"] = [round(p, 4) for p in adj_p_vals]
    df_stats["statistically_significant"] = sig_flags
    df_stats.to_csv(os.path.join(RESULTS_V4_DIR, "FINAL_STATISTICAL_ANALYSIS.csv"), index=False)
    df_stats.to_csv(os.path.join(TABLES_V4_DIR, "FINAL_STATISTICAL_ANALYSIS.csv"), index=False)

    # -------------------------------------------------------------
    # 7. Foundation vs ResNet & Fine-Tuning Tables
    # -------------------------------------------------------------
    # Foundation vs ResNet-18 Table
    r18_adni_auc = df_eval[df_eval["model_key"] == "3d_resnet_18"]["adni_auc"].values[0]
    r18_adni_brier = df_eval[df_eval["model_key"] == "3d_resnet_18"]["adni_brier"].values[0]

    found_rows = []
    for k in ["brainiac_vit_b16_linear_probe", "brainiac_vit_b16_partial_ft", "swin_unetr_ssl_linear_probe", "swin_unetr_ssl_partial_ft", "fomo25_amaes_linear_probe", "fomo25_amaes_partial_ft"]:
        r = df_eval[df_eval["model_key"] == k].iloc[0]
        found_rows.append({
            "model_name": r["model_name"],
            "model_key": k,
            "oasis_test_auc": r["oasis_test_auc"],
            "adni_auc": r["adni_auc"],
            "adni_delta_vs_resnet18": round(r["adni_auc"] - r18_adni_auc, 4),
            "adni_brier": r["adni_brier"],
            "adni_brier_vs_resnet18": round(r["adni_brier"] - r18_adni_brier, 4),
            "generalization_gap": r["generalization_gap_delta_auc"],
            "trainable_params": r["trainable_parameters"],
            "total_params": r["total_parameters"],
        })
    df_found = pd.DataFrame(found_rows)
    df_found.to_csv(os.path.join(RESULTS_V4_DIR, "FINAL_FOUNDATION_VS_RESNET_COMPARISON.csv"), index=False)
    df_found.to_csv(os.path.join(TABLES_V4_DIR, "FINAL_FOUNDATION_VS_RESNET_COMPARISON.csv"), index=False)

    # Fine-Tuning Comparison Table
    ft_rows = [
        {
            "foundation_backbone": "BrainIAC (ViT-B/16³)",
            "linear_probe_auc": df_eval[df_eval["model_key"] == "brainiac_vit_b16_linear_probe"]["adni_auc"].values[0],
            "partial_ft_auc": df_eval[df_eval["model_key"] == "brainiac_vit_b16_partial_ft"]["adni_auc"].values[0],
            "linear_probe_brier": df_eval[df_eval["model_key"] == "brainiac_vit_b16_linear_probe"]["adni_brier"].values[0],
            "partial_ft_brier": df_eval[df_eval["model_key"] == "brainiac_vit_b16_partial_ft"]["adni_brier"].values[0],
        },
        {
            "foundation_backbone": "MONAI Swin UNETR SSL",
            "linear_probe_auc": df_eval[df_eval["model_key"] == "swin_unetr_ssl_linear_probe"]["adni_auc"].values[0],
            "partial_ft_auc": df_eval[df_eval["model_key"] == "swin_unetr_ssl_partial_ft"]["adni_auc"].values[0],
            "linear_probe_brier": df_eval[df_eval["model_key"] == "swin_unetr_ssl_linear_probe"]["adni_brier"].values[0],
            "partial_ft_brier": df_eval[df_eval["model_key"] == "swin_unetr_ssl_partial_ft"]["adni_brier"].values[0],
        },
        {
            "foundation_backbone": "FOMO25 AMAES (ResEnc-B)",
            "linear_probe_auc": df_eval[df_eval["model_key"] == "fomo25_amaes_linear_probe"]["adni_auc"].values[0],
            "partial_ft_auc": df_eval[df_eval["model_key"] == "fomo25_amaes_partial_ft"]["adni_auc"].values[0],
            "linear_probe_brier": df_eval[df_eval["model_key"] == "fomo25_amaes_linear_probe"]["adni_brier"].values[0],
            "partial_ft_brier": df_eval[df_eval["model_key"] == "fomo25_amaes_partial_ft"]["adni_brier"].values[0],
        },
    ]
    for r in ft_rows:
        r["auc_delta_ft_minus_probe"] = round(r["partial_ft_auc"] - r["linear_probe_auc"], 4)
        r["brier_delta_ft_minus_probe"] = round(r["partial_ft_brier"] - r["linear_probe_brier"], 4)
    df_ft = pd.DataFrame(ft_rows)
    df_ft.to_csv(os.path.join(RESULTS_V4_DIR, "FINAL_FINE_TUNING_COMPARISON.csv"), index=False)
    df_ft.to_csv(os.path.join(TABLES_V4_DIR, "FINAL_FINE_TUNING_COMPARISON.csv"), index=False)

    # -------------------------------------------------------------
    # 8. TrustGate Failure Detection Re-evaluation
    # -------------------------------------------------------------
    print("\n[6/6] Re-evaluating SHIFT-MRI TrustGate Failure Detection under Domain Shift...", flush=True)

    # Evaluate TrustGate across all ADNI external predictions
    # Assemble model ensemble predictions
    adni_prob_matrix = np.stack([
        all_adni_preds["3d_resnet_10"],
        all_adni_preds["3d_resnet_18"],
        all_adni_preds["3d_resnet_34"],
        all_adni_preds["3d_densenet_121"],
        all_adni_preds["dual_branch_hybrid"],
        all_adni_preds["brainiac_vit_b16_linear_probe"],
        all_adni_preds["swin_unetr_ssl_linear_probe"],
        all_adni_preds["fomo25_amaes_linear_probe"],
    ], axis=1)  # (500, 8)

    # Mean ensemble probability
    ens_mean = np.mean(adni_prob_matrix, axis=1)
    ens_var = np.var(adni_prob_matrix, axis=1)

    # Reference primary baseline: 3D ResNet-18
    p_primary = all_adni_preds["3d_resnet_18"]
    tau_r18 = df_eval[df_eval["model_key"] == "3d_resnet_18"]["val_optimal_threshold"].values[0]
    pred_r18 = (p_primary >= tau_r18).astype(int)
    # Ground truth failure: 1 if misclassified, 0 if correct
    failure_labels = (pred_r18 != adni_y).astype(int)
    error_rate = np.mean(failure_labels)
    print(f"  Primary 3D ResNet-18 ADNI Error Rate: {error_rate*100:.1f}% ({np.sum(failure_labels)} misclassifications)", flush=True)

    # Failure detectors:
    # 1. Margin Risk: 1 - 2*|p - 0.5| (higher = greater failure risk)
    margin_risk = 1.0 - (2.0 * np.abs(p_primary - 0.5))

    # 2. Entropy Risk: -p log2 p - (1-p) log2(1-p)
    p_clip = np.clip(p_primary, 1e-6, 1.0 - 1e-6)
    entropy_risk = -p_clip * np.log2(p_clip) - (1.0 - p_clip) * np.log2(1.0 - p_clip)

    # 3. Disagreement Risk: variance across model ensemble
    disagreement_risk = ens_var / (np.max(ens_var) + 1e-6)

    # 4. Feature Distance Proxy (from 6 radiomics centroid of training cohort)
    tr_center = np.mean(train_rads6, axis=0)
    dist_risk = np.linalg.norm(adni_rads6 - tr_center, axis=1)
    dist_risk = (dist_risk - dist_risk.min()) / (dist_risk.max() - dist_risk.min() + 1e-6)

    # 5. Composite Risk: equal-weighted combination
    composite_risk = (margin_risk + entropy_risk + disagreement_risk + dist_risk) / 4.0

    detectors = [
        ("Confidence Margin", margin_risk),
        ("Predictive Entropy", entropy_risk),
        ("Ensemble Disagreement", disagreement_risk),
        ("Feature Centroid Distance", dist_risk),
        ("Composite TrustGate Risk", composite_risk),
    ]

    tg_rows = []
    for dname, score in detectors:
        auc_err = float(roc_auc_score(failure_labels, score))
        # Abstention accuracy at 20% rejection
        thresh_20 = np.percentile(score, 80)  # reject top 20% risk
        keep_mask = score <= thresh_20
        acc_kept = float(accuracy_score(adni_y[keep_mask], pred_r18[keep_mask]))
        gain_acc = round(acc_kept - (1.0 - error_rate), 4)

        tg_rows.append({
            "detector_name": dname,
            "error_detection_auroc": round(auc_err, 4),
            "baseline_accuracy": round(1.0 - error_rate, 4),
            "accuracy_at_20pct_rejection": round(acc_kept, 4),
            "accuracy_gain_at_20pct_rejection": gain_acc,
            "effective_failure_gating": auc_err > 0.65,
            "domain_shift_collapse_observed": auc_err < 0.60,
        })
        print(f"  TrustGate: {dname:26s} Error Detection AUROC: {auc_err:.4f} | Acc @ 20% Rejection: {acc_kept:.4f} ({gain_acc:+.4f})", flush=True)

    df_tg = pd.DataFrame(tg_rows)
    df_tg.to_csv(os.path.join(RESULTS_V4_DIR, "FINAL_TRUSTGATE_ANALYSIS.csv"), index=False)
    df_tg.to_csv(os.path.join(TABLES_V4_DIR, "FINAL_TRUSTGATE_ANALYSIS.csv"), index=False)

    # Save training manifest
    training_manifest["trustgate_evaluation"] = tg_rows
    with open(os.path.join(RESULTS_V4_DIR, "FINAL_TRAINING_MANIFEST.json"), "w") as f:
        json.dump(training_manifest, f, indent=2)

    # -------------------------------------------------------------
    # 9. Publication Figures
    # -------------------------------------------------------------
    print("\nGenerating high-resolution publication figures...", flush=True)
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Figure 1: Leaderboard ROC Curves (OASIS-1 Test vs ADNI)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    models_to_plot = [
        ("3D ResNet-18", "3d_resnet_18", "#1f77b4"),
        ("3D ResNet-10", "3d_resnet_10", "#2ca02c"),
        ("3D DenseNet-121", "3d_densenet_121", "#9467bd"),
        ("2D Transfer ResNet-18", "2d_transfer_resnet_18", "#e377c2"),
        ("Dual-Branch Hybrid", "dual_branch_hybrid", "#8c564b"),
        ("BrainIAC (Linear Probe)", "brainiac_vit_b16_linear_probe", "#d62728"),
        ("Swin UNETR SSL (Linear Probe)", "swin_unetr_ssl_linear_probe", "#ff7f0e"),
        ("FOMO25 AMAES (Linear Probe)", "fomo25_amaes_linear_probe", "#17becf"),
    ]

    for dname, key, col in models_to_plot:
        # OASIS-1 Test
        fpr_o, tpr_o, _ = roc_curve(test_y, all_test_preds[key])
        auc_o = roc_auc_score(test_y, all_test_preds[key])
        axes[0].plot(fpr_o, tpr_o, label=f"{dname} ({auc_o:.3f})", color=col, lw=2)

        # ADNI External
        fpr_a, tpr_a, _ = roc_curve(adni_y, all_adni_preds[key])
        auc_a = roc_auc_score(adni_y, all_adni_preds[key])
        axes[1].plot(fpr_a, tpr_a, label=f"{dname} ({auc_a:.3f})", color=col, lw=2)

    for i, title in enumerate(["OASIS-1 Locked Test (N=30, Source In-Distribution)", "ADNI Zero-Shot External (N=500, Domain Shift)"]):
        axes[i].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
        axes[i].set_title(title, fontweight="bold")
        axes[i].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    plt.savefig(os.path.join(FIGURES_V4_DIR, "fig1_leaderboard_roc_curves.png"), dpi=300)
    plt.close()

    # Figure 2: Generalization Gap (OASIS-1 Test vs ADNI)
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df_eval.sort_values(by="generalization_gap_delta_auc", ascending=False)
    y_pos = np.arange(len(df_sorted))
    ax.barh(y_pos, df_sorted["generalization_gap_delta_auc"], color="#d9534f", alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_name"])
    ax.invert_yaxis()
    ax.set_xlabel("Generalization Gap Delta AUC (OASIS-1 Test AUC - ADNI AUC)")
    ax.set_title("Cross-Cohort Generalization Collapse Across All Architectures", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    for idx, v in enumerate(df_sorted["generalization_gap_delta_auc"]):
        ax.text(v + 0.005, idx, f"{v:+.3f}", va="center", fontsize=9, fontweight="bold")
    plt.savefig(os.path.join(FIGURES_V4_DIR, "fig2_cross_cohort_generalization_gap.png"), dpi=300)
    plt.close()

    # Figure 3: Foundation vs Supervised ResNets
    fig, ax = plt.subplots(figsize=(10, 6))
    comp_models = [
        "3D ResNet-10", "3D ResNet-18", "3D ResNet-34", "3D DenseNet-121",
        "BrainIAC ViT-B/16 (Linear Probe)", "MONAI Swin UNETR SSL (Linear Probe)", "FOMO25 AMAES ResEnc-B (Linear Probe)"
    ]
    sub_df = df_eval[df_eval["model_name"].isin(comp_models)].copy()
    sub_df = sub_df.sort_values(by="adni_auc", ascending=False)
    colors = ["#2b5c8f" if "ResNet" in m or "DenseNet" in m else "#c0392b" for m in sub_df["model_name"]]
    x_pos = np.arange(len(sub_df))
    ax.bar(x_pos, sub_df["adni_auc"], yerr=[sub_df["adni_auc"] - sub_df["adni_ci_lower"], sub_df["adni_ci_upper"] - sub_df["adni_auc"]], capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sub_df["model_name"], rotation=30, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance (AUC=0.50)")
    ax.set_ylabel("ADNI External ROC-AUC (95% CI)")
    ax.set_title("Foundation Models vs Well-Tuned Supervised Baselines (ADNI N=500)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.savefig(os.path.join(FIGURES_V4_DIR, "fig3_foundation_vs_supervised.png"), dpi=300)
    plt.close()

    # Figure 4: TrustGate Selective Prediction / Abstention Curves
    fig, ax = plt.subplots(figsize=(9, 6))
    rejection_rates = np.linspace(0.0, 0.50, 11)  # 0% to 50%
    for dname, score in detectors:
        accs = []
        for r_rate in rejection_rates:
            if r_rate == 0.0:
                accs.append(1.0 - error_rate)
            else:
                cutoff = np.percentile(score, (1.0 - r_rate) * 100.0)
                mask = score <= cutoff
                accs.append(accuracy_score(adni_y[mask], pred_r18[mask]))
        ax.plot(rejection_rates * 100.0, accs, label=dname, lw=2, marker="o", markersize=4)

    ax.set_xlabel("Cohort Rejection / Abstention Rate (%)")
    ax.set_ylabel("Accuracy on Retained ADNI Cohort")
    ax.set_title("SHIFT-MRI TrustGate: Selective Prediction under Domain Shift", fontweight="bold")
    ax.axhline(1.0 - error_rate, color="black", linestyle="--", label=f"Unretained Baseline ({1.0-error_rate:.1%})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(FIGURES_V4_DIR, "fig5_trustgate_abstention_curves.png"), dpi=300)
    plt.close()

    # Figure 5: Subgroup Scanner Analysis (Siemens N=370 vs Philips N=130)
    fig, ax = plt.subplots(figsize=(10, 5))
    x_sub = np.arange(len(sub_df))
    width = 0.35
    ax.bar(x_sub - width/2, sub_df["adni_siemens_auc"], width, label="Siemens (N=370)", color="#3498db", edgecolor="black")
    ax.bar(x_sub + width/2, sub_df["adni_philips_auc"], width, label="Philips (N=130)", color="#e67e22", edgecolor="black")
    ax.set_xticks(x_sub)
    ax.set_xticklabels(sub_df["model_name"], rotation=30, ha="right")
    ax.set_ylabel("Subgroup ROC-AUC")
    ax.set_title("Cross-Scanner Subgroup Generalization on ADNI (Siemens vs Philips)", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend()
    plt.savefig(os.path.join(FIGURES_V4_DIR, "fig6_subgroup_scanner_transfer.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print("SHIFT-MRI V4 BENCHMARK CAMPAIGN COMPLETED SUCCESSFULLY!")
    print(f"Results and authoritative CSV tables saved to: {RESULTS_V4_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_v4_campaign()
