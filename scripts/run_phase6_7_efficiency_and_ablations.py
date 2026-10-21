"""
Phase 6 & 7: Label Efficiency and Preprocessing Ablation Suite.

Phase 6: Multi-Paradigm Label-Efficiency Benchmark
Evaluates sample efficiency across data budgets: 10% (N=14), 25% (N=34), 50% (N=69), 100% (N=138).
Compares:
1. Supervised 3D ResNet-18
2. 2D Pretrained Transfer Learning (ImageNet ResNet-18)
3. Classical Machine Learning (Radiomics L2 Logistic Regression)
4. MONAI Swin UNETR SSL Foundation Model (Linear Probe)

Phase 7: Preprocessing & Masking Ablations
Evaluates downstream performance under:
1. Standard Non-zero Z-Score (Baseline)
2. Min-Max Normalization [0, 1]
3. Mask Erosion (Peripheral Margin Zeroing)
4. Analyzes the false-positive suppression mechanism behind the mask-erosion AUC shift.
"""

import os
import sys
import time
import json
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_classification_metrics
from models.baseline.resnet3d import build_resnet18_3d, build_resnet34_3d
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from scripts.run_master_benchmark_campaign import OASISPreprocessedDataset

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
SWIN_WEIGHTS_PATH = r"D:\ml project\weights\model_swinvit.pt"

RUNS_DIR = r"D:\brain_mri_data\benchmark_runs"
TABLES_DIR = os.path.join(RUNS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_stratified_subsets(train_df, fractions=[0.10, 0.25, 0.50, 1.00], seed=42):
    """Generates stratified subsets preserving class balance."""
    cn_df = train_df[train_df["binary_label"] == 0.0].sample(frac=1.0, random_state=seed)
    ad_df = train_df[train_df["binary_label"] == 1.0].sample(frac=1.0, random_state=seed)

    subsets = {}
    for frac in fractions:
        n_cn = max(2, int(round(len(cn_df) * frac)))
        n_ad = max(2, int(round(len(ad_df) * frac)))
        sub_df = pd.concat([cn_df.iloc[:n_cn], ad_df.iloc[:n_ad]]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        subsets[int(frac * 100)] = sub_df
    return subsets


def run_phase6_label_efficiency():
    print("=" * 70)
    print("Phase 6: Multi-Paradigm Label-Efficiency Benchmark across Budgets")
    print("=" * 70)

    splits_df = pd.read_csv(OASIS_SPLITS_CSV)
    if "binary_label" not in splits_df.columns:
        splits_df["binary_label"] = (splits_df["cdr"] > 0.0).astype(float)
    train_df = splits_df[splits_df["split"] == "train"]
    val_df = splits_df[splits_df["split"] == "val"]

    budgets = get_stratified_subsets(train_df, fractions=[0.10, 0.25, 0.50, 1.00])
    val_ds = OASISPreprocessedDataset(val_df, OASIS_TENSORS_DIR, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # 1. Load cached radiomics features if available
    radiomics_cache = os.path.join(TABLES_DIR, "extracted_radiomics_features.pkl")
    has_radiomics = os.path.exists(radiomics_cache)
    if has_radiomics:
        with open(radiomics_cache, "rb") as f:
            rad_data = pickle.load(f)
        X_train_full, y_train_full, ids_train_full = rad_data["train"]
        X_val, y_val, ids_val = rad_data["val"]
        id_to_idx = {sid: i for i, sid in enumerate(ids_train_full)}

    results = {
        "budgets": [10, 25, 50, 100],
        "sample_counts": {b: len(sub_df) for b, sub_df in budgets.items()},
        "paradigms": {
            "3D_ResNet18": {},
            "Classical_Radiomics_LR": {},
            "MONAI_SwinUNETR_LinearProbe": {}
        }
    }

    for b, sub_df in budgets.items():
        n_samples = len(sub_df)
        print(f"\n--- Evaluating Budget {b}% (N={n_samples} subjects) ---")

        # A. Classical Radiomics L2 LR
        if has_radiomics:
            sub_indices = [id_to_idx[sid] for sid in sub_df["session_id"] if sid in id_to_idx]
            X_sub = X_train_full[sub_indices]
            y_sub = y_train_full[sub_indices]
            scaler = StandardScaler()
            X_sub_scaled = scaler.fit_transform(X_sub)
            X_val_scaled = scaler.transform(X_val)

            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_sub_scaled, y_sub)
            p_val = clf.predict_proba(X_val_scaled)[:, 1]
            m_rad = compute_classification_metrics(y_val, p_val)
            results["paradigms"]["Classical_Radiomics_LR"][b] = m_rad
            print(f"  Radiomics LR (N={n_samples}): Val AUC = {m_rad['roc_auc']:.4f}, BalAcc = {m_rad['balanced_accuracy']:.4f}")

        # B. MONAI Swin UNETR Linear Probe (Fast feature adaptation)
        sub_ds = OASISPreprocessedDataset(sub_df, OASIS_TENSORS_DIR, is_train=True)
        sub_loader = DataLoader(sub_ds, batch_size=4, shuffle=True)
        
        swin_adapter = SwinUNETRSSLAdapter(
            weights_path=SWIN_WEIGHTS_PATH,
            adaptation_mode="linear_probe",
            dropout=0.2
        ).to(DEVICE)
        opt_swin = AdamW(swin_adapter.head.parameters(), lr=2e-3, weight_decay=1e-3)
        use_cuda = (DEVICE.type == "cuda")

        for _ in range(8): # Quick 8 epochs adaptation
            swin_adapter.train()
            for imgs, tgts, _ in sub_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                opt_swin.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    out = swin_adapter(imgs)
                    loss = F.binary_cross_entropy_with_logits(out, tgts)
                loss.backward()
                opt_swin.step()

        swin_adapter.eval()
        swin_probs, swin_targets = [], []
        with torch.no_grad():
            for imgs, tgts, _ in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    out = swin_adapter(imgs)
                    p = torch.sigmoid(out).cpu().numpy().flatten()
                swin_probs.extend(p)
                swin_targets.extend(tgts.numpy().flatten())

        m_swin = compute_classification_metrics(np.array(swin_targets), np.array(swin_probs))
        results["paradigms"]["MONAI_SwinUNETR_LinearProbe"][b] = m_swin
        print(f"  Swin UNETR LP (N={n_samples}): Val AUC = {m_swin['roc_auc']:.4f}, BalAcc = {m_swin['balanced_accuracy']:.4f}")

        # C. 3D ResNet-18 (Supervised Baseline)
        r18 = build_resnet18_3d(output_dim=1, dropout=0.3).to(DEVICE)
        opt_r18 = AdamW(r18.parameters(), lr=2e-4, weight_decay=1e-3)
        scaler_r18 = torch.amp.GradScaler(device="cuda", enabled=use_cuda)

        for _ in range(8):
            r18.train()
            for imgs, tgts, _ in sub_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                opt_r18.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    out = r18(imgs)
                    loss = F.binary_cross_entropy_with_logits(out, tgts)
                scaler_r18.scale(loss).backward()
                scaler_r18.step(opt_r18)
                scaler_r18.update()

        r18.eval()
        r18_probs, r18_targets = [], []
        with torch.no_grad():
            for imgs, tgts, _ in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    out = r18(imgs)
                    p = torch.sigmoid(out).cpu().numpy().flatten()
                r18_probs.extend(p)
                r18_targets.extend(tgts.numpy().flatten())

        m_r18 = compute_classification_metrics(np.array(r18_targets), np.array(r18_probs))
        results["paradigms"]["3D_ResNet18"][b] = m_r18
        print(f"  3D ResNet-18  (N={n_samples}): Val AUC = {m_r18['roc_auc']:.4f}, BalAcc = {m_r18['balanced_accuracy']:.4f}")

    out_file = os.path.join(TABLES_DIR, "phase6_multiparadigm_label_efficiency_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPhase 6 results saved to {out_file}")
    return results


def run_phase7_preprocessing_ablations():
    print("=" * 70)
    print("Phase 7: Preprocessing Robustness & Masking Ablations")
    print("=" * 70)

    splits_df = pd.read_csv(OASIS_SPLITS_CSV)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)

    # Load baseline ResNet-18 and ResNet-34 checkpoints
    r18_ckpt = os.path.join(RUNS_DIR, "checkpoints", "resnet18_seed42_best.pt")
    r34_ckpt = os.path.join(RUNS_DIR, "checkpoints", "resnet34_seed42_best.pt")

    r18 = build_resnet18_3d(output_dim=1).to(DEVICE)
    r18.load_state_dict(torch.load(r18_ckpt, map_location=DEVICE, weights_only=False))
    r18.eval()

    r34 = build_resnet34_3d(output_dim=1).to(DEVICE)
    r34.load_state_dict(torch.load(r34_ckpt, map_location=DEVICE, weights_only=False))
    r34.eval()

    targets = (val_df["cdr"] > 0.0).astype(float).values

    perturbations = ["baseline", "minmax_norm", "mask_erosion", "gaussian_blur"]
    ablation_results = {}

    for pert in perturbations:
        probs_r18, probs_r34 = [], []
        for _, row in val_df.iterrows():
            sid = str(row["session_id"]).strip()
            arr = np.load(os.path.join(OASIS_TENSORS_DIR, f"{sid}.npy"))
            t = torch.from_numpy(arr).float()

            if pert == "minmax_norm":
                # Rescale non-zero tissue to [0, 1]
                mask = torch.abs(t) > 1e-4
                t_min, t_max = t[mask].min(), t[mask].max()
                t_norm = torch.zeros_like(t)
                t_norm[mask] = (t[mask] - t_min) / (t_max - t_min + 1e-6)
                t = t_norm
            elif pert == "mask_erosion":
                # Zero out peripheral boundary voxels (t < 0.1)
                t[torch.abs(t) < 0.1] = 0.0
            elif pert == "gaussian_blur":
                # Slight smoothing
                kernel = torch.ones((1, 1, 3, 3, 3)) / 27.0
                t = F.conv3d(t.unsqueeze(0), kernel, padding=1).squeeze(0)

            t = t.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    p18 = torch.sigmoid(r18(t)).item()
                    p34 = torch.sigmoid(r34(t)).item()
            probs_r18.append(p18)
            probs_r34.append(p34)

        m18 = compute_classification_metrics(targets, np.array(probs_r18))
        m34 = compute_classification_metrics(targets, np.array(probs_r34))

        # Inspect mean probability shift for CN vs AD
        cn_mask = (targets == 0.0)
        ad_mask = (targets == 1.0)
        cn_p18 = float(np.mean(np.array(probs_r18)[cn_mask]))
        ad_p18 = float(np.mean(np.array(probs_r18)[ad_mask]))
        cn_p34 = float(np.mean(np.array(probs_r34)[cn_mask]))
        ad_p34 = float(np.mean(np.array(probs_r34)[ad_mask]))

        ablation_results[pert] = {
            "ResNet-18": {
                **m18,
                "cn_mean_prob": round(cn_p18, 4),
                "ad_mean_prob": round(ad_p18, 4),
                "prob_spread": round(ad_p18 - cn_p18, 4)
            },
            "ResNet-34": {
                **m34,
                "cn_mean_prob": round(cn_p34, 4),
                "ad_mean_prob": round(ad_p34, 4),
                "prob_spread": round(ad_p34 - cn_p34, 4)
            }
        }
        print(f"[{pert:15s}] R18 AUC: {m18['roc_auc']:.4f} (CN prob: {cn_p18:.3f}, AD prob: {ad_p18:.3f}) | R34 AUC: {m34['roc_auc']:.4f} (CN prob: {cn_p34:.3f}, AD prob: {ad_p34:.3f})")

    out_file = os.path.join(TABLES_DIR, "phase7_preprocessing_ablation_results.json")
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Phase 7 results saved to {out_file}")
    return ablation_results


if __name__ == "__main__":
    run_phase6_label_efficiency()
    run_phase7_preprocessing_ablations()
