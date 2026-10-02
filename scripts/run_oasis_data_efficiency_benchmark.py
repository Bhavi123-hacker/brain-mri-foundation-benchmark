"""
Data-Efficiency Benchmark Execution Script (Priority 1).
Evaluates performance scaling across 10%, 25%, 50%, and 100% labeled training data fractions:
  - 10%:  4 subjects  (~11.4% actual)
  - 25%:  9 subjects  (~25.7% actual)
  - 50%: 18 subjects  (~51.4% actual)
  - 100%: 35 subjects (100.0% actual)
Fixed Evaluation Pools:
  - Validation: 7 subjects (4 CN, 3 Dementia) - FIXED
  - Locked Test: 8 subjects (4 CN, 4 Dementia) - FIXED, EVALUATED STRICTLY ONCE PER RUN
Models Evaluated:
  1. Supervised 3D ResNet-18 Baseline (lr=3e-4, wd=1e-5)
  2. BrainIAC Linear Probe (frozen backbone, trainable head)
  3. FOMO25 Winner Linear Probe (frozen backbone, trainable head)
  4. MONAI Swin UNETR SSL Linear Probe (frozen backbone, trainable head)
Random Seeds: 42, 43, 44, 45, 46
Outputs:
  - results/extensions/data_efficiency/oasis_data_efficiency_results.csv
  - results/extensions/data_efficiency/oasis_data_efficiency_results.json
  - results/extensions/data_efficiency_leaderboard.csv
  - Figures DE1-DE5 in results/extensions/figures/
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from utils.seed import set_seed
from models.baseline.resnet3d import build_resnet18_3d
from models.foundation_brainiac.adapter import BrainIACAdapter
from models.foundation_fomo25.adapter import FOMO25Adapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from preprocessing.cache_manager import PreprocessedCacheManager
from evaluation.metrics import compute_classification_metrics


def load_oasis_scans_and_metadata(splits_csv: str = "D:/OASIS1_PILOT/pilot_splits.csv", cache_dir: str = "data/cache/oasis1_pilot"):
    """Load all 50 preprocessed OASIS-1 scans into memory."""
    splits_df = pd.read_csv(splits_csv)
    splits_df = splits_df[~splits_df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)

    cache_mgr = PreprocessedCacheManager(cache_dir=cache_dir)
    scans_dict = {}
    labels_dict = {}

    for _, row in splits_df.iterrows():
        sid = row["ID"]
        fpath = os.path.join("D:/OASIS1_PILOT", row["mri_file"])
        tensor = cache_mgr.get(fpath, transform_id="oasis1_zscore_128")
        if tensor is None:
            raise FileNotFoundError(f"Missing preprocessed tensor for {sid} ({fpath})")
        scans_dict[sid] = tensor.to(torch.float32)
        labels_dict[sid] = float(row["label"])

    return scans_dict, labels_dict, splits_df


def train_linear_probe_run(
    model,
    model_name: str,
    slug: str,
    frac_key: str,
    seed: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    device: str,
    max_epochs: int = 30,
    patience: int = 7,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
):
    set_seed(seed)
    for layer in model.head.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    model.head.to(device)
    effective_bs = min(batch_size, len(train_feats))

    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=effective_bs, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=effective_bs, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_cnt = 0
    best_head_state = None

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.head.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model.head(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.head.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model.head(x)
                loss = criterion(out, y)
                val_losses.append(loss.item())

        ep_val_loss = float(np.mean(val_losses))
        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_epoch = epoch
            patience_cnt = 0
            best_head_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    train_duration = time.time() - t0

    # Restore best head and evaluate on locked test set
    model.head.load_state_dict(best_head_state)
    model.head.eval()

    t_inf0 = time.time()
    with torch.no_grad():
        test_logits = model.head(test_feats.to(device))
        test_probs = torch.sigmoid(test_logits).cpu().squeeze(-1).numpy()
    inf_duration = time.time() - t_inf0

    y_test = test_labels.cpu().squeeze(-1).numpy()
    metrics = compute_classification_metrics(y_test, test_probs)

    return {
        "model": model_name,
        "slug": slug,
        "paradigm": "foundation_linear_probe",
        "fraction": frac_key,
        "n_train": len(train_feats),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "test_roc_auc": metrics["roc_auc"],
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_specificity": metrics["specificity"],
        "test_f1": metrics["f1"],
        "test_brier_score": metrics["brier_score"],
        "training_time_seconds": round(train_duration, 4),
        "inference_time_seconds": round(inf_duration, 4),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def train_resnet18_data_efficiency_run(
    frac_key: str,
    seed: int,
    train_tensors: torch.Tensor,
    train_labels: torch.Tensor,
    val_tensors: torch.Tensor,
    val_labels: torch.Tensor,
    test_tensors: torch.Tensor,
    test_labels: torch.Tensor,
    device: str,
    max_epochs: int = 12,
    patience: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 4,
):
    set_seed(seed)
    model = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2).to(device)

    effective_bs = min(batch_size, len(train_tensors))
    train_ds = TensorDataset(train_tensors, train_labels)
    val_ds = TensorDataset(val_tensors, val_labels)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=effective_bs, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=effective_bs, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_cnt = 0
    best_model_state = None

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_losses.append(loss.item())

        ep_val_loss = float(np.mean(val_losses))
        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_epoch = epoch
            patience_cnt = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    train_duration = time.time() - t0

    # Restore best model state and evaluate on locked test set
    model.load_state_dict(best_model_state)
    model.eval()

    t_inf0 = time.time()
    with torch.no_grad():
        test_logits = model(test_tensors.to(device))
        test_probs = torch.sigmoid(test_logits).cpu().squeeze(-1).numpy()
    inf_duration = time.time() - t_inf0

    y_test = test_labels.cpu().squeeze(-1).numpy()
    metrics = compute_classification_metrics(y_test, test_probs)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model": "Supervised 3D ResNet-18",
        "slug": "resnet18",
        "paradigm": "supervised_baseline",
        "fraction": frac_key,
        "n_train": len(train_tensors),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "test_roc_auc": metrics["roc_auc"],
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_specificity": metrics["specificity"],
        "test_f1": metrics["f1"],
        "test_brier_score": metrics["brier_score"],
        "training_time_seconds": round(train_duration, 2),
        "inference_time_seconds": round(inf_duration, 4),
        "total_parameters": total_p,
        "trainable_parameters": train_p,
    }


def run_oasis_data_efficiency_benchmark(
    splits_csv: str = "D:/OASIS1_PILOT/pilot_splits.csv",
    fractions_csv: str = "results/extensions/data_efficiency/oasis_data_fraction_splits.csv",
    output_dir: str = "results/extensions/data_efficiency",
    seeds: list = [42, 43, 44, 45, 46],
):
    logger = get_logger("OASISDataEfficiencyBenchmark")
    logger.info("=================================================================")
    logger.info("Starting Priority 1: OASIS Data-Efficiency Benchmark")
    logger.info("Training fractions: 10% (N=4), 25% (N=9), 50% (N=18), 100% (N=35)")
    logger.info(f"Random Seeds: {seeds}")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Execution Device: {device}")

    # 1. Load scans into RAM
    scans_dict, labels_dict, splits_df = load_oasis_scans_and_metadata(splits_csv)
    val_sids = splits_df[splits_df["split"] == "val"]["ID"].tolist()
    test_sids = splits_df[splits_df["split"] == "test"]["ID"].tolist()

    val_tensors = torch.stack([scans_dict[sid] for sid in val_sids])
    val_labels = torch.tensor([[labels_dict[sid]] for sid in val_sids], dtype=torch.float32)

    test_tensors = torch.stack([scans_dict[sid] for sid in test_sids])
    test_labels = torch.tensor([[labels_dict[sid]] for sid in test_sids], dtype=torch.float32)

    all_train_sids = splits_df[splits_df["split"] == "train"]["ID"].tolist()
    all_train_tensors = torch.stack([scans_dict[sid] for sid in all_train_sids])
    all_train_labels = torch.tensor([[labels_dict[sid]] for sid in all_train_sids], dtype=torch.float32)

    frac_splits_df = pd.read_csv(fractions_csv)
    fractions = ["10%", "25%", "50%", "100%"]

    all_results = []

    # -------------------------------------------------------------------------
    # PART A: FOUNDATION MODEL LINEAR PROBES
    # -------------------------------------------------------------------------
    foundation_models = [
        ("BrainIAC", "brainiac", lambda: BrainIACAdapter(output_dim=1, weights_path="weights/brainiac_vit_base.pt", adaptation_mode="linear_probe", dropout=0.2)),
        ("FOMO25 Winner", "fomo25", lambda: FOMO25Adapter(output_dim=1, weights_path="weights/fomo25_winner_encoder.pt", adaptation_mode="linear_probe", dropout=0.2)),
        ("MONAI Swin UNETR SSL", "swin_unetr", lambda: SwinUNETRSSLAdapter(output_dim=1, weights_path="weights/model_swinvit.pt", adaptation_mode="linear_probe", dropout=0.2)),
    ]

    for model_name, slug, builder in foundation_models:
        logger.info(f"\n--- Extracting Features for {model_name} ---")
        model = builder().to(device)
        model.freeze_backbone()
        model.eval()

        with torch.no_grad():
            train_feats_dict = {}
            for sid in all_train_sids:
                train_feats_dict[sid] = model.extract_features(scans_dict[sid].unsqueeze(0).to(device)).cpu()
            val_feats = torch.cat([model.extract_features(scans_dict[sid].unsqueeze(0).to(device)).cpu() for sid in val_sids], dim=0)
            test_feats = torch.cat([model.extract_features(scans_dict[sid].unsqueeze(0).to(device)).cpu() for sid in test_sids], dim=0)

        for frac in fractions:
            for s in seeds:
                sample_sids = frac_splits_df[(frac_splits_df["fraction"] == frac) & (frac_splits_df["seed"] == s)]["subject_id"].tolist()
                frac_train_feats = torch.cat([train_feats_dict[sid] for sid in sample_sids], dim=0)
                frac_train_labels = torch.tensor([[labels_dict[sid]] for sid in sample_sids], dtype=torch.float32)

                res = train_linear_probe_run(
                    model=model,
                    model_name=model_name,
                    slug=slug,
                    frac_key=frac,
                    seed=s,
                    train_feats=frac_train_feats,
                    train_labels=frac_train_labels,
                    val_feats=val_feats,
                    val_labels=val_labels,
                    test_feats=test_feats,
                    test_labels=test_labels,
                    device=device,
                    max_epochs=30,
                    patience=7,
                )
                all_results.append(res)
                logger.info(f"[{model_name}] Frac: {frac:4s} | Seed: {s:2d} | N={res['n_train']:2d} -> Test AUC: {res['test_roc_auc']:.4f} | Acc: {res['test_accuracy']:.4f} | F1: {res['test_f1']:.4f}")

    # -------------------------------------------------------------------------
    # PART B: SUPERVISED 3D RESNET-18 BASELINE
    # -------------------------------------------------------------------------
    logger.info(f"\n--- Training Supervised 3D ResNet-18 Across Data Fractions ---")
    for frac in fractions:
        for s in seeds:
            sample_sids = frac_splits_df[(frac_splits_df["fraction"] == frac) & (frac_splits_df["seed"] == s)]["subject_id"].tolist()
            frac_train_tensors = torch.stack([scans_dict[sid] for sid in sample_sids])
            frac_train_labels = torch.tensor([[labels_dict[sid]] for sid in sample_sids], dtype=torch.float32)

            res = train_resnet18_data_efficiency_run(
                frac_key=frac,
                seed=s,
                train_tensors=frac_train_tensors,
                train_labels=frac_train_labels,
                val_tensors=val_tensors,
                val_labels=val_labels,
                test_tensors=test_tensors,
                test_labels=test_labels,
                device=device,
                max_epochs=12,
                patience=4,
                lr=3e-4,
                weight_decay=1e-5,
            )
            all_results.append(res)
            logger.info(f"[ResNet-18] Frac: {frac:4s} | Seed: {s:2d} | N={res['n_train']:2d} -> Test AUC: {res['test_roc_auc']:.4f} | Acc: {res['test_accuracy']:.4f} | F1: {res['test_f1']:.4f} (Time: {res['training_time_seconds']}s)")

    # -------------------------------------------------------------------------
    # SAVE RAW RESULTS
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_dir, "oasis_data_efficiency_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"\nSaved data-efficiency results to {results_csv_path}")

    results_json_path = os.path.join(output_dir, "oasis_data_efficiency_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved data-efficiency JSON to {results_json_path}")

    # -------------------------------------------------------------------------
    # COMPOSE DATA-EFFICIENCY LEADERBOARD
    # -------------------------------------------------------------------------
    leaderboard_rows = []
    for model_name in ["Supervised 3D ResNet-18", "BrainIAC", "FOMO25 Winner", "MONAI Swin UNETR SSL"]:
        m_df = results_df[results_df["model"] == model_name]
        row = {
            "Model": model_name,
            "Paradigm": m_df["paradigm"].iloc[0],
            "Trainable_Params": m_df["trainable_parameters"].iloc[0],
        }
        for frac in fractions:
            sub = m_df[m_df["fraction"] == frac]["test_roc_auc"]
            row[f"{frac}_AUC_mean"] = round(float(sub.mean()), 4)
            row[f"{frac}_AUC_std"] = round(float(sub.std()), 4)

        # Performance gain from 10% to 100%
        auc_10 = row["10%_AUC_mean"]
        auc_100 = row["100%_AUC_mean"]
        row["Gain_10_to_100"] = round(auc_100 - auc_10, 4)

        # Average across all fractions
        row["Overall_Avg_AUC"] = round(float(m_df["test_roc_auc"].mean()), 4)
        row["Overall_Avg_Acc"] = round(float(m_df["test_accuracy"].mean()), 4)
        row["Overall_Avg_F1"] = round(float(m_df["test_f1"].mean()), 4)
        leaderboard_rows.append(row)

    lead_df = pd.DataFrame(leaderboard_rows).sort_values(by="Overall_Avg_AUC", ascending=False)
    lead_csv_path = os.path.join(os.path.dirname(output_dir), "data_efficiency_leaderboard.csv")
    lead_df.to_csv(lead_csv_path, index=False)
    logger.info(f"Saved data-efficiency leaderboard to {lead_csv_path}")

    # Also save copy in data_efficiency subfolder
    sub_lead_csv_path = os.path.join(output_dir, "data_efficiency_leaderboard.csv")
    lead_df.to_csv(sub_lead_csv_path, index=False)

    return results_df, lead_df

if __name__ == "__main__":
    run_oasis_data_efficiency_benchmark()
