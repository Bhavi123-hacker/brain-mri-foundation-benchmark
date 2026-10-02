#!/usr/bin/env python3
"""
Targeted Foundation Model Adaptation & Improvement Campaign.
Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines.

Investigates downstream task adaptation strategies for:
1. BrainIAC (AIM-KannLab / Harvard BWH)
2. FOMO25 Winner (BrainFM4Challenges / CHUV / MICCAI 2025)
3. MONAI Swin UNETR SSL (Project-MONAI / Tang et al.)

Strategies evaluated:
- Strategy A: Linear Probe (Head-only control)
- Strategy B: Partial Fine-Tuning (Final representation/projection block + LayerNorm + head)
- Strategy C: Deeper Fine-Tuning (Final 25–50% of backbone + head)
- Strategy D: Full Fine-Tuning (Complete backbone + head)

Data Partitioning Protocol:
- Development & Tuning: Strictly OASIS-1 Train (N=35) and Val (N=7).
- Locked Holdout Evaluation: OASIS-1 Locked Test (N=8) and ADNI External Holdout (N=8)
  evaluated strictly AFTER development runs are frozen.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
import matplotlib.pyplot as plt

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.foundation_adaptation import build_adapted_foundation_model, build_discriminative_optimizer

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_campaign" / "foundation_adaptation"
CURVES_DIR = OUTPUT_DIR / "adaptation_curves"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CANONICAL_FRONTEND_DIR = PROJECT_ROOT / "results" / "final_campaign" / "canonical_frontend_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CURVES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(CANONICAL_FRONTEND_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_binary_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 5) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper if i < n_bins - 1 else y_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def compute_full_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute comprehensive diagnostic classification metrics."""
    y_preds = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_preds))
    bal_acc = float(balanced_accuracy_score(y_true, y_preds))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = 0.5

    brier = float(brier_score_loss(y_true, y_probs))
    ece = compute_binary_ece(y_true, y_probs, n_bins=5)

    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = sens
    total_failures = int(fp + fn)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "brier": brier,
        "ece": ece,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total_failures": total_failures,
    }


def load_oasis_cache_into_ram() -> Dict[str, torch.Tensor]:
    """Load all 50 preprocessed OASIS-1 pilot scans into RAM."""
    print("Loading OASIS-1 pilot volumes from cache into RAM...", flush=True)
    cache_dir = PROJECT_ROOT / "data" / "cache" / "oasis1_pilot"
    cache_map = {}
    for fname in os.listdir(cache_dir):
        if fname.endswith(".pt"):
            full_path = cache_dir / fname
            try:
                data = torch.load(full_path, map_location="cpu", weights_only=False)
                src = data.get("source_path")
                if src:
                    basename = os.path.basename(src)
                    cache_map[basename] = data["tensor"].to(torch.float32)
            except Exception:
                pass
    print(f"Loaded {len(cache_map)} OASIS-1 pilot volumes into RAM.", flush=True)
    return cache_map


def load_adni_test_into_ram() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the 8 ADNI external holdout scans and labels into RAM."""
    print("Loading ADNI external holdout scans into RAM...", flush=True)
    split_path = PROJECT_ROOT / "results" / "adni_cn_ad_split.csv"
    df = pd.read_csv(split_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    cache_dir = PROJECT_ROOT / "data" / "cache" / "adni_cn_ad"

    scans = []
    labels = []
    for _, row in test_df.iterrows():
        sid = str(row["subject_id"]).strip()
        npy_path = cache_dir / f"{sid}.npy"
        arr = np.load(npy_path)
        tensor = torch.from_numpy(arr).to(torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        scans.append(tensor)
        labels.append(float(row["label"]))

    X_adni = torch.stack(scans)
    y_adni = torch.tensor(labels, dtype=torch.float32)
    print(f"Loaded {len(X_adni)} ADNI test scans (Shape: {X_adni.shape}).", flush=True)
    return X_adni, y_adni


def plot_adaptation_curve(history: List[Dict[str, Any]], model_name: str, strategy: str, save_path: Path):
    """Generate high-resolution dual-axis training curve plot."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["validation_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_auc = [h["validation_AUC"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    plt.title(f"{model_name} — Strategy: {strategy.replace('_', ' ').title()}\nTraining & Validation Trajectory", fontsize=12, fontweight="bold", pad=12)

    color_loss_train = "#2563eb" # Blue
    color_loss_val = "#ea580c"   # Orange
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Binary Cross-Entropy Loss", color=color_loss_train, fontsize=11, fontweight="semibold")
    l1 = ax1.plot(epochs, train_loss, color=color_loss_train, linestyle="-", linewidth=2, label="Train Loss")
    l2 = ax1.plot(epochs, val_loss, color=color_loss_val, linestyle="--", linewidth=2, label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss_train)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color_metric_train = "#0d9488" # Teal
    color_metric_val = "#dc2626"   # Red
    ax2.set_ylabel("Accuracy / ROC-AUC", color=color_metric_val, fontsize=11, fontweight="semibold")
    l3 = ax2.plot(epochs, train_acc, color=color_metric_train, linestyle="-.", linewidth=1.8, label="Train Acc")
    l4 = ax2.plot(epochs, val_auc, color=color_metric_val, linestyle=":", linewidth=2.2, label="Val ROC-AUC")
    ax2.tick_params(axis="y", labelcolor=color_metric_val)
    ax2.set_ylim(0.2, 1.05)

    # Mark best checkpoint epoch
    best_entry = min(history, key=lambda h: (h["validation_loss"], -h["validation_AUC"]))
    best_ep = best_entry["epoch"]
    v_line = ax1.axvline(x=best_ep, color="#16a34a", linestyle=":", linewidth=2, label=f"Best Ep ({best_ep})")

    lines = l1 + l2 + l3 + l4 + [v_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def extract_intermediate_features(model: nn.Module, model_name: str, strategy: str, X: torch.Tensor, batch_size: int = 4) -> torch.Tensor:
    """Extract intermediate layer activations through frozen prefix blocks."""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = X[i:i + batch_size]
            if strategy == "linear_probe":
                f = model.extract_features(b)
            elif strategy == "partial_ft":
                if model_name == "BrainIAC":
                    f = model.patch_embed[:9](b)
                elif model_name == "FOMO25":
                    f = model.encoder[:14](b)
                elif model_name in ("SwinUNETR", "Swin"):
                    f = model.swin_encoder[:11](b)
            elif strategy == "deeper_ft":
                if model_name == "BrainIAC":
                    f = model.patch_embed[:4](b)
                elif model_name == "FOMO25":
                    f = model.encoder[:9](b)
                elif model_name in ("SwinUNETR", "Swin"):
                    f = model.swin_encoder[:6](b)
            elif strategy == "full_ft":
                f = b
            feats.append(f)
    return torch.cat(feats, dim=0)


def forward_unfrozen_suffix(model: nn.Module, model_name: str, strategy: str, f: torch.Tensor) -> torch.Tensor:
    """Forward intermediate activation through the unfrozen suffix layers and classification head."""
    if strategy == "linear_probe":
        return model.head(f).squeeze(-1)
    elif strategy == "partial_ft":
        if model_name == "BrainIAC":
            emb = model.patch_embed[9:](f)
        elif model_name == "FOMO25":
            emb = model.encoder[14:](f)
        elif model_name in ("SwinUNETR", "Swin"):
            emb = model.swin_encoder[11:](f)
        return model.head(emb).squeeze(-1)
    elif strategy == "deeper_ft":
        if model_name == "BrainIAC":
            emb = model.patch_embed[4:](f)
        elif model_name == "FOMO25":
            emb = model.encoder[9:](f)
        elif model_name in ("SwinUNETR", "Swin"):
            emb = model.swin_encoder[6:](f)
        return model.head(emb).squeeze(-1)
    elif strategy == "full_ft":
        return model(f).squeeze(-1)


def train_and_evaluate_run(
    model_name: str,
    strategy: str,
    seed: int,
    X_train_raw: torch.Tensor,
    y_train: torch.Tensor,
    X_val_raw: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 4,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], nn.Module]:
    """Train a single adaptation configuration with epoch-by-epoch logging and early stopping."""
    set_seed(seed)
    model, meta = build_adapted_foundation_model(model_name, strategy=strategy)
    optimizer = build_discriminative_optimizer(
        model, strategy=strategy, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # Pre-extract intermediate activations through frozen prefix to accelerate training
    F_train = extract_intermediate_features(model, model_name, strategy, X_train_raw, batch_size=4)
    F_val = extract_intermediate_features(model, model_name, strategy, X_val_raw, batch_size=4)

    history = []
    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 1
    best_weights = None

    n_train = len(F_train)

    for ep in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        perm = torch.randperm(n_train)
        epoch_losses = []
        all_train_probs = []
        all_train_trues = []

        for i in range(0, n_train, batch_size):
            b_idx = perm[i:i + batch_size]
            b_x = F_train[b_idx]
            b_y = y_train[b_idx]

            # Gentle feature perturbation / jitter during training if not linear probe
            if strategy != "linear_probe":
                if torch.rand(1).item() > 0.4:
                    b_x = b_x + 0.005 * torch.randn_like(b_x)

            optimizer.zero_grad()
            logits = forward_unfrozen_suffix(model, model_name, strategy, b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                all_train_probs.extend(probs)
                all_train_trues.extend(b_y.cpu().numpy())

        scheduler.step()

        train_loss = float(np.mean(epoch_losses))
        all_train_probs = np.array(all_train_probs)
        all_train_trues = np.array(all_train_trues)
        all_train_preds = (all_train_probs >= 0.5).astype(int)

        train_acc = float(accuracy_score(all_train_trues, all_train_preds))
        train_bal_acc = float(balanced_accuracy_score(all_train_trues, all_train_preds))
        train_f1 = float(f1_score(all_train_trues, all_train_preds, zero_division=0))
        try:
            train_auc = float(roc_auc_score(all_train_trues, all_train_probs))
        except Exception:
            train_auc = 0.5

        # 2. Validation Phase (Deterministic)
        model.eval()
        with torch.no_grad():
            val_logits = forward_unfrozen_suffix(model, model_name, strategy, F_val)
            val_loss = float(criterion(val_logits, y_val).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            y_val_np = y_val.cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)

            val_acc = float(accuracy_score(y_val_np, val_preds))
            val_bal_acc = float(balanced_accuracy_score(y_val_np, val_preds))
            val_f1 = float(f1_score(y_val_np, val_preds, zero_division=0))
            try:
                val_auc = float(roc_auc_score(y_val_np, val_probs))
            except Exception:
                val_auc = 0.5

        rec = {
            "model_name": model_name,
            "strategy": strategy,
            "seed": seed,
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "train_balanced_accuracy": round(train_bal_acc, 4),
            "train_F1": round(train_f1, 4),
            "train_AUC": round(train_auc, 4),
            "validation_loss": round(val_loss, 4),
            "validation_accuracy": round(val_acc, 4),
            "validation_balanced_accuracy": round(val_bal_acc, 4),
            "validation_F1": round(val_f1, 4),
            "validation_AUC": round(val_auc, 4),
            "trainable_parameters": meta["trainable_parameters"],
            "total_parameters": meta["total_parameters"],
            "learning_rate": head_lr,
            "weight_decay": weight_decay,
        }
        history.append(rec)

        # Early stopping checkpoint tracking (prioritize validation loss, tie-break on AUC)
        if val_loss < best_val_loss or (val_loss == best_val_loss and val_auc > best_val_auc):
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    best_record = next(h for h in history if h["epoch"] == best_epoch)
    run_summary = {
        "model_name": model_name,
        "strategy": strategy,
        "seed": seed,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "train_loss": best_record["train_loss"],
        "train_accuracy": best_record["train_accuracy"],
        "train_balanced_accuracy": best_record["train_balanced_accuracy"],
        "train_f1": best_record["train_F1"],
        "train_auc": best_record["train_AUC"],
        "validation_loss": best_record["validation_loss"],
        "validation_accuracy": best_record["validation_accuracy"],
        "validation_balanced_accuracy": best_record["validation_balanced_accuracy"],
        "validation_f1": best_record["validation_F1"],
        "validation_auc": best_record["validation_AUC"],
        "train_validation_accuracy_gap": round(best_record["train_accuracy"] - best_record["validation_accuracy"], 4),
        "trainable_parameters": meta["trainable_parameters"],
        "total_parameters": meta["total_parameters"],
        "trainable_fraction": meta["trainable_fraction"],
        "unfrozen_layers": meta["unfrozen_layers"],
    }

    return history, run_summary, model


def main():
    print("=" * 75, flush=True)
    print(" TARGETED FOUNDATION MODEL ADAPTATION & IMPROVEMENT CAMPAIGN", flush=True)
    print(" Fair Benchmarking of Brain-MRI Foundation Models", flush=True)
    print("=" * 75, flush=True)

    # 1. Load OASIS-1 pilot scans and labels into RAM
    cache_map = load_oasis_cache_into_ram()
    splits_df = pd.read_csv(r"D:\OASIS1_PILOT\pilot_splits.csv")

    train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

    X_train = torch.stack([cache_map[row["mri_file"]] for _, row in train_df.iterrows()])
    y_train = torch.tensor(train_df["label"].values, dtype=torch.float32)

    X_val = torch.stack([cache_map[row["mri_file"]] for _, row in val_df.iterrows()])
    y_val = torch.tensor(val_df["label"].values, dtype=torch.float32)

    # Test partitions remain untouched until development freeze
    X_test_oasis = torch.stack([cache_map[row["mri_file"]] for _, row in test_df.iterrows()])
    y_test_oasis = torch.tensor(test_df["label"].values, dtype=torch.float32)
    X_test_adni, y_test_adni = load_adni_test_into_ram()

    print(f"Partitions Prepared -> Train: {len(X_train)}, Val: {len(X_val)} | Locked Test: {len(X_test_oasis)}, ADNI: {len(X_test_adni)}", flush=True)

    models = ["BrainIAC", "FOMO25", "SwinUNETR"]
    strategies = ["linear_probe", "partial_ft", "deeper_ft", "full_ft"]

    all_epoch_records = []
    strategy_summaries = []
    trained_models = {}

    # 2. Execute Primary Adaptation Grid (Seed 42)
    print("\n" + "-" * 75, flush=True)
    print(" STAGE 1: EXECUTING 4-STRATEGY ADAPTATION GRID (SEED 42)", flush=True)
    print("-" * 75, flush=True)

    for model_name in models:
        print(f"\nEvaluating Foundation Model: {model_name}", flush=True)
        for strat in strategies:
            t0 = time.time()
            history, summary, model = train_and_evaluate_run(
                model_name=model_name,
                strategy=strat,
                seed=42,
                X_train_raw=X_train,
                y_train=y_train,
                X_val_raw=X_val,
                y_val=y_val,
                epochs=20,
                batch_size=4,
                backbone_lr=1e-5,
                head_lr=1e-4,
                weight_decay=1e-4,
            )
            elapsed = time.time() - t0
            all_epoch_records.extend(history)
            strategy_summaries.append(summary)
            trained_models[f"{model_name}_{strat}_seed42"] = model

            # Save checkpoint
            chk_path = CHECKPOINTS_DIR / f"{model_name.lower()}_{strat}_seed42_best.pt"
            torch.save(model.state_dict(), chk_path)
            summary["checkpoint_path"] = str(chk_path)

            # Generate dual-axis curve plot
            plot_path = CURVES_DIR / f"{model_name.lower()}_{strat}_train_vs_val.png"
            plot_adaptation_curve(history, model_name, strat, plot_path)
            summary["curve_plot"] = str(plot_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

            gap_sign = "+" if summary["train_validation_accuracy_gap"] >= 0 else ""
            print(f"  [{strat:12s}] Epoch {summary['best_epoch']:2d}/20 | Train Acc: {summary['train_accuracy']*100:4.1f}% | Val Acc: {summary['validation_accuracy']*100:4.1f}% (Gap: {gap_sign}{summary['train_validation_accuracy_gap']*100:4.1f}%) | Val AUC: {summary['validation_auc']:5.4f} | Time: {elapsed:4.1f}s", flush=True)

    # 3. Fit Classification (Against Linear Probe Control)
    print("\n" + "-" * 75, flush=True)
    print(" STAGE 2: EMPIRICAL FIT CLASSIFICATION (vs. LINEAR PROBE)", flush=True)
    print("-" * 75, flush=True)

    for model_name in models:
        ctrl = next(s for s in strategy_summaries if s["model_name"] == model_name and s["strategy"] == "linear_probe" and s["seed"] == 42)
        ctrl_auc = ctrl["validation_auc"]
        ctrl_gap = ctrl["train_validation_accuracy_gap"]

        for s in strategy_summaries:
            if s["model_name"] == model_name and s["seed"] == 42:
                if s["strategy"] == "linear_probe":
                    s["fit_classification"] = "control (underfitting baseline)"
                    s["classification_rationale"] = "Frozen backbone baseline serving as experimental control."
                else:
                    d_auc = s["validation_auc"] - ctrl_auc
                    train_acc = s["train_accuracy"]
                    val_acc = s["validation_accuracy"]
                    val_loss = s["validation_loss"]

                    if d_auc > 0.05 and val_loss <= ctrl["validation_loss"] + 0.1:
                        s["fit_classification"] = "improved generalization"
                        s["classification_rationale"] = f"Validation ROC-AUC increased by +{d_auc:.4f} with stable validation loss, confirming effective task adaptation."
                    elif train_acc > ctrl["train_accuracy"] and d_auc >= 0:
                        s["fit_classification"] = "reduced underfitting"
                        s["classification_rationale"] = f"Training accuracy increased to {train_acc*100:.1f}% while preserving validation generalization, alleviating the representation bottleneck."
                    elif train_acc >= 0.85 and (val_loss > ctrl["validation_loss"] + 0.15 or d_auc < -0.05):
                        s["fit_classification"] = "increased overfitting"
                        s["classification_rationale"] = f"High training accuracy ({train_acc*100:.1f}%) accompanied by validation degradation, reflecting sample memorization."
                    else:
                        s["fit_classification"] = "inconclusive"
                        s["classification_rationale"] = "Adaptation yielded modest shifts without statistically decisive generalization gain."

                print(f"  {model_name:10s} | {s['strategy']:12s} | Val AUC: {s['validation_auc']:5.4f} | Classification: {s['fit_classification']}", flush=True)

    # 4. Multi-Seed Stability for Top Promising Strategies (Seeds 43, 44)
    print("\n" + "-" * 75, flush=True)
    print(" STAGE 3: MULTI-SEED STABILITY EVALUATION (SEEDS 43 & 44)", flush=True)
    print("-" * 75, flush=True)

    best_strategy_per_model = {}
    for model_name in models:
        m_summaries = [s for s in strategy_summaries if s["model_name"] == model_name and s["seed"] == 42 and s["strategy"] != "linear_probe"]
        # Select on validation ROC-AUC (tie-break on validation loss)
        best_s = max(m_summaries, key=lambda s: (s["validation_auc"], -s["validation_loss"]))
        best_strategy = best_s["strategy"]
        best_strategy_per_model[model_name] = best_strategy
        print(f"Top adaptation strategy for {model_name}: {best_strategy} (Val AUC: {best_s['validation_auc']:.4f})", flush=True)

        # Run additional seeds 43 and 44
        for extra_seed in [43, 44]:
            t0 = time.time()
            history, summary, model = train_and_evaluate_run(
                model_name=model_name,
                strategy=best_strategy,
                seed=extra_seed,
                X_train_raw=X_train,
                y_train=y_train,
                X_val_raw=X_val,
                y_val=y_val,
                epochs=20,
                batch_size=4,
                backbone_lr=1e-5,
                head_lr=1e-4,
                weight_decay=1e-4,
            )
            all_epoch_records.extend(history)
            summary["fit_classification"] = best_s["fit_classification"]
            summary["classification_rationale"] = f"Multi-seed stability run (Seed {extra_seed})."
            strategy_summaries.append(summary)
            trained_models[f"{model_name}_{best_strategy}_seed{extra_seed}"] = model

            chk_path = CHECKPOINTS_DIR / f"{model_name.lower()}_{best_strategy}_seed{extra_seed}_best.pt"
            torch.save(model.state_dict(), chk_path)
            summary["checkpoint_path"] = str(chk_path)
            summary["curve_plot"] = best_s["curve_plot"]
            print(f"    Seed {extra_seed}: Val Acc: {summary['validation_accuracy']*100:.1f}%, Val AUC: {summary['validation_auc']:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # 5. POST-FREEZE LOCKED HOLDOUT EVALUATIONS
    print("\n" + "=" * 75, flush=True)
    print(" STAGE 4: HARD RESEARCH FREEZE & LOCKED HOLDOUT EVALUATIONS", flush=True)
    print("=" * 75, flush=True)

    comparison_results = []

    for model_name in models:
        best_strat = best_strategy_per_model[model_name]
        print(f"\n--- Model: {model_name} (Selected Best Strategy: {best_strat}) ---", flush=True)

        # Evaluate both Baseline Linear Probe (seed 42) and Selected Adapted Model (seed 42)
        eval_pairs = [
            ("Baseline Linear Probe", "linear_probe", trained_models[f"{model_name}_linear_probe_seed42"]),
            ("Adapted / Fine-Tuned", best_strat, trained_models[f"{model_name}_{best_strat}_seed42"]),
        ]

        model_comp = {
            "model_name": model_name,
            "selected_strategy": best_strat,
            "evaluations": {}
        }

        for label, strat_id, eval_model in eval_pairs:
            eval_model.eval()
            with torch.no_grad():
                # A. OASIS-1 Locked Test (N=8)
                logits_oasis = eval_model(X_test_oasis).squeeze(-1)
                probs_oasis = torch.sigmoid(logits_oasis).cpu().numpy()
                y_oasis_np = y_test_oasis.cpu().numpy()
                metrics_oasis = compute_full_metrics(y_oasis_np, probs_oasis)

                # B. ADNI External Holdout (N=8)
                logits_adni = eval_model(X_test_adni).squeeze(-1)
                probs_adni = torch.sigmoid(logits_adni).cpu().numpy()
                y_adni_np = y_test_adni.cpu().numpy()
                metrics_adni = compute_full_metrics(y_adni_np, probs_adni)

            model_comp["evaluations"][label] = {
                "strategy": strat_id,
                "oasis_locked_test": metrics_oasis,
                "adni_external_holdout": metrics_adni,
            }

            print(f"  [{label:24s}] OASIS Test AUC: {metrics_oasis['roc_auc']:5.4f}, Acc: {metrics_oasis['accuracy']*100:4.1f}% | ADNI AUC: {metrics_adni['roc_auc']:5.4f}, Acc: {metrics_adni['accuracy']*100:4.1f}%", flush=True)

        comparison_results.append(model_comp)

    # 6. Save Artifacts
    print("\n" + "-" * 75, flush=True)
    print(" STAGE 5: SERIALIZING BENCHMARK AUDIT ARTIFACTS", flush=True)
    print("-" * 75, flush=True)

    # A. CSV & JSON History
    history_df = pd.DataFrame(all_epoch_records)
    csv_path = OUTPUT_DIR / "adaptation_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(history_df)} epoch records)", flush=True)

    json_history_path = OUTPUT_DIR / "adaptation_history.json"
    with open(json_history_path, "w", encoding="utf-8") as f:
        json.dump(all_epoch_records, f, indent=2)
    print(f"Saved: {json_history_path}", flush=True)

    # B. Strategy Summaries
    summary_path = OUTPUT_DIR / "adaptation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(strategy_summaries, f, indent=2)
    print(f"Saved: {summary_path}", flush=True)

    # C. Baseline vs Adapted Comparison
    comp_path = OUTPUT_DIR / "adaptation_vs_baseline_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved: {comp_path}", flush=True)

    # D. Canonical Frontend Data Package
    frontend_data = {
        "report_title": "Foundation Model Downstream Task Adaptation",
        "section_tag": "Section 17 / Task Adaptation",
        "description": "Evaluation of task adaptation depth across BrainIAC, FOMO25, and MONAI Swin UNETR SSL.",
        "methodology_disclaimer": "Task adaptation was explored strictly on the OASIS-1 development partition (35 train / 7 val). Locked OASIS-1 (N=8) and external ADNI (N=8) holdouts were strictly evaluated post-freeze to assess genuine generalization changes.",
        "models_evaluated": len(models),
        "total_training_runs": len(strategy_summaries),
        "total_epoch_records": len(all_epoch_records),
        "best_strategies": best_strategy_per_model,
        "strategy_summaries": strategy_summaries,
        "baseline_vs_adapted_comparison": comparison_results,
    }
    frontend_json_path = CANONICAL_FRONTEND_DIR / "foundation_adaptation.json"
    with open(frontend_json_path, "w", encoding="utf-8") as f:
        json.dump(frontend_data, f, indent=2)
    print(f"Saved canonical frontend package: {frontend_json_path}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(" FOUNDATION MODEL ADAPTATION CAMPAIGN COMPLETE!", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
