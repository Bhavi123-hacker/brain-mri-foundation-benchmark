"""
ADNI Foundation Model Linear Probing Pilots (Day 3).
Evaluates frozen pretrained backbones with trainable linear heads for:
  1. BrainIAC (weights/brainiac_vit_base.pt)
  2. FOMO25 Winner (weights/fomo25_winner_encoder.pt)
  3. MONAI Swin UNETR SSL (weights/model_swinvit.pt)
Multi-seed execution: Seeds 42, 43, 44, 45, 46
Test set status: STRICTLY LOCKED & UNEVALUATED (Evaluated in Day 4)
Outputs:
  - results/checkpoints/adni_<model_slug>_linear_seed<seed>_best.pt
  - results/adni_foundation_linear_probe_summary.csv
  - results/adni_foundation_linear_probe_report.json
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
from models.foundation_brainiac.adapter import BrainIACAdapter
from models.foundation_fomo25.adapter import FOMO25Adapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from evaluation.metrics import compute_classification_metrics


def get_foundation_model(model_name: str, device: str):
    """Instantiate and configure foundation model with official checkpoint in linear probe mode."""
    if model_name == "BrainIAC":
        weights_path = os.path.abspath(os.path.join("weights", "brainiac_vit_base.pt"))
        assert os.path.exists(weights_path), f"Missing weights: {weights_path}"
        model = BrainIACAdapter(output_dim=1, weights_path=weights_path, adaptation_mode="linear_probe", dropout=0.2)
        slug = "brainiac"
    elif model_name == "FOMO25":
        weights_path = os.path.abspath(os.path.join("weights", "fomo25_winner_encoder.pt"))
        assert os.path.exists(weights_path), f"Missing weights: {weights_path}"
        model = FOMO25Adapter(output_dim=1, weights_path=weights_path, adaptation_mode="linear_probe", dropout=0.2)
        slug = "fomo25"
    elif model_name == "SwinUNETR":
        weights_path = os.path.abspath(os.path.join("weights", "model_swinvit.pt"))
        assert os.path.exists(weights_path), f"Missing weights: {weights_path}"
        model = SwinUNETRSSLAdapter(output_dim=1, weights_path=weights_path, adaptation_mode="linear_probe", dropout=0.2)
        slug = "swin_unetr"
    else:
        raise ValueError(f"Unknown foundation model: {model_name}")

    model.freeze_backbone()
    model.to(device)
    return model, slug, weights_path


def load_cached_scans(df: pd.DataFrame, cache_dir: str = "data/cache/adni_cn_ad"):
    """Load preprocessed 3D scans and labels into tensors in RAM."""
    tensors = []
    labels = []
    sids = []
    for _, row in df.iterrows():
        sid = str(row["subject_id"]).strip()
        npy_path = os.path.join(cache_dir, f"{sid}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Missing cached volume: {npy_path}")
        arr = np.load(npy_path)
        tensors.append(torch.from_numpy(arr).to(torch.float32))
        labels.append(float(row["label"]))
        sids.append(sid)

    tensors_stacked = torch.stack(tensors)  # (N, 1, 128, 128, 128)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    return tensors_stacked, labels_tensor, sids


def extract_features_batched(model, tensors: torch.Tensor, device: str, batch_size: int = 4):
    """Extract bottleneck features through the frozen backbone."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size].to(device)
            feats = model.extract_features(batch)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def train_linear_head_single_seed(
    model,
    slug: str,
    seed: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    checkpoint_dir: str,
    logger,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 7,
    batch_size: int = 4,
):
    set_seed(seed)
    # Re-initialize head weights with seed for deterministic reproducibility
    for layer in model.head.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    device = next(model.parameters()).device
    model.head.to(device)

    train_dataset = TensorDataset(train_feats, train_labels)
    val_dataset = TensorDataset(val_feats, val_labels)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    checkpoint_path = os.path.join(checkpoint_dir, f"adni_{slug}_linear_seed{seed}_best.pt")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_head_state = None
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Training
        model.head.train()
        train_losses = []
        for x_feat, y in train_loader:
            x_feat, y = x_feat.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model.head(x_feat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        epoch_train_loss = float(np.mean(train_losses))

        # Validation
        model.head.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_feat, y in val_loader:
                x_feat, y = x_feat.to(device), y.to(device)
                logits = model.head(x_feat)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().squeeze(-1).tolist())
                val_targets.extend(y.cpu().squeeze(-1).tolist())

        epoch_val_loss = float(np.mean(val_losses))
        val_preds_arr = np.array(val_preds, dtype=np.float32)
        val_targets_arr = np.array(val_targets, dtype=np.float32)
        val_metrics = compute_classification_metrics(val_targets_arr, val_preds_arr)

        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_head_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best head weights into model and save full checkpoint
    model.head.load_state_dict(best_head_state)
    torch.save(
        {
            "model_name": model.model_name,
            "slug": slug,
            "adaptation_mode": "linear_probe",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "config": {
                "lr": lr,
                "weight_decay": weight_decay,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "batch_size": batch_size,
            },
        },
        checkpoint_path,
    )

    total_time = time.time() - start_time
    logger.info(
        f"  [{model.model_name}] Seed {seed:2d} -> Best Epoch: {best_epoch:2d} | Val Loss: {best_val_loss:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | Time: {total_time:.2f}s"
    )

    return {
        "model": model.model_name,
        "slug": slug,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "checkpoint_path": checkpoint_path,
        "training_time_seconds": round(total_time, 2),
    }


def run_adni_foundation_linear_probes(
    splits_csv: str = "results/adni_cn_ad_split.csv",
    output_dir: str = "results",
    seeds: list = [42, 43, 44, 45, 46],
):
    logger = get_logger("ADNIFoundationLinearProbes")
    logger.info("=================================================================")
    logger.info("Starting ADNI Foundation Model Linear Probing Pilots (Day 3)")
    logger.info("Models: BrainIAC, FOMO25 Winner, MONAI Swin UNETR SSL")
    logger.info(f"Multi-Seed Evaluation: {seeds}")
    logger.info("=================================================================")

    assert os.path.exists(splits_csv), f"Missing splits: {splits_csv}"
    splits_df = pd.read_csv(splits_csv)

    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)

    logger.info(f"Cohorts: Train={len(train_df)}, Val={len(val_df)}, Locked Test={len(test_df)}")
    logger.info("LOCKED TEST SET POLICY: Strictly 0 evaluations on test cohort during probing.")

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Execution Device: {device}")

    # Load preprocessed volumes into memory once
    logger.info("Preloading cached 3D volumes into host RAM...")
    train_tensors, train_labels, _ = load_cached_scans(train_df)
    val_tensors, val_labels, _ = load_cached_scans(val_df)
    logger.info(f"Loaded: Train Tensors={train_tensors.shape}, Val Tensors={val_tensors.shape}")

    foundation_models = ["BrainIAC", "FOMO25", "SwinUNETR"]
    all_summary_rows = []
    models_report = {}
    overall_start = time.time()

    for mname in foundation_models:
        logger.info(f"\n-----------------------------------------------------------------")
        logger.info(f"Starting Model: {mname}")
        logger.info(f"-----------------------------------------------------------------")

        model, slug, weights_path = get_foundation_model(mname, device)

        # Audit parameter counts & verify 100% backbone freeze
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        backbone_frozen = (trainable_params <= 1000)  # Only linear head trainable

        assert backbone_frozen, f"Backbone of {mname} is not frozen!"
        logger.info(f"  Total Params: {total_params:,} | Frozen: {frozen_params:,} | Trainable: {trainable_params:,}")
        logger.info(f"  Backbone Freeze Verified: {backbone_frozen}")

        # Extract features through the frozen backbone
        logger.info(f"  Extracting bottleneck features through frozen {mname} backbone...")
        t_feat0 = time.time()
        train_feats = extract_features_batched(model, train_tensors, device, batch_size=4)
        val_feats = extract_features_batched(model, val_tensors, device, batch_size=4)
        feat_time = time.time() - t_feat0
        logger.info(f"  Extracted Train Feats: {train_feats.shape}, Val Feats: {val_feats.shape} in {feat_time:.2f}s")

        # Multi-seed linear probing
        m_seed_results = []
        for s in seeds:
            res = train_linear_head_single_seed(
                model=model,
                slug=slug,
                seed=s,
                train_feats=train_feats,
                train_labels=train_labels,
                val_feats=val_feats,
                val_labels=val_labels,
                checkpoint_dir=checkpoint_dir,
                logger=logger,
                lr=1e-3,
                weight_decay=1e-4,
                max_epochs=30,
                patience=7,
                batch_size=4,
            )
            m_seed_results.append(res)
            all_summary_rows.append(res)

        val_accs = [r["val_accuracy"] for r in m_seed_results]
        val_f1s = [r["val_f1"] for r in m_seed_results]
        val_aucs = [r["val_roc_auc"] for r in m_seed_results]
        val_losses = [r["best_val_loss"] for r in m_seed_results]

        models_report[mname] = {
            "slug": slug,
            "weights_path": weights_path,
            "total_parameters": total_params,
            "frozen_parameters": frozen_params,
            "trainable_parameters": trainable_params,
            "feature_extraction_time_seconds": round(feat_time, 2),
            "cross_seed_validation_metrics": {
                "val_accuracy_mean": round(float(np.mean(val_accs)), 4),
                "val_accuracy_std": round(float(np.std(val_accs)), 4),
                "val_f1_mean": round(float(np.mean(val_f1s)), 4),
                "val_f1_std": round(float(np.std(val_f1s)), 4),
                "val_roc_auc_mean": round(float(np.mean(val_aucs)), 4),
                "val_roc_auc_std": round(float(np.std(val_aucs)), 4),
                "val_loss_mean": round(float(np.mean(val_losses)), 4),
                "val_loss_std": round(float(np.std(val_losses)), 4),
            },
            "seed_runs": m_seed_results,
        }

    total_elapsed = time.time() - overall_start

    # Save summary CSV
    summary_df = pd.DataFrame(all_summary_rows)
    summary_csv_path = os.path.join(output_dir, "adni_foundation_linear_probe_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"\nSaved linear probe summary CSV to {summary_csv_path}")

    # Save report JSON
    full_report = {
        "benchmark_name": "ADNI CN vs AD Foundation Model Linear Probing Pilots",
        "models_evaluated": foundation_models,
        "seeds_evaluated": seeds,
        "locked_hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "batch_size": 4,
            "loss": "BCEWithLogitsLoss",
            "max_epochs": 30,
            "early_stopping_patience": 7,
            "selection_metric": "val_loss",
        },
        "dataset_split": {
            "train_subjects": len(train_df),
            "val_subjects": len(val_df),
            "test_subjects": len(test_df),
            "test_set_status": "STRICTLY LOCKED & UNEVALUATED",
        },
        "total_wall_clock_seconds": round(total_elapsed, 2),
        "models": models_report,
    }

    report_json_path = os.path.join(output_dir, "adni_foundation_linear_probe_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Saved full linear probe report JSON to {report_json_path}")
    logger.info("Day 3 Foundation Model Linear Probing Pilots Complete!")

    return full_report


if __name__ == "__main__":
    run_adni_foundation_linear_probes()
