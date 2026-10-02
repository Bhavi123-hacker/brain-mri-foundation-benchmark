"""
ADNI Supervised 3D ResNet-18 Multi-Seed Training Script.
Seeds: 42, 43, 44, 45, 46
Locked Baseline Configuration:
  - Architecture: 3D ResNet-18 (~33.1M params)
  - Optimizer: AdamW (lr=3e-4, weight_decay=1e-5)
  - Scheduler: CosineAnnealingLR
  - Loss: BCEWithLogitsLoss
  - Batch size: 4
  - Max epochs: 15, Early stopping patience: 5 (monitored on validation loss)
  - Test set status: STRICTLY LOCKED & UNTOUCHED (Zero test evaluation during training)
Outputs:
  - results/checkpoints/adni_resnet18_seed<seed>_best.pt
  - results/adni_resnet18_training_report.json
  - results/adni_resnet18_training_summary.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from utils.seed import set_seed
from models.baseline.resnet3d import build_resnet18_3d
from evaluation.metrics import compute_classification_metrics


class CachedADNIDataset(Dataset):
    """
    Fast in-memory/cached PyTorch Dataset for preprocessed ADNI tensors.
    Preloads all tensors directly to RAM for zero disk I/O during training.
    """
    def __init__(self, df: pd.DataFrame, cache_dir: str = "data/cache/adni_cn_ad"):
        self.records = []
        for _, row in df.iterrows():
            sid = str(row["subject_id"]).strip()
            npy_path = os.path.join(cache_dir, f"{sid}.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"Missing cached volume: {npy_path}")
            tensor = torch.from_numpy(np.load(npy_path)).to(torch.float32)
            label = torch.tensor([float(row["label"])], dtype=torch.float32)
            self.records.append((tensor, label, sid))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def train_single_seed(
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    checkpoint_dir: str,
    logger,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 15,
    patience: int = 5,
    batch_size: int = 4,
):
    logger.info(f"\n=======================================================")
    logger.info(f"Starting 3D ResNet-18 Training — Seed {seed}")
    logger.info(f"=======================================================")
    set_seed(seed)

    # In-memory datasets
    train_dataset = CachedADNIDataset(train_df)
    val_dataset = CachedADNIDataset(val_df)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    checkpoint_path = os.path.join(checkpoint_dir, f"adni_resnet18_seed{seed}_best.pt")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch_history = []
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()

        # Training Phase
        model.train()
        train_losses = []
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        epoch_train_loss = float(np.mean(train_losses))

        # Validation Phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().squeeze(-1).tolist())
                val_targets.extend(y.cpu().squeeze(-1).tolist())

        epoch_val_loss = float(np.mean(val_losses))
        val_preds_arr = np.array(val_preds, dtype=np.float32)
        val_targets_arr = np.array(val_targets, dtype=np.float32)
        val_metrics = compute_classification_metrics(val_targets_arr, val_preds_arr)

        ep_duration = time.time() - ep_t0
        is_best = epoch_val_loss < best_val_loss

        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "config": {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "architecture": "3D-ResNet-18",
                    },
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        rec = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "is_best": is_best,
            "epoch_time_seconds": round(ep_duration, 2),
        }
        epoch_history.append(rec)

        logger.info(
            f"Seed {seed} | Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Time: {ep_duration:.1f}s {'[BEST]' if is_best else ''}"
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered for seed {seed} at epoch {epoch} (patience={patience}).")
            break

    total_time = time.time() - start_time
    logger.info(f"Seed {seed} finished in {total_time:.1f}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # Load best metrics from saved checkpoint
    best_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_val_metrics = best_ckpt["val_metrics"]

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(epoch_history),
        "best_val_loss": round(best_val_loss, 4),
        "val_accuracy": best_val_metrics["accuracy"],
        "val_precision": best_val_metrics["precision"],
        "val_recall": best_val_metrics["recall"],
        "val_f1": best_val_metrics["f1"],
        "val_roc_auc": best_val_metrics["roc_auc"],
        "checkpoint_path": checkpoint_path,
        "training_time_seconds": round(total_time, 2),
        "epoch_history": epoch_history,
    }


def run_adni_resnet18_multiseed(
    splits_csv: str = "results/adni_cn_ad_split.csv",
    output_dir: str = "results",
    seeds: list = [42, 43, 44, 45, 46],
):
    logger = get_logger("ADNIResNet18MultiSeed")
    logger.info("=================================================================")
    logger.info("Starting ADNI 3D ResNet-18 Multi-Seed Baseline Training")
    logger.info(f"Seeds: {seeds}")
    logger.info("=================================================================")

    assert os.path.exists(splits_csv), f"Missing splits: {splits_csv}"
    splits_df = pd.read_csv(splits_csv)

    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)

    logger.info(f"Dataset partitions -> Train: {len(train_df)}, Val: {len(val_df)}, Locked Test: {len(test_df)}")
    logger.info("TEST SET IS LOCKED: Zero test evaluation will occur during training.")

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_results = []
    overall_start = time.time()

    for s in seeds:
        res = train_single_seed(
            seed=s,
            train_df=train_df,
            val_df=val_df,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
            lr=3e-4,
            weight_decay=1e-5,
            max_epochs=15,
            patience=5,
            batch_size=4,
        )
        seed_results.append(res)

    overall_elapsed = time.time() - overall_start

    # Build summary dataframe
    summary_rows = []
    for r in seed_results:
        summary_rows.append({
            "model": "3D ResNet-18",
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"],
            "best_val_loss": r["best_val_loss"],
            "val_accuracy": r["val_accuracy"],
            "val_precision": r["val_precision"],
            "val_recall": r["val_recall"],
            "val_f1": r["val_f1"],
            "val_roc_auc": r["val_roc_auc"],
            "training_time_seconds": r["training_time_seconds"],
            "checkpoint_path": r["checkpoint_path"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "adni_resnet18_training_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary CSV to {summary_csv_path}")

    # Aggregate statistics
    val_accs = [r["val_accuracy"] for r in seed_results]
    val_f1s = [r["val_f1"] for r in seed_results]
    val_aucs = [r["val_roc_auc"] for r in seed_results]
    val_losses = [r["best_val_loss"] for r in seed_results]

    full_report = {
        "benchmark_name": "ADNI CN vs AD 3D ResNet-18 Supervised Baseline",
        "model_architecture": "3D ResNet-18",
        "total_parameters": 33160513,
        "trainable_parameters": 33160513,
        "seeds_evaluated": seeds,
        "locked_hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "batch_size": 4,
            "loss": "BCEWithLogitsLoss",
            "early_stopping_patience": 5,
            "selection_metric": "val_loss",
        },
        "dataset_split": {
            "train_subjects": len(train_df),
            "val_subjects": len(val_df),
            "test_subjects": len(test_df),
            "test_set_status": "LOCKED & UNTOUCHED",
        },
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
        "total_training_wall_clock_seconds": round(overall_elapsed, 2),
        "seed_runs": seed_results,
    }

    report_json_path = os.path.join(output_dir, "adni_resnet18_training_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Saved full training report to {report_json_path}")
    logger.info("Multi-seed 3D ResNet-18 baseline training complete!")

    return full_report


if __name__ == "__main__":
    run_adni_resnet18_multiseed()
