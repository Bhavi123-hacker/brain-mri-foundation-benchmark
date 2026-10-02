"""
Phase 5C Controlled 3D ResNet-18 Baseline Tuning Script (OASIS-1 Pilot).
Executes a 2x2 grid search over learning rates (1e-3, 3e-4) and weight decays (1e-4, 1e-5)
using the 35/7 train/validation split. Selects the optimal configuration based strictly on validation loss.
Does NOT evaluate the 8-subject test set.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from utils.seed import set_seed
from models.baseline.resnet3d import build_resnet18_3d
from preprocessing.cache_manager import PreprocessedCacheManager
from preprocessing.dataloader import create_benchmark_dataloaders
from tasks.classification import ClassificationTask
from training.optimizer import build_optimizer_and_scheduler
from training.trainer import BenchmarkTrainer
from evaluation.metrics import compute_classification_metrics


def run_oasis1_baseline_tuning(data_dir: str = "D:\\OASIS1_PILOT", output_dir: str = "results"):
    logger = get_logger("OASIS1ResNet18Tuning")
    logger.info("=================================================================")
    logger.info("Starting Phase 5C Controlled 3D ResNet-18 Baseline Tuning")
    logger.info(f"Data Location: {data_dir}")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    set_seed(42)

    # 1. Load splits dataframe and exclude forbidden file if present
    splits_path = os.path.join(data_dir, "pilot_splits.csv")
    splits_df = pd.read_csv(splits_path)
    splits_df = splits_df[~splits_df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)
    splits_df["file_path"] = [os.path.join(data_dir, fname) for fname in splits_df["mri_file"]]

    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)

    splits_dict = {"train": train_df, "val": val_df, "test": test_df}

    logger.info(f"Partitions -> Train: {len(train_df)} subjects, Val: {len(val_df)} subjects, Test: {len(test_df)} subjects (Test set isolated & un-evaluated)")

    # 2. Hardware Setup
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Host)"

    logger.info(f"Execution Device: {device} ({device_name})")

    # 3. DataLoaders with Preprocessed Cache Manager
    cache_dir = os.path.join("data", "cache", "oasis1_pilot")
    cache_manager = PreprocessedCacheManager(cache_dir=cache_dir)

    task_cfg = {
        "task_type": "classification",
        "target_column": "label",
        "positive_class": 1.0,
        "negative_class": 0.0,
    }

    train_loader, val_loader, _ = create_benchmark_dataloaders(
        splits=splits_dict,
        task_config=task_cfg,
        batch_size=4,
        num_workers=0,
        spatial_shape=(128, 128, 128),
        label_budget=1.0,
        seed=42,
        cache_manager=cache_manager,
    )

    # 4. Define 2x2 Search Grid
    grid_search_space = [
        {"trial_id": 1, "learning_rate": 1.0e-3, "weight_decay": 1.0e-4},
        {"trial_id": 2, "learning_rate": 1.0e-3, "weight_decay": 1.0e-5},
        {"trial_id": 3, "learning_rate": 3.0e-4, "weight_decay": 1.0e-4},
        {"trial_id": 4, "learning_rate": 3.0e-4, "weight_decay": 1.0e-5},
    ]

    task = ClassificationTask(task_name="AD_vs_Dementia")
    max_epochs = 25
    early_stopping_patience = 7

    trial_summary_table = []
    best_overall_val_loss = float("inf")
    best_trial_record = None
    best_trial_checkpoint_path = None

    start_tuning_time = time.time()

    # 5. Execute 4 Tuning Trials
    for trial_cfg in grid_search_space:
        trial_id = trial_cfg["trial_id"]
        lr = trial_cfg["learning_rate"]
        wd = trial_cfg["weight_decay"]
        trial_name = f"oasis1_pilot_resnet18_trial_{trial_id:02d}"

        logger.info("-----------------------------------------------------------------")
        logger.info(f"Starting Trial {trial_id}/4: LR={lr:.1e}, WeightDecay={wd:.1e}")
        logger.info("-----------------------------------------------------------------")

        set_seed(42 + trial_id)

        model = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2).to(device)
        param_counts = model.get_parameter_count()

        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            lr=lr,
            weight_decay=wd,
            optimizer_type="adamw",
            scheduler_type="cosine",
            total_epochs=max_epochs,
        )

        trainer = BenchmarkTrainer(
            model=model,
            task=task,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=cuda_avail,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=checkpoint_dir,
            experiment_name=trial_name,
            logger=logger,
        )

        trial_start_time = time.time()
        best_val_loss_trial = float("inf")
        best_epoch_trial = 0
        best_val_metrics_trial = None

        for epoch in range(1, max_epochs + 1):
            ep_start = time.time()
            train_loss = trainer.train_epoch(train_loader)
            val_loss, val_preds_logits, val_targets = trainer.evaluate(val_loader)
            ep_time = time.time() - ep_start

            if scheduler is not None:
                scheduler.step()

            val_probs = torch.sigmoid(val_preds_logits).numpy().flatten()
            val_targets_np = val_targets.numpy().flatten()
            val_metrics = compute_classification_metrics(val_targets_np, val_probs)

            is_best = val_loss < best_val_loss_trial
            if is_best:
                best_val_loss_trial = val_loss
                best_epoch_trial = epoch
                best_val_metrics_trial = val_metrics
                torch.save(model.state_dict(), trainer.best_checkpoint_path)

            logger.info(
                f"[Trial {trial_id}] Ep {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} "
                f"{'[BEST]' if is_best else ''}"
            )

            # Check early stopping (7 epochs without improvement)
            if (epoch - best_epoch_trial) >= early_stopping_patience:
                logger.info(f"[Trial {trial_id}] Early stopping at epoch {epoch}.")
                break

        trial_time = round(time.time() - trial_start_time, 2)

        record = {
            "trial_id": trial_id,
            "learning_rate": lr,
            "weight_decay": wd,
            "epochs_run": epoch,
            "best_epoch": best_epoch_trial,
            "best_val_loss": round(best_val_loss_trial, 4),
            "val_accuracy": best_val_metrics_trial["accuracy"],
            "val_precision": best_val_metrics_trial["precision"],
            "val_recall": best_val_metrics_trial["recall"],
            "val_f1": best_val_metrics_trial["f1"],
            "val_roc_auc": best_val_metrics_trial["roc_auc"],
            "trial_time_seconds": trial_time,
            "checkpoint_path": trainer.best_checkpoint_path,
        }
        trial_summary_table.append(record)

        if best_val_loss_trial < best_overall_val_loss:
            best_overall_val_loss = best_val_loss_trial
            best_trial_record = record
            best_trial_checkpoint_path = trainer.best_checkpoint_path

    total_tuning_time = round(time.time() - start_tuning_time, 2)

    # 6. Save Best Tuned Checkpoint Copy
    tuned_best_path = os.path.join(checkpoint_dir, "oasis1_pilot_resnet18_tuned_best.pt")
    shutil.copyfile(best_trial_checkpoint_path, tuned_best_path)
    logger.info(f"Copied best trial checkpoint to: {tuned_best_path}")

    # 7. Compile Phase 5C Baseline Tuning Report JSON
    report = {
        "dataset_name": "OASIS-1 Pilot",
        "data_location": data_dir,
        "tuning_stage": "Phase 5C Controlled 3D ResNet-18 Baseline Tuning",
        "subject_counts": {
            "train_subjects": len(train_df),
            "val_subjects": len(val_df),
            "test_subjects": len(test_df),
            "test_set_status": "UNTOUCHED & ISOLATED (Not evaluated during tuning)",
        },
        "model_architecture": {
            "name": "3D ResNet-18 Baseline",
            "total_parameters": param_counts["total_parameters"],
        },
        "compute_hardware": {
            "device": device,
            "device_name": device_name,
            "cuda_available": cuda_avail,
            "total_tuning_time_seconds": total_tuning_time,
        },
        "search_grid": {
            "learning_rates": [1.0e-3, 3.0e-4],
            "weight_decays": [1.0e-4, 1.0e-5],
            "max_epochs_per_trial": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "seed": 42,
        },
        "tuning_trials": trial_summary_table,
        "selected_optimal_configuration": {
            "trial_id": best_trial_record["trial_id"],
            "learning_rate": best_trial_record["learning_rate"],
            "weight_decay": best_trial_record["weight_decay"],
            "best_epoch": best_trial_record["best_epoch"],
            "best_val_loss": best_trial_record["best_val_loss"],
            "val_accuracy": best_trial_record["val_accuracy"],
            "val_f1": best_trial_record["val_f1"],
            "val_roc_auc": best_trial_record["val_roc_auc"],
            "selected_checkpoint_path": tuned_best_path,
        },
        "methodological_disclaimer": (
            "This is a controlled pilot tuning exercise on 7 validation subjects for hyperparameter pipeline verification. "
            "It does not claim statistical superiority or clinical generalization prior to full benchmark scaling. "
            "The 8-subject test set remained strictly isolated and was NOT evaluated during tuning."
        ),
        "warnings_or_errors": [],
    }

    report_path = os.path.join(output_dir, "oasis1_pilot_resnet18_tuning_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Tuning Report saved to: {report_path}")
    logger.info(f"Phase 5C Controlled Tuning Completed Successfully! Optimal Config: Trial {best_trial_record['trial_id']} (LR={best_trial_record['learning_rate']:.1e}, WD={best_trial_record['weight_decay']:.1e}, Val Loss={best_trial_record['best_val_loss']:.4f})")
    return report


if __name__ == "__main__":
    run_oasis1_baseline_tuning()
