"""
Phase 5B Real-Data 3D ResNet-18 Baseline Training Script (OASIS-1 Pilot).
Trains 3D ResNet-18 on the 35-subject training partition, monitors validation performance for early
stopping & checkpoint selection, and evaluates ONCE on the untouched 8-subject holdout test set.
"""

import os
import sys
import json
import time
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
from evaluation.profiler import ComputeTracker


def run_oasis1_baseline_training(data_dir: str = "D:\\OASIS1_PILOT", output_dir: str = "results"):
    logger = get_logger("OASIS1ResNet18Training")
    logger.info("=================================================================")
    logger.info("Starting Phase 5B Real-Data 3D ResNet-18 Baseline Training Run")
    logger.info(f"Data Location: {data_dir}")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    set_seed(42)

    # 1. Load splits dataframe and filter out forbidden files if any
    splits_path = os.path.join(data_dir, "pilot_splits.csv")
    splits_df = pd.read_csv(splits_path)

    # Exclude forbidden file: OAS1_0061_MR2_t1.npy
    splits_df = splits_df[~splits_df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)
    splits_df["file_path"] = [os.path.join(data_dir, fname) for fname in splits_df["mri_file"]]

    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)

    splits_dict = {"train": train_df, "val": val_df, "test": test_df}

    logger.info(f"Splits Breakdown -> Train: {len(train_df)} subjects, Val: {len(val_df)} subjects, Test: {len(test_df)} subjects")

    # 2. Hardware & Device Setup
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

    train_loader, val_loader, test_loader = create_benchmark_dataloaders(
        splits=splits_dict,
        task_config=task_cfg,
        batch_size=4,
        num_workers=0,
        spatial_shape=(128, 128, 128),
        label_budget=1.0,
        seed=42,
        cache_manager=cache_manager,
    )

    # 4. Instantiate Model & Optimizer
    model = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2).to(device)
    param_counts = model.get_parameter_count()
    logger.info(f"Model: 3D ResNet-18, Total Parameters: {param_counts['total_parameters']:,}")

    task = ClassificationTask(task_name="AD_vs_Dementia")
    max_epochs = 25
    early_stopping_patience = 7

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=1.0e-3,
        weight_decay=1.0e-4,
        optimizer_type="adamw",
        scheduler_type="cosine",
        total_epochs=max_epochs,
    )

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    experiment_name = "oasis1_pilot_resnet18"

    trainer = BenchmarkTrainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=cuda_avail,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        logger=logger,
    )

    # 5. Execute Training Loop with Validation Monitoring
    logger.info(f"Beginning training for up to {max_epochs} epochs (Early Stopping Patience = {early_stopping_patience})...")
    start_time = time.time()
    
    # Custom training loop to capture per-epoch validation metrics (Accuracy, F1, ROC-AUC)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        ep_start = time.time()
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_preds_logits, val_targets = trainer.evaluate(val_loader)
        ep_time = time.time() - ep_start

        if scheduler is not None:
            scheduler.step()

        # Compute validation classification metrics
        val_probs = torch.sigmoid(val_preds_logits).numpy().flatten()
        val_targets_np = val_targets.numpy().flatten()
        val_metrics = compute_classification_metrics(val_targets_np, val_probs)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), trainer.best_checkpoint_path)
        else:
            patience_counter += 1

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "epoch_time_seconds": round(ep_time, 2),
            "is_best": is_best,
        }
        history.append(record)

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Time: {ep_time:.2f}s {'[BEST]' if is_best else ''}"
        )

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {early_stopping_patience} consecutive epochs).")
            break

    total_train_time = round(time.time() - start_time, 2)
    logger.info(f"Training completed! Total Wall-Clock Time: {total_train_time}s | Best Epoch: {best_epoch} (Best Val Loss: {best_val_loss:.4f})")

    # 6. Load Best Validation Checkpoint for Final Test Set Evaluation
    assert os.path.exists(trainer.best_checkpoint_path), f"Checkpoint missing: {trainer.best_checkpoint_path}"
    model.load_state_dict(torch.load(trainer.best_checkpoint_path, map_location=device))
    model.eval()

    logger.info(f"Loaded best validation checkpoint from: {trainer.best_checkpoint_path}")
    logger.info("Evaluating ONCE on the untouched 8-subject holdout test set...")

    # 7. Final Holdout Test Set Evaluation & Latency Profiling
    test_start = time.time()
    test_loss, test_preds_logits, test_targets = trainer.evaluate(test_loader)
    test_eval_time = time.time() - test_start

    test_probs = torch.sigmoid(test_preds_logits).numpy().flatten()
    test_targets_np = test_targets.numpy().flatten()
    test_metrics = compute_classification_metrics(test_targets_np, test_probs)

    num_test_samples = len(test_targets_np)
    latency_ms = round((test_eval_time / max(1, num_test_samples)) * 1000.0, 2)
    throughput = round(num_test_samples / max(1e-6, test_eval_time), 2)

    test_metrics["test_loss"] = round(test_loss, 4)
    test_metrics["latency_ms_per_sample"] = latency_ms
    test_metrics["throughput_samples_per_sec"] = throughput

    logger.info(f"Final Untouched Test Metrics: {test_metrics}")

    # 8. Compile Complete Real-Data ResNet-18 Training Report
    report = {
        "dataset_name": "OASIS-1 Pilot",
        "data_location": data_dir,
        "configuration": {
            "task_type": "binary_classification",
            "target": "CDR (0=CN, >=1=DEMENTIA)",
            "batch_size": 4,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "seed": 42,
        },
        "subject_counts": {
            "total_pilot_subjects": 50,
            "train_subjects": len(train_df),
            "val_subjects": len(val_df),
            "test_subjects": len(test_df),
        },
        "model_architecture": {
            "name": "3D ResNet-18 Baseline",
            "stem": "Conv3d(1->64, k=7, s=2, p=3)",
            "stages": [64, 128, 256, 512],
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"],
        },
        "compute_hardware": {
            "device": device,
            "device_name": device_name,
            "cuda_available": cuda_avail,
            "total_training_time_seconds": total_train_time,
        },
        "training_execution": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "checkpoint_path": trainer.best_checkpoint_path,
            "epoch_history": history,
        },
        "final_untouched_test_metrics": test_metrics,
        "methodological_disclaimer": (
            "This is a pilot baseline result on 50 OASIS-1 subjects for software & data pipeline verification. "
            "It does not claim statistical superiority or clinical generalization prior to full benchmark scaling."
        ),
        "warnings_or_errors": [],
    }

    report_path = os.path.join(output_dir, "oasis1_pilot_resnet18_training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training Report saved to: {report_path}")
    logger.info("Phase 5B Real-Data ResNet-18 Baseline Training Completed Successfully!")
    return report


if __name__ == "__main__":
    run_oasis1_baseline_training()
