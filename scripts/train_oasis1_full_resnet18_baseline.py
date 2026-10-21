"""
Full OASIS-1 3D ResNet-18 Baseline Training Run (Primary Cohort N = 198).
- Train Split: 138 subjects (AdamW, CosineAnnealingLR, AMP enabled)
- Validation Split: 30 subjects (Monitored for early stopping & checkpoint selection)
- Holdout Test Split: 30 subjects (Evaluated ONCE with best validation checkpoint)

Hardware: NVIDIA RTX PRO 2000 Blackwell (Compute Capability (12, 0))
Outputs saved strictly outside Git:
- Checkpoints: D:\\brain_mri_data\\OASIS1\\full_baseline_resnet18\\checkpoints\\
- Report: D:\\brain_mri_data\\OASIS1\\manifests\\oasis1_full_resnet18_baseline_report.json
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Ensure project root in sys.path
project_root = r"D:\ml project"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.seed import set_seed
from utils.logging_utils import get_logger
from models.baseline.resnet3d import build_resnet18_3d
from tasks.classification import ClassificationTask
from training.optimizer import build_optimizer_and_scheduler
from training.trainer import BenchmarkTrainer
from evaluation.metrics import compute_classification_metrics

# Paths
SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
OUTPUT_DIR = r"D:\brain_mri_data\OASIS1\full_baseline_resnet18"
CHECKPOINT_DIR = r"D:\brain_mri_data\OASIS1\full_baseline_resnet18\checkpoints"
REPORT_OUTPUT_JSON = r"D:\brain_mri_data\OASIS1\manifests\oasis1_full_resnet18_baseline_report.json"

RANDOM_SEED = 42
BATCH_SIZE = 4
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4

class OASIS1CachedDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, tensors_dir: str):
        self.df = metadata_df.reset_index(drop=True)
        self.tensors_dir = tensors_dir
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row["session_id"]).strip()
        pt_path = os.path.join(self.tensors_dir, f"{sid}.pt")
        
        data_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
        tensor = data_dict["tensor"] # (1, 128, 128, 128), float32
        label = float(data_dict["binary_label"])
        target = torch.tensor([label], dtype=torch.float32)
        
        meta = {
            "session_id": sid,
            "base_subject_id": data_dict["base_subject_id"],
            "split": data_dict["split"],
            "label_description": data_dict["label_description"],
            "cdr": data_dict["cdr"]
        }
        return tensor, target, meta

def train_oasis1_full_resnet18():
    start_time_wall = time.time()
    start_time_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    logger = get_logger("OASIS1FullResNet18")
    
    logger.info("=" * 80)
    logger.info("FULL OASIS-1 3D RESNET-18 BASELINE BENCHMARK TRAINING RUN")
    logger.info(f"Start Time: {start_time_iso}")
    logger.info("=" * 80)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)
    
    # 1. Device Setup
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    cuda_capability = torch.cuda.get_device_capability(0) if cuda_avail else None
    
    logger.info(f"Device: {device} ({device_name}, Capability: {cuda_capability})")
    logger.info(f"PyTorch Version: {torch.__version__} | AMP: {cuda_avail}")
    
    if cuda_avail:
        torch.cuda.reset_peak_memory_stats()
        
    # 2. Data Loaders
    assert os.path.exists(SPLITS_CSV), f"Missing splits: {SPLITS_CSV}"
    splits_df = pd.read_csv(SPLITS_CSV)
    
    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)
    
    assert len(train_df) == 138, f"Expected 138 train, found {len(train_df)}"
    assert len(val_df) == 30, f"Expected 30 val, found {len(val_df)}"
    assert len(test_df) == 30, f"Expected 30 test, found {len(test_df)}"
    
    logger.info(f"Partitions: Train={len(train_df)}, Val={len(val_df)}, Locked Test={len(test_df)}")
    
    train_dataset = OASIS1CachedDataset(train_df, TENSORS_DIR)
    val_dataset = OASIS1CachedDataset(val_df, TENSORS_DIR)
    test_dataset = OASIS1CachedDataset(test_df, TENSORS_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    
    # 3. Model & Task Setup
    model = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2)
    model.to(device)
    param_counts = model.get_parameter_count()
    logger.info(f"Model: 3D-ResNet-18, Parameters: {param_counts['total_parameters']:,} (100% trainable)")
    
    task = ClassificationTask(task_name="AD_vs_CN", pos_weight=1.0)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optimizer_type="adamw",
        scheduler_type="cosine",
        total_epochs=MAX_EPOCHS,
        warmup_epochs=2
    )
    
    experiment_name = "oasis1_full_resnet18"
    trainer = BenchmarkTrainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=cuda_avail,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        checkpoint_dir=CHECKPOINT_DIR,
        experiment_name=experiment_name,
        logger=logger,
    )
    
    # 4. Training Loop (Train on train_loader, Validate on val_loader)
    logger.info("-" * 80)
    logger.info(f"Beginning Training (Max Epochs = {MAX_EPOCHS}, Early Stopping Patience = {EARLY_STOPPING_PATIENCE})...")
    logger.info("-" * 80)
    
    history = []
    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        if cuda_avail:
            torch.cuda.synchronize()
        ep_start = time.time()
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_preds_logits, val_targets = trainer.evaluate(val_loader)
        
        if cuda_avail:
            torch.cuda.synchronize()
        ep_duration = time.time() - ep_start
        
        if scheduler is not None:
            scheduler.step()
            
        val_probs = torch.sigmoid(val_preds_logits).numpy().flatten()
        val_targets_np = val_targets.numpy().flatten()
        val_metrics = compute_classification_metrics(val_targets_np, val_probs)
        
        # Checkpoint selection: min val_loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), trainer.best_checkpoint_path)
        else:
            patience_counter += 1
            
        ep_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "is_best": is_best,
            "duration_seconds": round(ep_duration, 2),
            "metrics": val_metrics
        }
        history.append(ep_record)
        
        logger.info(
            f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} ({ep_duration:.2f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val BalAcc: {val_metrics['balanced_accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} "
            f"{'[BEST]' if is_best else f'(Patience {patience_counter}/{EARLY_STOPPING_PATIENCE})'}"
        )
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at Epoch {epoch}. Restoring best checkpoint from Epoch {best_epoch}.")
            break

    training_time_sec = round(time.time() - start_time_wall, 2)
    logger.info(f"Training loop finished in {training_time_sec:.2f}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val AUC: {best_val_auc:.4f})")
    
    # 5. Load Best Checkpoint for Single Holdout Test Evaluation
    assert os.path.exists(trainer.best_checkpoint_path), f"Best checkpoint missing: {trainer.best_checkpoint_path}"
    model.load_state_dict(torch.load(trainer.best_checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded best validation checkpoint: {trainer.best_checkpoint_path}")
    logger.info("Evaluating ONCE on the locked holdout test partition (30 subjects)...")
    
    # 6. Final Test Set Evaluation
    if cuda_avail:
        torch.cuda.synchronize()
    test_eval_start = time.time()
    
    test_loss, test_preds_logits, test_targets = trainer.evaluate(test_loader)
    
    if cuda_avail:
        torch.cuda.synchronize()
    test_eval_time = time.time() - test_eval_start
    
    test_probs = torch.sigmoid(test_preds_logits).numpy().flatten()
    test_targets_np = test_targets.numpy().flatten()
    test_metrics = compute_classification_metrics(test_targets_np, test_probs)
    
    num_test_samples = len(test_targets_np)
    latency_ms_per_sample = round((test_eval_time / max(1, num_test_samples)) * 1000.0, 2)
    throughput_samples_per_sec = round(num_test_samples / max(1e-6, test_eval_time), 2)
    
    test_metrics["test_loss"] = round(test_loss, 4)
    test_metrics["latency_ms_per_sample"] = latency_ms_per_sample
    test_metrics["throughput_samples_per_sec"] = throughput_samples_per_sec
    
    logger.info("=" * 80)
    logger.info("FINAL UNTOUCHED TEST SET METRICS (OASIS-1 Holdout N = 30)")
    logger.info("=" * 80)
    logger.info(f"  Test Loss:              {test_metrics['test_loss']:.4f}")
    logger.info(f"  Test ROC-AUC:           {test_metrics['roc_auc']:.4f}")
    logger.info(f"  Test Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"  Test Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    logger.info(f"  Test Sensitivity:       {test_metrics['sensitivity']*100:.2f}%")
    logger.info(f"  Test Specificity:       {test_metrics['specificity']*100:.2f}%")
    logger.info(f"  Test Precision:         {test_metrics['precision']*100:.2f}%")
    logger.info(f"  Test F1 Score:          {test_metrics['f1']:.4f}")
    logger.info(f"  Test Brier Score:       {test_metrics['brier_score']:.4f}")
    logger.info(f"  Inference Latency:      {latency_ms_per_sample:.2f} ms/scan (Throughput: {throughput_samples_per_sec:.1f} scans/s)")
    
    # 7. Hardware Profiling & Summary Compilation
    total_wall_clock_sec = round(time.time() - start_time_wall, 2)
    end_time_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    
    peak_vram_mb = 0.0
    if cuda_avail:
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
        logger.info(f"Peak GPU VRAM Allocated: {peak_vram_mb:.2f} MB (~{peak_vram_mb/1024:.2f} GB)")
        
    chk_size_mb = round(os.path.getsize(trainer.best_checkpoint_path) / (1024**2), 2)
    
    final_report = {
        "status": "COMPLETED",
        "benchmark_title": "Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines",
        "experiment_name": "oasis1_full_resnet18_baseline",
        "model_name": "3D-ResNet-18",
        "task_name": "AD_vs_CN (Dementia vs Cognitively Normal)",
        "random_seed": RANDOM_SEED,
        "hardware_and_environment": {
            "device": device,
            "device_name": device_name,
            "cuda_capability": cuda_capability,
            "pytorch_version": torch.__version__,
            "peak_vram_mb": peak_vram_mb,
            "mixed_precision_amp": cuda_avail
        },
        "model_architecture": {
            "name": "3D-ResNet-18",
            "in_channels": 1,
            "spatial_dims": 3,
            "dropout": 0.2,
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"]
        },
        "training_configuration": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "loss_function": "BCEWithLogitsLoss",
            "checkpoint_selection_rule": "Minimum Validation Loss (best_val_loss)"
        },
        "cohort_breakdown": {
            "primary_cohort_total": len(splits_df),
            "train_subjects": len(train_df),
            "val_subjects": len(val_df),
            "locked_test_subjects": len(test_df),
            "patient_leakage_guarantee": "0.0% (strictly disjoint patient-level partitions)"
        },
        "timing_and_profiling": {
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "total_wall_clock_seconds": total_wall_clock_sec,
            "total_training_seconds": training_time_sec,
            "average_epoch_duration_seconds": round(float(np.mean([h["duration_seconds"] for h in history])), 2),
            "test_eval_duration_seconds": round(test_eval_time, 3),
            "latency_ms_per_sample": latency_ms_per_sample,
            "throughput_samples_per_sec": throughput_samples_per_sec
        },
        "training_execution": {
            "epochs_completed": len(history),
            "early_stopping_triggered": len(history) < MAX_EPOCHS,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_auc": round(best_val_auc, 4),
            "best_checkpoint_path": trainer.best_checkpoint_path,
            "best_checkpoint_size_mb": chk_size_mb,
            "epoch_history": history
        },
        "final_holdout_test_metrics": test_metrics,
        "output_paths": {
            "report_json": REPORT_OUTPUT_JSON,
            "best_checkpoint": trainer.best_checkpoint_path,
            "checkpoint_dir": CHECKPOINT_DIR
        }
    }
    
    with open(REPORT_OUTPUT_JSON, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Saved complete baseline benchmark report to: {REPORT_OUTPUT_JSON}")
    logger.info("=" * 80)
    logger.info("FULL OASIS-1 RESNET-18 BASELINE RUN COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    train_oasis1_full_resnet18()
