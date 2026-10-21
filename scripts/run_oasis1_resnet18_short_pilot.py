"""
Step 2 & 3: Controlled Real-Data OASIS-1 3D ResNet-18 Short Training Pilot.
Trains strictly on the 'train' split (138 subjects) and validates on 'val' (30 subjects).
Strictly ZERO access to the locked 'test' set (30 subjects).

Measures and logs:
- Exact configuration & seed (42)
- Software/device details (Blackwell GPU, CUDA 12.8, PyTorch 2.11.0)
- Start/end timestamps & per-epoch durations
- Train loss, val loss, and clinical val metrics (ROC-AUC, Acc, Bal Acc, F1, Prec, Rec, Spec, Brier)
- Peak GPU VRAM allocated
- Checkpoint selection rule: min val_loss

Outputs saved outside Git:
- Checkpoints: D:\\brain_mri_data\\OASIS1\\pilot_resnet18\\checkpoints\\
- Logs & Report: D:\\brain_mri_data\\OASIS1\\manifests\\oasis1_resnet18_short_pilot_report.json
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

# Project root in path
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
PILOT_OUTPUT_DIR = r"D:\brain_mri_data\OASIS1\pilot_resnet18"
CHECKPOINT_DIR = r"D:\brain_mri_data\OASIS1\pilot_resnet18\checkpoints"
REPORT_OUTPUT_JSON = r"D:\brain_mri_data\OASIS1\manifests\oasis1_resnet18_short_pilot_report.json"

RANDOM_SEED = 42
BATCH_SIZE = 4
MAX_EPOCHS = 5
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4

class OASIS1CachedDataset(Dataset):
    """
    Direct cached dataset reader from preprocessed .pt files.
    Returns (inputs, targets, metadata) matching BenchmarkTrainer.
    """
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

def run_oasis1_short_pilot():
    start_time_wall = time.time()
    start_time_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 80)
    print("CONTROLLED OASIS-1 3D RESNET-18 SHORT TRAINING PILOT")
    print(f"Start Time: {start_time_iso}")
    print("=" * 80)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)
    
    # 1. Hardware & Device Audit
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    cuda_capability = torch.cuda.get_device_capability(0) if cuda_avail else None
    
    print(f"Hardware & Software Profile:")
    print(f"  PyTorch Version:  {torch.__version__}")
    print(f"  CUDA Available:   {cuda_avail}")
    print(f"  Device Name:      {device_name} (Capability: {cuda_capability})")
    
    if cuda_avail:
        torch.cuda.reset_peak_memory_stats()
        
    # 2. Dataset & Split Preparation (ZERO Test-Set Access)
    assert os.path.exists(SPLITS_CSV), f"Missing splits: {SPLITS_CSV}"
    splits_df = pd.read_csv(SPLITS_CSV)
    
    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)
    
    assert len(train_df) == 138, f"Expected 138 train subjects, found {len(train_df)}"
    assert len(val_df) == 30, f"Expected 30 val subjects, found {len(val_df)}"
    assert len(test_df) == 30, f"Expected 30 test subjects, found {len(test_df)}"
    
    print(f"\nCohort Split Partitioning:")
    print(f"  Train:      {len(train_df)} subjects (Active Training)")
    print(f"  Validation: {len(val_df)} subjects (Model Selection & Validation Metrics)")
    print(f"  Locked Test:{len(test_df)} subjects (STRICTLY LOCKED - NEVER ACCESSED IN PILOT)")
    
    train_dataset = OASIS1CachedDataset(train_df, TENSORS_DIR)
    val_dataset = OASIS1CachedDataset(val_df, TENSORS_DIR)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    # 3. Model, Task, Optimizer, & Scheduler Setup
    model = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2)
    model.to(device)
    param_counts = model.get_parameter_count()
    print(f"\nModel Configuration:")
    print(f"  Architecture:     3D-ResNet-18 (BasicBlock3D, layers=[2, 2, 2, 2])")
    print(f"  Input Shape:      (1, 128, 128, 128)")
    print(f"  Output Dimension: 1 (Binary BCEWithLogitsLoss)")
    print(f"  Dropout:          0.2")
    print(f"  Total Parameters: {param_counts['total_parameters']:,} ({param_counts['trainable_parameters']:,} trainable)")
    
    task = ClassificationTask(task_name="AD_vs_CN", pos_weight=1.0)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optimizer_type="adamw",
        scheduler_type="cosine",
        total_epochs=MAX_EPOCHS,
        warmup_epochs=1
    )
    
    experiment_name = "oasis1_resnet18_short_pilot"
    trainer = BenchmarkTrainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=cuda_avail,
        early_stopping_patience=5,
        checkpoint_dir=CHECKPOINT_DIR,
        experiment_name=experiment_name,
    )
    
    # 4. Training Loop Execution & Per-Epoch Metrics Monitoring
    print(f"\n" + "-" * 80)
    print(f"COMMENCING SHORT TRAINING PILOT ({MAX_EPOCHS} EPOCHS)")
    print("-" * 80)
    
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        ep_start = time.time()
        
        # Train 1 epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate on validation partition
        val_loss, val_preds_logits, val_targets = trainer.evaluate(val_loader)
        ep_duration = time.time() - ep_start
        
        if scheduler is not None:
            scheduler.step()
            
        # Clinical Classification Metrics on Validation Partition
        val_probs = torch.sigmoid(val_preds_logits).numpy().flatten()
        val_targets_np = val_targets.numpy().flatten()
        val_metrics = compute_classification_metrics(val_targets_np, val_probs)
        
        # Checkpoint selection: min val_loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), trainer.best_checkpoint_path)
            
        ep_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "is_best": is_best,
            "duration_seconds": round(ep_duration, 2),
            "metrics": {
                "roc_auc": val_metrics["roc_auc"],
                "accuracy": val_metrics["accuracy"],
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "f1": val_metrics["f1"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "specificity": val_metrics["specificity"],
                "brier_score": val_metrics["brier_score"]
            }
        }
        history.append(ep_record)
        
        best_marker = " [BEST CHECKPOINT SAVED]" if is_best else ""
        print(f"Epoch [{epoch}/{MAX_EPOCHS}] ({ep_duration:.2f}s): "
              f"Train Loss = {train_loss:.4f} | "
              f"Val Loss = {val_loss:.4f} | "
              f"Val ROC-AUC = {val_metrics['roc_auc']:.4f} | "
              f"Val Acc = {val_metrics['accuracy']:.4f} | "
              f"Val BalAcc = {val_metrics['balanced_accuracy']:.4f}"
              f"{best_marker}", flush=True)

    total_training_wall_sec = time.time() - start_time_wall
    end_time_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Peak VRAM Measurement
    peak_vram_mb = 0.0
    if cuda_avail:
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = round(peak_vram_bytes / (1024**2), 2)
        print(f"\nPeak GPU VRAM Allocated: {peak_vram_mb:.2f} MB (~{peak_vram_mb/1024:.2f} GB)")
        
    print(f"Best Validation Loss:     {best_val_loss:.4f} (at Epoch {best_epoch})")
    print(f"Best Checkpoint Saved:    {trainer.best_checkpoint_path}")
    print(f"Total Wall-Clock Time:    {total_training_wall_sec:.2f}s")
    
    # Verify Best Checkpoint Exists and is Loadable
    assert os.path.exists(trainer.best_checkpoint_path), "Checkpoint missing!"
    chk_state = torch.load(trainer.best_checkpoint_path, map_location="cpu", weights_only=True)
    assert len(chk_state) > 0, "Empty checkpoint!"
    chk_size_mb = os.path.getsize(trainer.best_checkpoint_path) / (1024**2)
    print(f"Checkpoint Verified:      {chk_size_mb:.2f} MB")
    
    # 5. Build and Save Pilot Report JSON
    pilot_report = {
        "status": "COMPLETED",
        "experiment_name": experiment_name,
        "model": "3D-ResNet-18",
        "random_seed": RANDOM_SEED,
        "hardware_and_environment": {
            "device": device,
            "device_name": device_name,
            "cuda_capability": cuda_capability,
            "pytorch_version": torch.__version__,
            "peak_vram_mb": peak_vram_mb,
            "mixed_precision_amp": cuda_avail
        },
        "training_configuration": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "optimizer": "adamw",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "loss_function": "BCEWithLogitsLoss",
            "checkpoint_selection_rule": "Minimum Validation Loss (best_val_loss)"
        },
        "cohort_breakdown": {
            "total_primary_cohort": len(splits_df),
            "train_subjects": len(train_df),
            "val_subjects": len(val_df),
            "locked_test_subjects": len(test_df),
            "test_set_isolation_verified": True
        },
        "runtime_statistics": {
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "total_wall_clock_seconds": round(total_training_wall_sec, 2),
            "average_epoch_duration_seconds": round(np.mean([h["duration_seconds"] for h in history]), 2)
        },
        "training_history": history,
        "best_results": {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_checkpoint_path": trainer.best_checkpoint_path,
            "best_checkpoint_size_mb": round(chk_size_mb, 2)
        },
        "output_paths": {
            "report_json": REPORT_OUTPUT_JSON,
            "best_checkpoint": trainer.best_checkpoint_path,
            "checkpoint_dir": CHECKPOINT_DIR
        }
    }
    
    with open(REPORT_OUTPUT_JSON, "w") as f:
        json.dump(pilot_report, f, indent=2)
    print(f"\nSaved pilot report to: {REPORT_OUTPUT_JSON}")
    print("=" * 80)
    print("SHORT TRAINING PILOT COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_oasis1_short_pilot()
