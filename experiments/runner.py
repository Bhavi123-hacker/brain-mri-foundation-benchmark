"""
Experiment Orchestrator and Staged Execution Engine.
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

# Ensure project root is in sys.path when running script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_parser import load_config
from utils.seed import set_seed
from utils.logging_utils import get_logger, setup_experiment_logging
from preprocessing.transforms import get_preprocessing_transforms
from preprocessing.pipeline import create_patient_level_splits, BrainMRIDataset
from preprocessing.mock_data import generate_mock_dataset
from models import build_model
from tasks import ClassificationTask, RegressionTask
from training import sample_training_budget, build_optimizer_and_scheduler, BenchmarkTrainer
from evaluation import evaluate_model_on_test_set, ComputeTracker


def run_stage_a_smoke_test(config_dir: str = "configs") -> Dict[str, Any]:
    """
    Stage A: Execute a lightweight CPU smoke test on synthetic phantoms.
    Verifies that forward pass, loss calculation, backward step, optimizer,
    and metrics function without errors on local CPU.
    """
    logger = get_logger("StageA_SmokeTest")
    logger.info("Initializing Stage A Smoke Test on Local CPU...")

    config = load_config(config_dir)
    set_seed(42)

    # 1. Generate or load synthetic mock dataset
    mock_meta_path = generate_mock_dataset(
        output_dir="data/mock_data",
        num_patients=6,
        scans_per_patient=2,
        volume_shape=(32, 32, 32), # Small miniature shape for fast CPU smoke test
        seed=42
    )
    df = pd.read_csv(mock_meta_path)
    logger.info(f"Loaded synthetic dataset with {len(df)} scans across {df['PTID'].nunique()} patients.")

    # 2. Patient-level split
    splits = create_patient_level_splits(df, patient_col="PTID", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
    logger.info(f"Splits created: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # 3. Create datasets and loaders
    train_transform = get_preprocessing_transforms(spatial_shape=(32, 32, 32), voxel_spacing=None, is_training=True)
    eval_transform = get_preprocessing_transforms(spatial_shape=(32, 32, 32), voxel_spacing=None, is_training=False)

    train_ds = BrainMRIDataset(splits["train"], target_column="DX", task_type="classification", transform=train_transform)
    val_ds = BrainMRIDataset(splits["val"], target_column="DX", task_type="classification", transform=eval_transform)
    test_ds = BrainMRIDataset(splits["test"], target_column="DX", task_type="classification", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    # 4. Instantiate Baseline Model
    model = build_model("baseline_resnet18", task_type="classification", output_dim=1, config=config)
    task = ClassificationTask(task_name="AD_vs_CN")
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr=1e-3, total_epochs=1)

    # 5. Profile and execute 1 epoch
    tracker = ComputeTracker(device="cpu")
    tracker.start()

    trainer = BenchmarkTrainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        use_amp=False,
        early_stopping_patience=2,
        experiment_name="stage_a_smoke_run",
        logger=logger,
    )

    fit_summary = trainer.fit(train_loader, val_loader, max_epochs=1)
    eval_results = evaluate_model_on_test_set(model, test_loader, task_type="classification", device="cpu")
    
    elapsed_time = tracker.stop()
    compute_summary = tracker.get_summary(model=model, checkpoint_path=fit_summary["best_checkpoint_path"])

    logger.info(f"Stage A Smoke Test Completed in {elapsed_time:.2f}s!")
    logger.info(f"Test Set Evaluation Results: {eval_results}")
    logger.info(f"Compute Profile: {compute_summary}")

    return {
        "status": "SUCCESS",
        "fit_summary": fit_summary,
        "eval_results": eval_results,
        "compute_summary": compute_summary,
    }


if __name__ == "__main__":
    results = run_stage_a_smoke_test()
    print("Stage A test returned:", results["status"])
