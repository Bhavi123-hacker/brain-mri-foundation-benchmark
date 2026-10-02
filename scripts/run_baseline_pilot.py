"""
Phase 3 Supervised 3D ResNet Baseline Pilot & Verification Engine.
Validates Task 1 (AD vs CN Classification) and Task 2 (Brain-Age Regression)
end-to-end on synthetic 3D phantoms with compute profiling and metric logging.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List
import pandas as pd
import torch

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_parser import load_config
from utils.seed import set_seed
from utils.logging_utils import get_logger, setup_experiment_logging
from preprocessing.mock_data import generate_mock_dataset
from preprocessing.split_manager import PersistentSplitManager
from preprocessing.dataloader import create_benchmark_dataloaders
from models.baseline.resnet3d import build_resnet18_3d
from tasks import ClassificationTask, RegressionTask
from training import build_optimizer_and_scheduler, BenchmarkTrainer, HyperparameterTuner
from evaluation import evaluate_model_on_test_set, ComputeTracker
from visualization.leaderboard import BenchmarkLeaderboard


def run_baseline_pilot(config_dir: str = "configs", output_dir: str = "results") -> Dict[str, Any]:
    logger = get_logger("BaselinePilotRunner")
    logger.info("=================================================================")
    logger.info("Initializing Phase 3 Supervised 3D ResNet Baseline Pilot Runs")
    logger.info("=================================================================")

    config = load_config(config_dir)
    os.makedirs(output_dir, exist_ok=True)
    leaderboard = BenchmarkLeaderboard(output_csv=os.path.join(output_dir, "leaderboard.csv"))
    split_manager = PersistentSplitManager(splits_dir="data/splits")

    # 1. Generate synthetic mock dataset for software verification
    logger.info("Generating synthetic verification cohort (32x32x32 phantoms)...")
    mock_csv = generate_mock_dataset(
        output_dir="data/mock_data",
        num_patients=12,
        scans_per_patient=2,
        volume_shape=(32, 32, 32),
        seed=42
    )
    df = pd.read_csv(mock_csv)
    df["subject_id"] = df["PTID"]
    df["file_path"] = df["FILE_PATH"]
    df["diagnosis"] = df["DX"]
    df["age"] = df["AGE"]
    df["is_valid_scan"] = True

    pilot_results = {}

    # =========================================================================
    # TASK 1: AD vs CN Binary Classification
    # =========================================================================
    logger.info("\n>>> Executing Task 1: AD vs CN Binary Classification (3D ResNet-18) <<<")
    set_seed(42)
    
    ad_cn_df = df[df["diagnosis"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    task1_splits = split_manager.create_and_persist_splits(
        df=ad_cn_df,
        dataset_name="SYNTHETIC_MOCK",
        task_name="AD_vs_CN",
        subject_col="subject_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    task1_cfg = config["tasks"]["ad_vs_cn"]
    train_loader1, val_loader1, test_loader1 = create_benchmark_dataloaders(
        splits=task1_splits,
        task_config=task1_cfg,
        batch_size=2,
        spatial_shape=(32, 32, 32),
        label_budget=1.0,
        seed=42
    )

    model1 = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2)
    task1 = ClassificationTask(task_name="AD_vs_CN")
    optimizer1, scheduler1 = build_optimizer_and_scheduler(
        model1, lr=1.0e-3, weight_decay=1.0e-4, optimizer_type="adamw", scheduler_type="cosine", total_epochs=3
    )

    tracker1 = ComputeTracker(device="cpu")
    tracker1.start()

    trainer1 = BenchmarkTrainer(
        model=model1,
        task=task1,
        optimizer=optimizer1,
        scheduler=scheduler1,
        device="cpu",
        use_amp=False,
        early_stopping_patience=3,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        experiment_name="baseline_resnet18_ad_vs_cn_mock",
        logger=logger,
    )

    fit_summary1 = trainer1.fit(train_loader1, val_loader1, max_epochs=3)
    eval_metrics1 = evaluate_model_on_test_set(model1, test_loader1, task_type="classification", device="cpu")
    elapsed_sec1 = tracker1.stop()
    compute_summary1 = tracker1.get_summary(model=model1, checkpoint_path=fit_summary1["best_checkpoint_path"])

    leaderboard.add_entry(
        model_name="3D-ResNet-18",
        dataset_name="SYNTHETIC_MOCK",
        task_name="AD_vs_CN",
        label_budget=1.0,
        seed=42,
        metrics=eval_metrics1,
        compute=compute_summary1,
    )

    logger.info(f"Task 1 Test Metrics: {eval_metrics1}")
    logger.info(f"Task 1 Compute Profile: {compute_summary1}")
    pilot_results["Task_1_AD_vs_CN"] = {
        "fit_summary": fit_summary1,
        "test_metrics": eval_metrics1,
        "compute_profile": compute_summary1,
    }

    # =========================================================================
    # TASK 2: Brain-Age Continuous Regression
    # =========================================================================
    logger.info("\n>>> Executing Task 2: Brain-Age Continuous Regression (3D ResNet-18) <<<")
    set_seed(42)

    age_df = df.dropna(subset=["age"]).copy().reset_index(drop=True)
    task2_splits = split_manager.create_and_persist_splits(
        df=age_df,
        dataset_name="SYNTHETIC_MOCK",
        task_name="Brain_Age",
        subject_col="subject_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    task2_cfg = config["tasks"]["brain_age"]
    train_loader2, val_loader2, test_loader2 = create_benchmark_dataloaders(
        splits=task2_splits,
        task_config=task2_cfg,
        batch_size=2,
        spatial_shape=(32, 32, 32),
        label_budget=1.0,
        seed=42
    )

    model2 = build_resnet18_3d(output_dim=1, task_type="regression", dropout=0.2)
    task2 = RegressionTask(task_name="Brain_Age", loss_type="MSELoss")
    optimizer2, scheduler2 = build_optimizer_and_scheduler(
        model2, lr=1.0e-3, weight_decay=1.0e-4, optimizer_type="adamw", scheduler_type="cosine", total_epochs=3
    )

    tracker2 = ComputeTracker(device="cpu")
    tracker2.start()

    trainer2 = BenchmarkTrainer(
        model=model2,
        task=task2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        device="cpu",
        use_amp=False,
        early_stopping_patience=3,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        experiment_name="baseline_resnet18_brain_age_mock",
        logger=logger,
    )

    fit_summary2 = trainer2.fit(train_loader2, val_loader2, max_epochs=3)
    eval_metrics2 = evaluate_model_on_test_set(model2, test_loader2, task_type="regression", device="cpu")
    elapsed_sec2 = tracker2.stop()
    compute_summary2 = tracker2.get_summary(model=model2, checkpoint_path=fit_summary2["best_checkpoint_path"])

    leaderboard.add_entry(
        model_name="3D-ResNet-18",
        dataset_name="SYNTHETIC_MOCK",
        task_name="Brain_Age",
        label_budget=1.0,
        seed=42,
        metrics=eval_metrics2,
        compute=compute_summary2,
    )

    logger.info(f"Task 2 Test Metrics: {eval_metrics2}")
    logger.info(f"Task 2 Compute Profile: {compute_summary2}")
    pilot_results["Task_2_Brain_Age"] = {
        "fit_summary": fit_summary2,
        "test_metrics": eval_metrics2,
        "compute_profile": compute_summary2,
    }

    # =========================================================================
    # HYPERPARAMETER TUNER VERIFICATION (2 Mini-Trials on Validation Set)
    # =========================================================================
    logger.info("\n>>> Verifying Hyperparameter Tuning Infrastructure (2-Trial Mini-Sweep) <<<")
    tuner = HyperparameterTuner(
        model_builder=lambda hp: build_resnet18_3d(output_dim=1, task_type="classification", dropout=hp.get("dropout", 0.2)),
        task=task1,
        task_type="classification",
        search_space={
            "learning_rate": [1.0e-3, 3.0e-4],
            "weight_decay": [1.0e-4],
            "optimizer": ["adamw"],
            "scheduler": ["cosine"],
            "dropout": [0.2],
        },
        max_trials=2,
        epochs_per_trial=2,
        patience=2,
        device="cpu",
        output_dir=os.path.join(output_dir, "tuning"),
        logger=logger,
    )

    tuning_manifest = tuner.run_tuning(
        train_loader=train_loader1,
        val_loader=val_loader1,
        experiment_prefix="resnet18_ad_vs_cn_test_tune",
        seed=42,
    )
    pilot_results["Tuning_Infrastructure_Verification"] = tuning_manifest

    # Save Leaderboard
    leaderboard_df = leaderboard.save()
    logger.info(f"\n=================================================================")
    logger.info(f"Leaderboard Updated:\n{leaderboard.to_markdown()}")
    logger.info(f"=================================================================")

    # Save complete Phase 3 pilot report JSON
    report_path = os.path.join(output_dir, "phase3_baseline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pilot_results, f, indent=2)

    logger.info(f"Phase 3 Baseline Pilot Report saved to: {report_path}")
    return pilot_results


if __name__ == "__main__":
    run_baseline_pilot()
