"""
Hyperparameter Tuning Engine for Fair Baseline and Foundation Model Optimization.
Guarantees matched tuning budgets without test-set leakage.
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Callable

from utils.seed import set_seed
from utils.logging_utils import get_logger
from training.optimizer import build_optimizer_and_scheduler
from training.trainer import BenchmarkTrainer
from evaluation.profiler import ComputeTracker


DEFAULT_SEARCH_SPACE = {
    "learning_rate": [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3],
    "weight_decay": [1.0e-5, 1.0e-4, 1.0e-3],
    "optimizer": ["adamw", "sgd"],
    "scheduler": ["cosine", "step"],
    "dropout": [0.1, 0.2, 0.3],
}


class HyperparameterTuner:
    """
    Orchestrates systematic hyperparameter search across a predefined budget of trials.
    Model selection is performed strictly using the Validation split.
    """
    def __init__(
        self,
        model_builder: Callable[[Dict[str, Any]], torch.nn.Module],
        task: Any,
        task_type: str = "classification",
        search_space: Optional[Dict[str, List[Any]]] = None,
        max_trials: int = 8,
        epochs_per_trial: int = 25,
        patience: int = 7,
        device: str = "cpu",
        output_dir: str = "results/tuning",
        logger: Optional[Any] = None,
    ):
        self.model_builder = model_builder
        self.task = task
        self.task_type = task_type
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.max_trials = max_trials
        self.epochs_per_trial = epochs_per_trial
        self.patience = patience
        self.device = device
        self.output_dir = output_dir
        self.logger = logger or get_logger("HyperparameterTuner")
        os.makedirs(output_dir, exist_ok=True)

    def generate_trial_configs(self, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Generate deterministic hyperparameter candidates from search space.
        """
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        np.random.seed(seed)
        shuffled_combos = list(np.random.permutation(all_combinations))
        selected_trials = shuffled_combos[:self.max_trials]
        return selected_trials

    def run_tuning(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_prefix: str = "resnet18_ad_vs_cn",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Execute tuning trials and select the best configuration based on validation performance.
        """
        trial_configs = self.generate_trial_configs(seed=seed)
        self.logger.info(f"Starting {len(trial_configs)} tuning trials for {experiment_prefix} on device: {self.device}")

        trial_records = []
        best_config = None
        best_val_metric = -float("inf") if self.task_type == "classification" else float("inf")
        best_checkpoint_path = None

        for trial_idx, hp_config in enumerate(trial_configs):
            trial_name = f"{experiment_prefix}_trial_{trial_idx + 1:02d}"
            self.logger.info(f"--- Running Trial {trial_idx + 1}/{len(trial_configs)} with config: {hp_config} ---")

            set_seed(seed + trial_idx)

            # Build model and optimizer
            model = self.model_builder(hp_config)
            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                lr=hp_config.get("learning_rate", 1e-3),
                weight_decay=hp_config.get("weight_decay", 1e-4),
                optimizer_type=hp_config.get("optimizer", "adamw"),
                scheduler_type=hp_config.get("scheduler", "cosine"),
                total_epochs=self.epochs_per_trial,
            )

            tracker = ComputeTracker(device=self.device)
            tracker.start()

            trainer = BenchmarkTrainer(
                model=model,
                task=self.task,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                use_amp=(self.device == "cuda"),
                early_stopping_patience=self.patience,
                checkpoint_dir=os.path.join(self.output_dir, "checkpoints"),
                experiment_name=trial_name,
                logger=self.logger,
            )

            fit_res = trainer.fit(train_loader, val_loader, max_epochs=self.epochs_per_trial)
            elapsed_sec = tracker.stop()
            compute_summary = tracker.get_summary(model=model, checkpoint_path=fit_res["best_checkpoint_path"])

            # Evaluate best validation score
            val_loss = fit_res["best_val_loss"]
            # For classification, lower val_loss represents higher performance
            # In general, compare on validation loss
            is_new_best = False
            if self.task_type == "classification":
                # Lower val loss is better
                score = -val_loss
                if score > best_val_metric:
                    best_val_metric = score
                    best_config = hp_config
                    best_checkpoint_path = fit_res["best_checkpoint_path"]
                    is_new_best = True
            else:
                # Lower val loss (MSE/MAE) is better
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    best_config = hp_config
                    best_checkpoint_path = fit_res["best_checkpoint_path"]
                    is_new_best = True

            record = {
                "trial_id": trial_idx + 1,
                "trial_name": trial_name,
                "val_loss": val_loss,
                "is_best": is_new_best,
                "total_epochs": fit_res["total_epochs"],
                "elapsed_seconds": round(elapsed_sec, 2),
                "peak_vram_mb": compute_summary.get("peak_vram_mb", 0.0),
                "total_parameters": compute_summary.get("total_parameters", 0),
                **hp_config
            }
            trial_records.append(record)

        # Save tuning log table
        trials_df = pd.DataFrame(trial_records)
        csv_path = os.path.join(self.output_dir, f"{experiment_prefix}_tuning_results.csv")
        trials_df.to_csv(csv_path, index=False)

        manifest = {
            "experiment": experiment_prefix,
            "num_trials": len(trial_records),
            "best_config": best_config,
            "best_val_score": best_val_metric,
            "best_checkpoint_path": best_checkpoint_path,
            "results_csv": csv_path,
        }
        json_path = os.path.join(self.output_dir, f"{experiment_prefix}_tuning_manifest.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Tuning completed! Best Configuration: {best_config} (Score: {best_val_metric:.4f})")
        return manifest
