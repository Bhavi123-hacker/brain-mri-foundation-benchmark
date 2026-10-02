"""
Unit tests for HyperparameterTuner and Fair Optimization Sweep.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest

from training.tuner import HyperparameterTuner
from tasks.classification import ClassificationTask


class Small3DCNN(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(self.pool(self.drop(self.conv(x))).flatten(1))


def test_hyperparameter_tuner(tmp_path):
    """Verify tuner runs systematic trials and saves results table without touching test set."""
    x_train = torch.randn(6, 1, 8, 8, 8)
    y_train = torch.randint(0, 2, (6, 1)).float()
    x_val = torch.randn(4, 1, 8, 8, 8)
    y_val = torch.randint(0, 2, (4, 1)).float()

    train_ds = TensorDataset(x_train, y_train, torch.zeros(6))
    val_ds = TensorDataset(x_val, y_val, torch.zeros(4))

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    task = ClassificationTask(task_name="AD_vs_CN")
    out_dir = str(tmp_path / "tuning_out")

    tuner = HyperparameterTuner(
        model_builder=lambda hp: Small3DCNN(dropout=hp.get("dropout", 0.2)),
        task=task,
        task_type="classification",
        search_space={
            "learning_rate": [1e-3, 3e-4],
            "weight_decay": [1e-4],
            "optimizer": ["adamw"],
            "scheduler": ["cosine"],
            "dropout": [0.1, 0.2],
        },
        max_trials=2,
        epochs_per_trial=2,
        patience=2,
        device="cpu",
        output_dir=out_dir,
    )

    manifest = tuner.run_tuning(
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_prefix="unit_test_tune",
        seed=42,
    )

    assert manifest["num_trials"] == 2
    assert manifest["best_config"] is not None
    assert os.path.exists(manifest["results_csv"])
    assert os.path.exists(manifest["best_checkpoint_path"])
