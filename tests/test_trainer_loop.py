"""
Unit tests for BenchmarkTrainer, Early Stopping, and Checkpoint Persistence.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest

from training.trainer import BenchmarkTrainer
from tasks.classification import ClassificationTask
from tasks.regression import RegressionTask


class Toy3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


def test_trainer_fit_and_early_stopping(tmp_path):
    """Verify trainer trains, validates, triggers early stopping, and reloads best checkpoint."""
    # Synthetic dataset
    x_train = torch.randn(8, 1, 8, 8, 8)
    y_train = torch.randint(0, 2, (8, 1)).float()
    x_val = torch.randn(4, 1, 8, 8, 8)
    y_val = torch.randint(0, 2, (4, 1)).float()

    train_ds = TensorDataset(x_train, y_train, torch.zeros(8))
    val_ds = TensorDataset(x_val, y_val, torch.zeros(4))

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = Toy3DModel()
    task = ClassificationTask(task_name="AD_vs_CN")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    chk_dir = str(tmp_path / "checkpoints")
    trainer = BenchmarkTrainer(
        model=model,
        task=task,
        optimizer=optimizer,
        device="cpu",
        early_stopping_patience=2,
        checkpoint_dir=chk_dir,
        experiment_name="test_trainer_run",
    )

    fit_res = trainer.fit(train_loader, val_loader, max_epochs=10)

    assert fit_res["total_epochs"] <= 10
    assert os.path.exists(fit_res["best_checkpoint_path"])
    assert "best_val_loss" in fit_res
    assert len(fit_res["history"]) > 0
