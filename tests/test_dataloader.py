"""
Unit tests for DataLoader Factory and Budget-Aware Batching.
"""

import os
import torch
import pandas as pd
import pytest
from preprocessing.dataloader import create_benchmark_dataloaders
from preprocessing.mock_data import generate_mock_dataset
from preprocessing.split_manager import PersistentSplitManager


@pytest.fixture(scope="module")
def mock_splits(tmp_path_factory):
    tmp_dir = str(tmp_path_factory.mktemp("loader_data"))
    csv_path = generate_mock_dataset(
        output_dir=tmp_dir,
        num_patients=12,
        scans_per_patient=2,
        volume_shape=(16, 16, 16),
        seed=42
    )
    df = pd.read_csv(csv_path)
    df["subject_id"] = df["PTID"]
    df["file_path"] = df["FILE_PATH"]
    # Ensure binary classes CN and AD only
    df["diagnosis"] = ["AD" if i % 2 == 0 else "CN" for i in range(len(df))]
    df["age"] = df["AGE"]
    df["is_valid_scan"] = True

    split_mgr = PersistentSplitManager(splits_dir=os.path.join(tmp_dir, "splits"))
    splits = split_mgr.create_and_persist_splits(
        df=df,
        dataset_name="MOCK_LOADER",
        task_name="AD_vs_CN",
        subject_col="subject_id",
        seed=42
    )
    return splits


def test_create_benchmark_dataloaders(mock_splits):
    """Verify DataLoaders yield expected batch shapes and targets."""
    task_config = {
        "task_type": "classification",
        "target_column": "diagnosis",
        "positive_class": "AD",
        "negative_class": "CN",
    }

    train_loader, val_loader, test_loader = create_benchmark_dataloaders(
        splits=mock_splits,
        task_config=task_config,
        batch_size=2,
        spatial_shape=(16, 16, 16),
        label_budget=1.0,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    batch_vols, batch_targets, batch_meta = next(iter(train_loader))
    assert batch_vols.ndim == 5 # (B, C, D, H, W)
    assert batch_vols.shape == (2, 1, 16, 16, 16)
    assert batch_targets.shape == (2, 1)


def test_budget_dataloader_reduction(mock_splits):
    """Verify budget subsampling reduces training batches while keeping val/test fixed."""
    task_config = {
        "task_type": "classification",
        "target_column": "diagnosis",
        "positive_class": "AD",
        "negative_class": "CN",
    }

    train_100, val_100, test_100 = create_benchmark_dataloaders(
        splits=mock_splits, task_config=task_config, batch_size=2, label_budget=1.0
    )
    train_50, val_50, test_50 = create_benchmark_dataloaders(
        splits=mock_splits, task_config=task_config, batch_size=2, label_budget=0.50
    )

    assert len(train_50.dataset) <= len(train_100.dataset)
    assert len(val_50.dataset) == len(val_100.dataset)
    assert len(test_50.dataset) == len(test_100.dataset)
