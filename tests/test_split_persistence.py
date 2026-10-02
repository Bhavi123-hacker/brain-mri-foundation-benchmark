"""
Unit tests for Persistent Split Management and Multi-Model Reproducibility.
"""

import os
import pandas as pd
import pytest
from preprocessing.split_manager import PersistentSplitManager


def test_persistent_splits_creation_and_reloading(tmp_path):
    """Verify split files are saved, manifests created, and reloaded accurately."""
    splits_dir = str(tmp_path / "splits")
    manager = PersistentSplitManager(splits_dir=splits_dir)

    # 10 subjects, 2 scans each
    records = []
    for s in range(10):
        sid = f"SUBJ_{s:03d}"
        for v in range(2):
            records.append({
                "subject_id": sid,
                "scan_id": f"{sid}_v{v}",
                "file_path": f"/dummy/{sid}_v{v}.nii",
                "diagnosis": "AD" if s % 2 == 0 else "CN",
                "age": 70.0 + s,
                "is_valid_scan": True,
            })
    df = pd.DataFrame(records)

    # Create & Persist
    splits = manager.create_and_persist_splits(
        df=df,
        dataset_name="TEST_DATASET",
        task_name="AD_vs_CN",
        subject_col="subject_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    assert "train" in splits and "val" in splits and "test" in splits
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0

    # Leakage assertion
    leakage = manager.check_leakage(splits, subject_col="subject_id")
    assert leakage["has_leakage"] is False

    # Reload from disk
    loaded_splits = manager.load_existing_splits(dataset_name="TEST_DATASET", task_name="AD_vs_CN")
    assert loaded_splits is not None
    pd.testing.assert_frame_equal(splits["train"], loaded_splits["train"])
    pd.testing.assert_frame_equal(splits["val"], loaded_splits["val"])
    pd.testing.assert_frame_equal(splits["test"], loaded_splits["test"])
