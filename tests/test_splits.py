"""
Unit test for Patient-Level Splitting and Leakage Prevention.
"""

import pandas as pd
import numpy as np
import pytest
from preprocessing.pipeline import create_patient_level_splits


def test_patient_level_split_no_leakage():
    """
    Test that longitudinal scans from the same patient NEVER appear across different splits.
    """
    # Create synthetic longitudinal dataframe with 20 patients and 3 scans each (60 rows)
    records = []
    for p in range(20):
        ptid = f"SUBJ_{p:03d}"
        for v in range(3):
            records.append({
                "PTID": ptid,
                "VISCODE": f"m{v*12}",
                "DX": "AD" if p % 2 == 0 else "CN",
                "AGE": 70.0 + v
            })
    df = pd.DataFrame(records)

    splits = create_patient_level_splits(
        metadata_df=df,
        patient_col="PTID",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    train_pts = set(train_df["PTID"].unique())
    val_pts = set(val_df["PTID"].unique())
    test_pts = set(test_df["PTID"].unique())

    # 1. Total patients accounted for
    all_pts = set(df["PTID"].unique())
    assert train_pts.union(val_pts).union(test_pts) == all_pts

    # 2. Strict ZERO intersection between all sets
    assert len(train_pts.intersection(val_pts)) == 0, "Leakage between Train and Val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Leakage between Train and Test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Leakage between Val and Test!"

    # 3. Check split sizes are non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_invalid_patient_ratios():
    """Verify assertion error if split ratios do not sum to 1.0."""
    df = pd.DataFrame({"PTID": ["P1", "P2", "P3"]})
    with pytest.raises(AssertionError):
        create_patient_level_splits(df, train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
