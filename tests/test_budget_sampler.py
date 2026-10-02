"""
Unit test for Label Budget Subsampling across 10%, 25%, 50%, and 100%.
"""

import pandas as pd
import numpy as np
import pytest
from training.budget_sampler import sample_training_budget


@pytest.fixture
def dummy_train_df():
    """Create synthetic training DataFrame with 40 patients."""
    records = []
    for p in range(40):
        ptid = f"PT_{p:03d}"
        for v in range(2):
            records.append({
                "PTID": ptid,
                "DX": "AD" if p % 2 == 0 else "CN",
                "AGE": 65.0 + (p % 15)
            })
    return pd.DataFrame(records)


def test_budget_fractions(dummy_train_df):
    """Test subsampling across all 4 benchmark budget fractions."""
    total_pts = dummy_train_df["PTID"].nunique() # 40 patients

    for frac in [0.10, 0.25, 0.50, 1.00]:
        subset_df = sample_training_budget(
            train_df=dummy_train_df,
            budget_fraction=frac,
            patient_col="PTID",
            stratify_col="DX",
            seed=42
        )
        subset_pts = subset_df["PTID"].nunique()

        if frac == 1.00:
            assert subset_pts == total_pts
            assert len(subset_df) == len(dummy_train_df)
        else:
            expected_pts = int(np.round(total_pts * frac))
            assert subset_pts == expected_pts
            assert len(subset_df) < len(dummy_train_df)


def test_budget_reproducibility(dummy_train_df):
    """Verify that same seed produces identical patient subsets."""
    sub1 = sample_training_budget(dummy_train_df, budget_fraction=0.25, seed=123)
    sub2 = sample_training_budget(dummy_train_df, budget_fraction=0.25, seed=123)
    pd.testing.assert_frame_equal(sub1, sub2)


def test_invalid_budget(dummy_train_df):
    """Verify ValueError on invalid fractions."""
    with pytest.raises(ValueError):
        sample_training_budget(dummy_train_df, budget_fraction=0.0)
    with pytest.raises(ValueError):
        sample_training_budget(dummy_train_df, budget_fraction=1.5)
