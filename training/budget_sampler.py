"""
Deterministic Label Budget Subsampler (10%, 25%, 50%, 100%).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def sample_training_budget(
    train_df: pd.DataFrame,
    budget_fraction: float = 1.0,
    patient_col: str = "PTID",
    stratify_col: Optional[str] = "DX",
    seed: int = 42
) -> pd.DataFrame:
    """
    Subsample training data by selecting a deterministic subset of patients.
    Guarantees patient-level consistency and reproducibility.

    Args:
        train_df: Full training partition DataFrame.
        budget_fraction: Fraction in [0.10, 0.25, 0.50, 1.00].
        patient_col: Subject ID column name.
        stratify_col: Optional class column for stratified patient subsampling.
        seed: Random seed.

    Returns:
        Subsampled training DataFrame.
    """
    if np.isclose(budget_fraction, 1.0):
        return train_df.copy().reset_index(drop=True)

    if budget_fraction <= 0.0 or budget_fraction > 1.0:
        raise ValueError(f"budget_fraction must be in (0, 1], got {budget_fraction}")

    unique_patients = train_df[patient_col].unique()
    num_total_patients = len(unique_patients)
    target_count = max(1, int(np.round(num_total_patients * budget_fraction)))

    np.random.seed(seed)

    if stratify_col and stratify_col in train_df.columns:
        # Group by patient to get patient-level diagnosis
        patient_meta = train_df.groupby(patient_col)[stratify_col].first().reset_index()
        sampled_pts = []
        
        # Stratified sampling across patient diagnoses
        for dx_val, group in patient_meta.groupby(stratify_col):
            dx_patients = group[patient_col].values
            shuffled_dx_pts = np.random.permutation(dx_patients)
            n_dx_target = max(1, int(np.round(len(dx_patients) * budget_fraction)))
            sampled_pts.extend(shuffled_dx_pts[:n_dx_target])
            
        selected_patients = set(sampled_pts[:target_count])
    else:
        shuffled = np.random.permutation(unique_patients)
        selected_patients = set(shuffled[:target_count])

    subset_df = train_df[train_df[patient_col].isin(selected_patients)].copy().reset_index(drop=True)
    return subset_df
