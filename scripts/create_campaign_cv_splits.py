"""
Stage 2: Development-Set Cross-Validation Strategy.
Partitions the 42 development subjects (Train=35 + Val=7) into stratified 5-fold CV splits
across multiple seeds while strictly isolating the 8-subject holdout test set.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger


def create_cv_splits(
    data_dir: str = "D:\\OASIS1_PILOT",
    output_dir: str = "results/final_campaign/cv_splits",
    seeds: list = [42, 43, 44],
    n_splits: int = 5,
):
    logger = get_logger("CampaignCVSplits")
    logger.info("=================================================================")
    logger.info("Creating Stratified 5-Fold Cross-Validation Splits on Development Set")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load original OASIS splits
    splits_path = os.path.join(data_dir, "pilot_splits.csv")
    df = pd.read_csv(splits_path)
    df = df[~df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)

    dev_df = df[df["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    logger.info(f"Total Cohort: {len(df)} subjects")
    logger.info(f"Development Set: {len(dev_df)} subjects ({dev_df['label_name'].value_counts().to_dict()})")
    logger.info(f"Locked Holdout Test Set: {len(test_df)} subjects ({test_df['label_name'].value_counts().to_dict()})")

    assert len(dev_df) == 42, f"Expected 42 development subjects, got {len(dev_df)}"
    assert len(test_df) == 8, f"Expected 8 test subjects, got {len(test_df)}"

    verification_records = []

    # 2. Generate Repeated Stratified K-Fold Splits
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_records = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df["label"])):
            fold_train = dev_df.iloc[train_idx].copy()
            fold_val = dev_df.iloc[val_idx].copy()

            # Verify 0% patient leakage
            train_sids = set(fold_train["ID"])
            val_sids = set(fold_val["ID"])
            test_sids = set(test_df["ID"])

            overlap_tv = train_sids.intersection(val_sids)
            overlap_tt = train_sids.intersection(test_sids)
            overlap_vt = val_sids.intersection(test_sids)

            assert len(overlap_tv) == 0, f"Leakage detected between train and val in fold {fold_idx}!"
            assert len(overlap_tt) == 0, f"Leakage detected between train and test in fold {fold_idx}!"
            assert len(overlap_vt) == 0, f"Leakage detected between val and test in fold {fold_idx}!"

            for _, row in fold_train.iterrows():
                fold_records.append({
                    "seed": seed,
                    "fold": fold_idx,
                    "split_type": "train",
                    "subject_id": row["ID"],
                    "mri_file": row["mri_file"],
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "age": row.get("Age", np.nan),
                    "cdr": row.get("CDR", np.nan),
                })

            for _, row in fold_val.iterrows():
                fold_records.append({
                    "seed": seed,
                    "fold": fold_idx,
                    "split_type": "val",
                    "subject_id": row["ID"],
                    "mri_file": row["mri_file"],
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "age": row.get("Age", np.nan),
                    "cdr": row.get("CDR", np.nan),
                })

        fold_df = pd.DataFrame(fold_records)
        csv_path = os.path.join(output_dir, f"oasis_dev_cv_5fold_seed{seed}.csv")
        fold_df.to_csv(csv_path, index=False)
        logger.info(f"Saved 5-fold CV splits for seed {seed} to {csv_path}")

        verification_records.append({
            "seed": seed,
            "n_splits": n_splits,
            "dev_subjects": len(dev_df),
            "fold_sizes_train": [int(np.sum((fold_df["fold"] == f) & (fold_df["split_type"] == "train"))) for f in range(n_splits)],
            "fold_sizes_val": [int(np.sum((fold_df["fold"] == f) & (fold_df["split_type"] == "val"))) for f in range(n_splits)],
            "leakage_train_val": False,
            "leakage_dev_test": False,
            "csv_path": csv_path,
        })

    # Save verification report
    report_path = os.path.join(output_dir, "split_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "verification_status": "PASS",
            "non_negotiable_leakage_rule": "0% Patient Overlap Verified",
            "locked_test_holdout": "Strictly Isolated (N=8 subjects untouched)",
            "cv_splits": verification_records,
        }, f, indent=2)
    logger.info(f"Saved verification report to {report_path}")


if __name__ == "__main__":
    create_cv_splits()
