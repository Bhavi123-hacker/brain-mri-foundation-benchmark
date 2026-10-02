"""
Creates the official stratified patient-level Train/Validation/Test split for the ADNI CN vs AD benchmark.
Strict 0% patient leakage guarantee.
Split allocation:
  - Total: 50 subjects (28 CN, 22 AD)
  - Train: 35 subjects (20 CN, 15 AD, ~70%)
  - Validation: 7 subjects (4 CN, 3 AD, ~14%)
  - Test: 8 subjects (4 CN, 4 AD, 16% - perfectly balanced)
Outputs:
  - results/adni_cn_ad_split.csv
  - results/adni_cn_ad_split.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

def create_adni_splits(
    manifest_path: str = "results/adni_cn_ad_benchmark_manifest.csv",
    output_csv: str = "results/adni_cn_ad_split.csv",
    output_json: str = "results/adni_cn_ad_split.json",
    seed: int = 42,
):
    print("=================================================================")
    print("Creating Stratified Patient-Level Splits for ADNI CN vs AD")
    print("=================================================================")

    assert os.path.exists(manifest_path), f"Missing manifest: {manifest_path}"
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 50, f"Expected 50 subjects, found {len(manifest)}"

    cn_subjects = manifest[manifest["baseline_diagnosis"] == "CN"].sort_values("subject_id").reset_index(drop=True)
    ad_subjects = manifest[manifest["baseline_diagnosis"] == "AD"].sort_values("subject_id").reset_index(drop=True)

    assert len(cn_subjects) == 28, f"Expected 28 CN, got {len(cn_subjects)}"
    assert len(ad_subjects) == 22, f"Expected 22 AD, got {len(ad_subjects)}"

    # Deterministic permutation using explicit random generator
    rng_cn = np.random.RandomState(seed)
    rng_ad = np.random.RandomState(seed)

    cn_idx = rng_cn.permutation(len(cn_subjects))
    ad_idx = rng_ad.permutation(len(ad_subjects))

    # Allocation: Train (20 CN, 15 AD), Val (4 CN, 3 AD), Test (4 CN, 4 AD)
    cn_train = cn_subjects.iloc[cn_idx[:20]].copy(); cn_train["split"] = "train"
    cn_val = cn_subjects.iloc[cn_idx[20:24]].copy(); cn_val["split"] = "val"
    cn_test = cn_subjects.iloc[cn_idx[24:28]].copy(); cn_test["split"] = "test"

    ad_train = ad_subjects.iloc[ad_idx[:15]].copy(); ad_train["split"] = "train"
    ad_val = ad_subjects.iloc[ad_idx[15:18]].copy(); ad_val["split"] = "val"
    ad_test = ad_subjects.iloc[ad_idx[18:22]].copy(); ad_test["split"] = "test"

    split_df = pd.concat([cn_train, ad_train, cn_val, ad_val, cn_test, ad_test]).sort_values("subject_id").reset_index(drop=True)

    # Verification of zero leakage
    train_pts = set(split_df[split_df["split"] == "train"]["subject_id"])
    val_pts = set(split_df[split_df["split"] == "val"]["subject_id"])
    test_pts = set(split_df[split_df["split"] == "test"]["subject_id"])

    assert len(train_pts & val_pts) == 0, "LEAKAGE: Train and Val overlap!"
    assert len(train_pts & test_pts) == 0, "LEAKAGE: Train and Test overlap!"
    assert len(val_pts & test_pts) == 0, "LEAKAGE: Val and Test overlap!"
    assert len(train_pts) == 35, f"Expected 35 train pts, got {len(train_pts)}"
    assert len(val_pts) == 7, f"Expected 7 val pts, got {len(val_pts)}"
    assert len(test_pts) == 8, f"Expected 8 test pts, got {len(test_pts)}"

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    split_df.to_csv(output_csv, index=False)
    print(f"Saved split CSV to {output_csv}")

    # Build summary JSON
    def get_split_stats(sub_df):
        return {
            "total_subjects": len(sub_df),
            "class_distribution": sub_df["baseline_diagnosis"].value_counts().to_dict(),
            "sex_distribution": sub_df["sex"].value_counts().to_dict(),
            "mean_age": round(float(sub_df["age"].mean()), 2) if sub_df["age"].notnull().any() else None,
            "subjects": sorted(sub_df["subject_id"].tolist()),
        }

    summary = {
        "split_name": "adni_cn_ad_stratified_seed42",
        "random_seed": seed,
        "total_subjects": len(split_df),
        "cohort_breakdown": {
            "train": get_split_stats(split_df[split_df["split"] == "train"]),
            "validation": get_split_stats(split_df[split_df["split"] == "val"]),
            "test": get_split_stats(split_df[split_df["split"] == "test"]),
        },
        "patient_leakage_audit": {
            "train_val_overlap": len(train_pts & val_pts),
            "train_test_overlap": len(train_pts & test_pts),
            "val_test_overlap": len(val_pts & test_pts),
            "passed": True,
        },
        "locked_test_set": {
            "is_locked": True,
            "subject_count": len(test_pts),
            "subjects": sorted(list(test_pts)),
            "instructions": "DO NOT evaluate this test set until Day 4 final benchmark evaluation.",
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved split summary JSON to {output_json}")

if __name__ == "__main__":
    create_adni_splits()
