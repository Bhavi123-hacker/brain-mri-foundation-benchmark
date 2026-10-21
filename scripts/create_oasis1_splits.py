"""
Step 3: Generate and Lock Patient-Level Splits for OASIS-1 Primary Cohort (N = 198).

Generates deterministic, stratified 70/15/15 Train / Validation / Locked-Test splits.
Strict 0% patient leakage guarantee.

Outputs saved outside Git to:
- D:\\brain_mri_data\\OASIS1\\manifests\\oasis1_primary_cohort_splits.csv
- D:\\brain_mri_data\\OASIS1\\manifests\\oasis1_primary_cohort_splits_summary.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Inputs
COHORT_MANIFEST = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_manifest.csv"
SCANS_MANIFEST = r"D:\brain_mri_data\OASIS1\manifests\oasis1_extracted_scans_manifest.csv"
INTEGRITY_AUDIT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_scan_integrity_audit.csv"

# Outputs
OUTPUT_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OUTPUT_SUMMARY_JSON = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits_summary.json"

RANDOM_SEED = 42

def create_oasis1_splits():
    start_time = time.time()
    print("=" * 80)
    print("STEP 3: GENERATE AND LOCK PATIENT-LEVEL SPLITS (N = 198)")
    print("=" * 80)
    
    # 1. Guard against accidental overwrite
    if os.path.exists(OUTPUT_SPLITS_CSV):
        raise FileExistsError(f"Target split CSV already exists: {OUTPUT_SPLITS_CSV}. Refusing to overwrite.")
    if os.path.exists(OUTPUT_SUMMARY_JSON):
        raise FileExistsError(f"Target summary JSON already exists: {OUTPUT_SUMMARY_JSON}. Refusing to overwrite.")
        
    # 2. Reconcile source manifests
    assert os.path.exists(COHORT_MANIFEST), f"Missing: {COHORT_MANIFEST}"
    assert os.path.exists(SCANS_MANIFEST), f"Missing: {SCANS_MANIFEST}"
    assert os.path.exists(INTEGRITY_AUDIT), f"Missing: {INTEGRITY_AUDIT}"
    
    cohort_df = pd.read_csv(COHORT_MANIFEST).sort_values("session_id").reset_index(drop=True)
    scans_df = pd.read_csv(SCANS_MANIFEST).sort_values("session_id").reset_index(drop=True)
    audit_df = pd.read_csv(INTEGRITY_AUDIT)
    
    print(f"Loaded cohort manifest: {len(cohort_df)} records")
    print(f"Loaded scans manifest:  {len(scans_df)} records")
    print(f"Loaded integrity audit: {len(audit_df)} records")
    
    # Validate cohort size
    assert len(cohort_df) == 198, f"Expected 198 cohort records, found {len(cohort_df)}"
    assert len(scans_df) == 198, f"Expected 198 scans records, found {len(scans_df)}"
    
    # Verify stable subject and session IDs
    cohort_sids = set(cohort_df["session_id"].str.strip())
    scans_sids = set(scans_df["session_id"].str.strip())
    assert cohort_sids == scans_sids, "Session ID mismatch between cohort and scans manifest!"
    
    # Check T88_111 primary representation integrity audit
    t88_audit = audit_df[audit_df["representation"] == "T88_111"].copy()
    assert len(t88_audit) == 198, f"Expected 198 T88_111 audit records, found {len(t88_audit)}"
    t88_audit_sids = set(t88_audit["session_id"].str.strip())
    assert cohort_sids == t88_audit_sids, "Session ID mismatch between cohort and T88_111 audit!"
    
    valid_t88 = t88_audit[t88_audit["status"] == "VALID"]
    assert len(valid_t88) == 198, f"Expected 198 valid T88_111 scans, found {len(valid_t88)}"
    print("Reconciliation confirmed: All 198 subjects have verified valid T88_111 scans.")
    
    # Verify binary label definitions
    # CN = 0, Dementia = 1
    assert set(cohort_df["binary_label"].unique()) == {0.0, 1.0}
    cn_count = (cohort_df["binary_label"] == 0.0).sum()
    dem_count = (cohort_df["binary_label"] == 1.0).sum()
    assert cn_count == 98, f"Expected 98 CN, got {cn_count}"
    assert dem_count == 100, f"Expected 100 Dementia, got {dem_count}"
    print(f"Cohort diagnosis labels confirmed: CN = {cn_count} (49.5%), Dementia = {dem_count} (50.5%)")
    
    # Merge required columns
    merged = cohort_df.copy()
    merged = merged.merge(
        scans_df[["session_id", "t88_hdr_path", "t88_img_path", "sbj_hdr_path", "sbj_img_path"]],
        on="session_id"
    )
    
    # 3. Generate deterministic, stratified 70/15/15 split
    # Step 3a: Split 70% train vs 30% temp (stratified by binary_label)
    train_df, temp_df = train_test_split(
        merged,
        test_size=0.30,
        stratify=merged["binary_label"],
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Step 3b: Split temp 50/50 into validation (15%) and test (15%) (stratified by binary_label)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["binary_label"],
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Assign split column
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"
    
    # Combine and sort deterministically
    splits_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    splits_df = splits_df.sort_values("session_id").reset_index(drop=True)
    splits_df["random_seed"] = RANDOM_SEED
    splits_df["primary_representation"] = "T88_111"
    
    # 4. Rigorous assertions: leakage, counts, and coverage
    train_subjects = set(splits_df[splits_df["split"] == "train"]["base_subject_id"])
    val_subjects = set(splits_df[splits_df["split"] == "val"]["base_subject_id"])
    test_subjects = set(splits_df[splits_df["split"] == "test"]["base_subject_id"])
    
    # Disjoint sets check (zero leakage)
    assert len(train_subjects & val_subjects) == 0, f"LEAKAGE: Train & Val overlap: {train_subjects & val_subjects}"
    assert len(train_subjects & test_subjects) == 0, f"LEAKAGE: Train & Test overlap: {train_subjects & test_subjects}"
    assert len(val_subjects & test_subjects) == 0, f"LEAKAGE: Val & Test overlap: {val_subjects & test_subjects}"
    
    # Session-level check
    train_sids = set(splits_df[splits_df["split"] == "train"]["session_id"])
    val_sids = set(splits_df[splits_df["split"] == "val"]["session_id"])
    test_sids = set(splits_df[splits_df["split"] == "test"]["session_id"])
    assert len(train_sids & val_sids) == 0, "LEAKAGE: Train & Val session overlap"
    assert len(train_sids & test_sids) == 0, "LEAKAGE: Train & Test session overlap"
    assert len(val_sids & test_sids) == 0, "LEAKAGE: Val & Test session overlap"
    
    # Complete coverage check
    assert (train_subjects | val_subjects | test_subjects) == set(cohort_df["base_subject_id"])
    assert len(splits_df) == 198
    
    # Actual split counts
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    
    print("\n--- Split Allocation Counts ---")
    print(f"  Train:      {n_train} subjects ({n_train/198*100:.2f}%)")
    print(f"  Validation: {n_val} subjects ({n_val/198*100:.2f}%)")
    print(f"  Test:       {n_test} subjects ({n_test/198*100:.2f}%)")
    print(f"  Total:      {len(splits_df)} subjects (100.0%)")
    
    # Calculate demographic and clinical statistics by split
    def compute_split_metrics(sub_df):
        cn_sub = (sub_df["binary_label"] == 0.0).sum()
        dem_sub = (sub_df["binary_label"] == 1.0).sum()
        females = (sub_df["gender"] == "F").sum()
        males = (sub_df["gender"] == "M").sum()
        cdr_counts = sub_df["cdr"].value_counts().sort_index().to_dict()
        
        return {
            "total_subjects": len(sub_df),
            "percentage_of_cohort": round(len(sub_df) / 198 * 100, 2),
            "class_distribution": {
                "CN_0": int(cn_sub),
                "Dementia_1": int(dem_sub),
                "CN_percentage": round(cn_sub / len(sub_df) * 100, 2),
                "Dementia_percentage": round(dem_sub / len(sub_df) * 100, 2)
            },
            "cdr_distribution": {str(k): int(v) for k, v in cdr_counts.items()},
            "sex_distribution": {
                "Female": int(females),
                "Male": int(males),
                "Female_percentage": round(females / len(sub_df) * 100, 2),
                "Male_percentage": round(males / len(sub_df) * 100, 2)
            },
            "age_statistics": {
                "mean": round(float(sub_df["age"].mean()), 2),
                "std": round(float(sub_df["age"].std()), 2),
                "min": float(sub_df["age"].min()),
                "max": float(sub_df["age"].max())
            },
            "subject_ids": sorted(sub_df["base_subject_id"].tolist()),
            "session_ids": sorted(sub_df["session_id"].tolist()),
        }

    train_stats = compute_split_metrics(train_df)
    val_stats = compute_split_metrics(val_df)
    test_stats = compute_split_metrics(test_df)
    
    # Print profile for review
    for name, s in [("TRAIN", train_stats), ("VALIDATION", val_stats), ("LOCKED TEST", test_stats)]:
        print(f"\n--- {name} Split Profile ---")
        print(f"  Count:  {s['total_subjects']} ({s['percentage_of_cohort']}%)")
        print(f"  Class:  CN={s['class_distribution']['CN_0']} ({s['class_distribution']['CN_percentage']}%), "
              f"Dementia={s['class_distribution']['Dementia_1']} ({s['class_distribution']['Dementia_percentage']}%)")
        print(f"  CDR:    {s['cdr_distribution']}")
        print(f"  Sex:    F={s['sex_distribution']['Female']} ({s['sex_distribution']['Female_percentage']}%), "
              f"M={s['sex_distribution']['Male']} ({s['sex_distribution']['Male_percentage']}%)")
        print(f"  Age:    {s['age_statistics']['mean']} ± {s['age_statistics']['std']} "
              f"[{s['age_statistics']['min']} - {s['age_statistics']['max']}]")

    # 5. Save Split Artifacts
    splits_df.to_csv(OUTPUT_SPLITS_CSV, index=False)
    print(f"\nSaved split CSV to: {OUTPUT_SPLITS_CSV}")
    
    summary_dict = {
        "split_name": "oasis1_primary_cohort_stratified_70_15_15",
        "description": "Deterministic, stratified 70/15/15 patient-level split for OASIS-1 primary cohort (age >= 60).",
        "random_seed": RANDOM_SEED,
        "splitting_method": "sklearn.model_selection.train_test_split (stratified by binary_label)",
        "primary_representation": "T88_111",
        "total_cohort_size": 198,
        "patient_leakage_audit": {
            "train_val_overlap_count": len(train_subjects & val_subjects),
            "train_test_overlap_count": len(train_subjects & test_subjects),
            "val_test_overlap_count": len(val_subjects & test_subjects),
            "disjoint_sets_verified": True,
            "patient_leakage_percentage": 0.0,
            "passed": True
        },
        "split_breakdown": {
            "train": train_stats,
            "validation": val_stats,
            "locked_test": test_stats
        },
        "demographic_balance_analysis": {
            "age_mean_comparison": {
                "train": train_stats["age_statistics"]["mean"],
                "validation": val_stats["age_statistics"]["mean"],
                "locked_test": test_stats["age_statistics"]["mean"]
            },
            "female_percentage_comparison": {
                "train": train_stats["sex_distribution"]["Female_percentage"],
                "validation": val_stats["sex_distribution"]["Female_percentage"],
                "locked_test": test_stats["sex_distribution"]["Female_percentage"]
            },
            "dementia_percentage_comparison": {
                "train": train_stats["class_distribution"]["Dementia_percentage"],
                "validation": val_stats["class_distribution"]["Dementia_percentage"],
                "locked_test": test_stats["class_distribution"]["Dementia_percentage"]
            }
        },
        "source_manifests": {
            "cohort_manifest": COHORT_MANIFEST,
            "scans_manifest": SCANS_MANIFEST,
            "integrity_audit": INTEGRITY_AUDIT
        },
        "output_files": {
            "split_csv": OUTPUT_SPLITS_CSV,
            "summary_json": OUTPUT_SUMMARY_JSON
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed_seconds": round(time.time() - start_time, 2)
    }
    
    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved split summary JSON to: {OUTPUT_SUMMARY_JSON}")
    print("\nStep 3 completed successfully.")

if __name__ == "__main__":
    create_oasis1_splits()
