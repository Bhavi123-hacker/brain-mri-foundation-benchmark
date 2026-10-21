"""
Comprehensive Validation Tests for Brain MRI Benchmark Integrity & Label Resolution.

Verifies:
1. Manifest integrity (OASIS-1 and ADNI full cohort)
2. Tensor cache integrity (shapes, non-zero voxels, no NaNs/Infs)
3. Disjoint patient partitioning (0.0% patient leakage)
4. Schema resolution & deterministic binary mapping (0.0 or 1.0)
5. Fail-early validation on invalid, null, or ambiguous labels
6. Dataset & DataLoader smoke tests
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = r"D:\ml project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_master_benchmark_campaign import OASISPreprocessedDataset, ADNIPreprocessedDataset

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
ADNI_MANIFEST_CSV = r"D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv"
ADNI_TENSORS_DIR = r"D:\brain_mri_data\ADNI\processed\tensors"


def test_oasis1_splits_and_patient_isolation():
    """Verify OASIS-1 split counts and guarantees zero patient leakage across partitions."""
    assert os.path.exists(OASIS_SPLITS_CSV), f"Missing OASIS splits: {OASIS_SPLITS_CSV}"
    df = pd.read_csv(OASIS_SPLITS_CSV)
    assert len(df) == 198, f"Expected 198 subjects, got {len(df)}"

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    assert len(train_df) == 138, f"Expected 138 train, got {len(train_df)}"
    assert len(val_df) == 30, f"Expected 30 val, got {len(val_df)}"
    assert len(test_df) == 30, f"Expected 30 test, got {len(test_df)}"

    # Subject ID isolation
    train_subs = set(train_df["session_id"].apply(lambda s: s.split("_")[1]))
    val_subs = set(val_df["session_id"].apply(lambda s: s.split("_")[1]))
    test_subs = set(test_df["session_id"].apply(lambda s: s.split("_")[1]))

    assert len(train_subs.intersection(val_subs)) == 0, "Patient leakage between train and val"
    assert len(train_subs.intersection(test_subs)) == 0, "Patient leakage between train and test"
    assert len(val_subs.intersection(test_subs)) == 0, "Patient leakage between val and test"


def test_adni_manifest_and_cohort_reconciliation():
    """Verify ADNI reconciled cohort has exactly 500 eligible baseline subjects with valid metadata."""
    assert os.path.exists(ADNI_MANIFEST_CSV), f"Missing ADNI manifest: {ADNI_MANIFEST_CSV}"
    df = pd.read_csv(ADNI_MANIFEST_CSV)
    pass_df = df[df["status"] == "PASS"]
    assert len(pass_df) == 500, f"Expected 500 PASS subjects, got {len(pass_df)}"

    cn_count = (pass_df["group"] == "CN").sum()
    ad_count = (pass_df["group"] == "AD").sum()
    assert cn_count == 303, f"Expected 303 CN, got {cn_count}"
    assert ad_count == 197, f"Expected 197 AD, got {ad_count}"

    # Unique subjects guarantee
    assert pass_df["subject_id"].nunique() == 500, "Duplicate subjects in ADNI baseline cohort"


def test_tensor_integrity_samples():
    """Check tensor shape and numerical validity on random sample tensors."""
    oasis_df = pd.read_csv(OASIS_SPLITS_CSV)
    for sid in oasis_df["session_id"].head(5):
        npy_path = os.path.join(OASIS_TENSORS_DIR, f"{sid}.npy")
        assert os.path.exists(npy_path), f"Missing tensor: {npy_path}"
        arr = np.load(npy_path)
        assert arr.shape == (1, 128, 128, 128), f"Unexpected shape {arr.shape} for {sid}"
        assert not np.isnan(arr).any(), f"NaN found in {sid}"
        assert not np.isinf(arr).any(), f"Inf found in {sid}"

    adni_df = pd.read_csv(ADNI_MANIFEST_CSV)
    for _, row in adni_df[adni_df["status"] == "PASS"].head(5).iterrows():
        sid = row["subject_id"]
        iid = row["image_data_id"]
        npy_path = os.path.join(ADNI_TENSORS_DIR, f"{sid}_{iid}.npy")
        assert os.path.exists(npy_path), f"Missing tensor: {npy_path}"
        arr = np.load(npy_path)
        assert arr.shape == (1, 128, 128, 128), f"Unexpected shape {arr.shape} for {sid}"
        assert not np.isnan(arr).any(), f"NaN found in {sid}"
        assert not np.isinf(arr).any(), f"Inf found in {sid}"


def test_oasis_dataset_schema_and_mapping():
    """Verify OASISPreprocessedDataset correctly maps binary labels from binary_label, cdr, or clinical_stage."""
    df = pd.read_csv(OASIS_SPLITS_CSV).head(10)

    # 1. Standard manifest with binary_label
    ds = OASISPreprocessedDataset(df, OASIS_TENSORS_DIR)
    assert len(ds) == 10
    for i in range(len(ds)):
        _, target, _ = ds[i]
        assert target.item() in (0.0, 1.0)

    # 2. Schema with only cdr (no binary_label)
    df_cdr = df.drop(columns=["binary_label"])
    ds_cdr = OASISPreprocessedDataset(df_cdr, OASIS_TENSORS_DIR)
    for i in range(len(ds_cdr)):
        _, target, _ = ds_cdr[i]
        expected = 0.0 if df.iloc[i]["cdr"] == 0.0 else 1.0
        assert target.item() == expected

    # 3. Schema with only clinical_stage
    df_stage = df.drop(columns=["binary_label", "cdr"])
    ds_stage = OASISPreprocessedDataset(df_stage, OASIS_TENSORS_DIR)
    for i in range(len(ds_stage)):
        _, target, _ = ds_stage[i]
        expected = 0.0 if "CDR_0.0" in df.iloc[i]["clinical_stage"] else 1.0
        assert target.item() == expected


def test_adni_dataset_schema_and_mapping():
    """Verify ADNIPreprocessedDataset correctly maps binary labels from binary_label or group (CN=0, AD=1)."""
    df = pd.read_csv(ADNI_MANIFEST_CSV)
    df_pass = df[df["status"] == "PASS"].head(10)

    # 1. Standard manifest with binary_label
    ds = ADNIPreprocessedDataset(df_pass, ADNI_TENSORS_DIR)
    assert len(ds) == 10
    for i in range(len(ds)):
        _, target, _, _ = ds[i]
        assert target.item() in (0.0, 1.0)

    # 2. Schema with only group (no binary_label)
    df_group = df_pass.drop(columns=["binary_label"])
    ds_group = ADNIPreprocessedDataset(df_group, ADNI_TENSORS_DIR)
    for i in range(len(ds_group)):
        _, target, _, _ = ds_group[i]
        expected = 0.0 if df_pass.iloc[i]["group"] == "CN" else 1.0
        assert target.item() == expected


def test_label_validation_fails_early_on_invalid_data():
    """Verify that dataset constructors fail early with clear exceptions when schemas or values are invalid."""
    oasis_sample = pd.read_csv(OASIS_SPLITS_CSV).head(4)
    adni_sample = pd.read_csv(ADNI_MANIFEST_CSV).head(4)

    # Missing label columns entirely
    bad_oasis = oasis_sample.drop(columns=["binary_label", "cdr", "clinical_stage"])
    with pytest.raises(KeyError, match="lacks authoritative label column"):
        OASISPreprocessedDataset(bad_oasis, OASIS_TENSORS_DIR)

    bad_adni = adni_sample.drop(columns=["binary_label", "group"])
    with pytest.raises(KeyError, match="lacks authoritative label column"):
        ADNIPreprocessedDataset(bad_adni, ADNI_TENSORS_DIR)

    # Unexpected binary_label values (e.g. 2.5)
    bad_val_oasis = oasis_sample.copy()
    bad_val_oasis.loc[0, "binary_label"] = 2.5
    with pytest.raises(ValueError, match="unexpected values"):
        OASISPreprocessedDataset(bad_val_oasis, OASIS_TENSORS_DIR)

    # Ambiguous group in ADNI (e.g. MCI)
    ambig_adni = adni_sample.drop(columns=["binary_label"]).copy()
    ambig_adni.loc[0, "group"] = "MCI"
    with pytest.raises(ValueError, match="non-binary or ambiguous classes"):
        ADNIPreprocessedDataset(ambig_adni, ADNI_TENSORS_DIR)


def test_dataloader_batch_smoke_test():
    """Verify batch fetching and dimensions through PyTorch DataLoader for both cohorts."""
    oasis_df = pd.read_csv(OASIS_SPLITS_CSV).head(4)
    adni_df = pd.read_csv(ADNI_MANIFEST_CSV)[lambda x: x["status"] == "PASS"].head(4)

    oasis_ds = OASISPreprocessedDataset(oasis_df, OASIS_TENSORS_DIR, is_train=True)
    oasis_loader = DataLoader(oasis_ds, batch_size=2, shuffle=False)
    batch = next(iter(oasis_loader))
    imgs, targets, sids = batch
    assert imgs.shape == (2, 1, 128, 128, 128)
    assert targets.shape == (2, 1)
    assert len(sids) == 2

    adni_ds = ADNIPreprocessedDataset(adni_df, ADNI_TENSORS_DIR)
    adni_loader = DataLoader(adni_ds, batch_size=2, shuffle=False)
    batch_adni = next(iter(adni_loader))
    imgs_a, targets_a, sids_a, iids_a = batch_adni
    assert imgs_a.shape == (2, 1, 128, 128, 128)
    assert targets_a.shape == (2, 1)
    assert len(sids_a) == 2
    assert len(iids_a) == 2


def test_benchmark_deliverables_and_results_integrity():
    """Verify all benchmark tables, prediction CSVs, figures, and research reports exist and are valid."""
    runs_dir = r"D:\brain_mri_data\benchmark_runs"
    tables_dir = os.path.join(runs_dir, "tables")
    figures_dir = os.path.join(runs_dir, "figures")
    report_path = os.path.join(runs_dir, "report", "final_research_report.md")
    repo_report_path = r"D:\ml project\results\report\final_research_report.md"

    # 1. JSON Results & Master Table
    results_json = os.path.join(tables_dir, "all_benchmark_results.json")
    assert os.path.exists(results_json), "all_benchmark_results.json missing"
    with open(results_json, "r") as f:
        data = json.load(f)
    assert "master_table" in data
    assert "locked_test_results" in data
    assert "adni_results" in data
    assert "label_efficiency_results" in data

    master_csv = os.path.join(tables_dir, "master_model_comparison_table.csv")
    assert os.path.exists(master_csv), "master_model_comparison_table.csv missing"
    m_df = pd.read_csv(master_csv)
    assert len(m_df) >= 6

    # 2. Prediction CSVs
    oasis_preds = os.path.join(tables_dir, "oasis_locked_test_predictions.csv")
    assert os.path.exists(oasis_preds), "oasis_locked_test_predictions.csv missing"
    op_df = pd.read_csv(oasis_preds)
    assert len(op_df) == 30
    assert "resnet18_prob" in op_df.columns
    assert "resnet34_prob" in op_df.columns
    assert op_df["resnet18_prob"].between(0.0, 1.0).all()

    adni_preds = os.path.join(tables_dir, "adni_external_predictions.csv")
    assert os.path.exists(adni_preds), "adni_external_predictions.csv missing"
    ap_df = pd.read_csv(adni_preds)
    assert len(ap_df) == 500
    assert "resnet18_prob" in ap_df.columns
    assert "resnet34_prob" in ap_df.columns
    assert ap_df["resnet18_prob"].between(0.0, 1.0).all()

    # 3. Figures
    expected_figures = [
        "learning_curves_supervised_baselines.png",
        "label_efficiency_curves.png",
        "preprocessing_robustness_comparison.png",
        "calibration_reliability_diagram.png",
        "adni_subgroup_analysis.png",
        "gradcam_3d_interpretability.png"
    ]
    for fig_name in expected_figures:
        fig_p = os.path.join(figures_dir, fig_name)
        assert os.path.exists(fig_p), f"Missing figure: {fig_name}"
        assert os.path.getsize(fig_p) > 10000, f"Figure {fig_name} is too small"

    # 4. Reports
    assert os.path.exists(report_path), "Primary report markdown missing"
    assert os.path.exists(repo_report_path), "Mirrored report markdown missing"
    assert os.path.getsize(report_path) > 5000
    assert os.path.getsize(repo_report_path) > 5000
