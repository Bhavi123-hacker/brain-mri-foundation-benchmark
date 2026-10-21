"""
Test Suite for Version 2 Benchmark Campaign Integrity.
Verifies multi-cohort prediction schemas, 10-model master comparison table,
paired bootstrap audits, and newly ingested OASIS-2 baseline dataset.
"""

import os
import pytest
import pandas as pd
import numpy as np
import torch

WORKSPACE_ROOT = r"D:\ml project"
V2_TABLES_DIR = os.path.join(WORKSPACE_ROOT, "results", "v2", "tables")
V2_CKPTS_DIR = os.path.join(WORKSPACE_ROOT, "results", "v2", "checkpoints")
OASIS2_MANIFEST = r"D:\brain_mri_data\OASIS2\manifests\oasis2_baseline_manifest.csv"


def test_v2_prediction_tables_exist_and_conform():
    """Verify prediction tables exist, have exact row counts, and valid probability ranges."""
    oasis_csv = os.path.join(V2_TABLES_DIR, "v2_oasis_locked_test_predictions.csv")
    adni_csv = os.path.join(V2_TABLES_DIR, "v2_adni_external_predictions.csv")
    oasis2_csv = os.path.join(V2_TABLES_DIR, "v2_oasis2_external_predictions.csv")

    assert os.path.exists(oasis_csv), f"Missing {oasis_csv}"
    assert os.path.exists(adni_csv), f"Missing {adni_csv}"
    assert os.path.exists(oasis2_csv), f"Missing {oasis2_csv}"

    df_o = pd.read_csv(oasis_csv)
    df_a = pd.read_csv(adni_csv)
    df_o2 = pd.read_csv(oasis2_csv)

    assert len(df_o) == 30, f"Expected 30 OASIS test samples, got {len(df_o)}"
    assert len(df_a) == 500, f"Expected 500 ADNI external samples, got {len(df_a)}"
    assert len(df_o2) == 150, f"Expected 150 OASIS-2 external samples, got {len(df_o2)}"

    for df, name in [(df_o, "OASIS-1"), (df_a, "ADNI"), (df_o2, "OASIS-2")]:
        prob_cols = [c for c in df.columns if c.endswith("_prob")]
        assert len(prob_cols) >= 4, f"{name} table missing required probability columns"
        for col in prob_cols:
            vals = df[col].values
            assert np.all(vals >= 0.0) and np.all(vals <= 1.0), f"Out-of-bounds probs in {col}"
            assert not np.any(np.isnan(vals)), f"NaN values found in {col}"


def test_v2_master_model_comparison_table():
    """Verify master model comparison table consolidates all 10 models."""
    master_csv = os.path.join(V2_TABLES_DIR, "v2_master_model_comparison_table.csv")
    assert os.path.exists(master_csv), f"Missing {master_csv}"

    df = pd.read_csv(master_csv)
    assert len(df) == 10, f"Expected exactly 10 evaluated models, found {len(df)}"
    assert set(df["Status"]) == {"COMPLETED"}, "All models in master table must be COMPLETED"

    # Verify key columns exist
    req_cols = [
        "Model", "Paradigm", "Weights_Source", "Total_Params", "Trainable_Params",
        "OASIS_Val_AUC", "OASIS_Test_AUC", "ADNI_Ext_AUC", "OASIS2_Ext_AUC"
    ]
    for col in req_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify V2 models have non-null OASIS2_Ext_AUC
    v2_rows = df[df["OASIS2_Ext_AUC"].notna()]
    assert len(v2_rows) == 4, f"Expected 4 V2 models with OASIS-2 metrics, found {len(v2_rows)}"


def test_v2_statistical_audit_table():
    """Verify 2000-iteration paired bootstrap statistical audit with Holm-Bonferroni."""
    audit_csv = os.path.join(V2_TABLES_DIR, "v2_paired_bootstrap_statistical_audit.csv")
    assert os.path.exists(audit_csv), f"Missing {audit_csv}"

    df = pd.read_csv(audit_csv)
    assert len(df) == 7, f"Expected 7 pairwise hypothesis tests, got {len(df)}"

    req_cols = [
        "model_a", "model_b", "hypothesis_description",
        "oasis_delta_auc", "oasis_delta_ci_95", "oasis_raw_p", "oasis_holm_p",
        "adni_delta_auc", "adni_delta_ci_95", "adni_raw_p", "adni_holm_p"
    ]
    for col in req_cols:
        assert col in df.columns, f"Missing column in audit table: {col}"

    # Verify Holm p >= raw p
    assert np.all(df["oasis_holm_p"] >= df["oasis_raw_p"] - 1e-6)
    assert np.all(df["adni_holm_p"] >= df["adni_raw_p"] - 1e-6)


def test_oasis2_dataset_expansion_integrity():
    """Verify OASIS-2 baseline dataset is complete and isolated."""
    assert os.path.exists(OASIS2_MANIFEST), f"Missing {OASIS2_MANIFEST}"
    df = pd.read_csv(OASIS2_MANIFEST)

    assert len(df) == 150, f"Expected 150 OASIS-2 baseline participants, got {len(df)}"
    assert set(df["binary_label"].unique()) == {0, 1}, "Invalid binary labels"

    # Class balance check
    n_cn = (df["binary_label"] == 0).sum()
    n_ad = (df["binary_label"] == 1).sum()
    assert n_cn == 85, f"Expected 85 CN participants, got {n_cn}"
    assert n_ad == 65, f"Expected 65 AD participants, got {n_ad}"

    # Check that processed tensors exist on disk
    tensors_dir = r"D:\brain_mri_data\OASIS2\processed\tensors"
    assert os.path.exists(tensors_dir)
    assert len(os.listdir(tensors_dir)) == 150, "Expected 150 saved numpy tensors"


def test_v2_checkpoints_exist_and_load():
    """Verify all 4 V2 model checkpoints exist and load into architecture definitions."""
    from models.v2_models import ResNet10_3D, DenseNet121_3D, MultiView25D_ResNet18, DualBranchHybridModel

    ckpts = {
        "3d_resnet_10_seed42_best.pt": ResNet10_3D(output_dim=1),
        "3d_densenet_121_seed42_best.pt": DenseNet121_3D(output_dim=1),
        "2.5d_multi_view_resnet_18_seed42_best.pt": MultiView25D_ResNet18(pretrained=False, output_dim=1),
        "dual_branch_hybrid_seed42_best.pt": DualBranchHybridModel(spatial_backbone="resnet10", num_radiomics=6, output_dim=1),
    }

    for fname, model in ckpts.items():
        path = os.path.join(V2_CKPTS_DIR, fname)
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
