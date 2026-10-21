"""
tests/test_v3_shift_mri_integrity.py
Pytest suite verifying the integrity, schema validity, and empirical consistency
of the SHIFT-MRI V3 benchmark and TrustGate failure-detection framework.
"""

import os
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v3")
V3_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "research", "v3")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "docs", "submission")


def test_v3_prediction_schemas():
    """Verify prediction tables exist, have non-null probabilities bounded in [0, 1]."""
    oasis_pred_path = os.path.join(PROJECT_ROOT, "results", "v2", "tables", "v2_oasis_locked_test_predictions.csv")
    adni_pred_path = os.path.join(PROJECT_ROOT, "results", "v2", "tables", "v2_adni_external_predictions.csv")
    oasis2_pred_path = os.path.join(PROJECT_ROOT, "results", "v2", "tables", "v2_oasis2_external_predictions.csv")

    assert os.path.exists(oasis_pred_path), "OASIS-1 test predictions missing"
    assert os.path.exists(adni_pred_path), "ADNI external predictions missing"
    assert os.path.exists(oasis2_pred_path), "OASIS-2 external predictions missing"

    df_o = pd.read_csv(oasis_pred_path)
    df_a = pd.read_csv(adni_pred_path)
    df_o2 = pd.read_csv(oasis2_pred_path)

    assert len(df_o) == 30, f"Expected 30 OASIS test samples, got {len(df_o)}"
    assert len(df_a) == 500, f"Expected 500 ADNI samples, got {len(df_a)}"
    assert len(df_o2) == 150, f"Expected 150 OASIS-2 samples, got {len(df_o2)}"

    for col in df_o.columns:
        if "prob" in col:
            assert df_o[col].min() >= 0.0 and df_o[col].max() <= 1.0, f"Probability out of range in {col}"
            assert not df_o[col].isnull().any(), f"Nulls in {col}"

    for col in df_a.columns:
        if "prob" in col:
            assert df_a[col].min() >= 0.0 and df_a[col].max() <= 1.0, f"Probability out of range in {col}"
            assert not df_a[col].isnull().any(), f"Nulls in {col}"


def test_v3_cross_cohort_reliability_matrix():
    """Verify Cross-Cohort Reliability Matrix (N01) contents and calibration columns."""
    rel_path = os.path.join(V3_RESULTS_DIR, "reliability", "cross_cohort_reliability_matrix.csv")
    assert os.path.exists(rel_path), "Cross-cohort reliability matrix missing"

    df_rel = pd.read_csv(rel_path)
    assert len(df_rel) == 24, f"Expected 24 rows in matrix (10 OASIS + 10 ADNI + 4 OASIS-2), got {len(df_rel)}"

    required_cols = [
        "Cohort", "Model", "N", "Prevalence", "ROC_AUC", "ROC_AUC_95CI",
        "PR_AUC", "BalAcc_t05", "PosRate_t05", "Collapse_t05",
        "Source_Optimal_t", "BalAcc_t_star", "Brier_Score", "ECE",
        "Calibration_Slope", "Calibration_Intercept", "AUC_Generalization_Gap"
    ]
    for col in required_cols:
        assert col in df_rel.columns, f"Missing column {col} in reliability matrix"

    # Verify that positive rate collapse occurs on ADNI
    adni_rows = df_rel[df_rel["Cohort"].str.contains("ADNI")]
    collapsed_count = adni_rows["Collapse_t05"].sum()
    assert collapsed_count >= 5, f"Expected at least 5 collapsed models on ADNI, got {collapsed_count}"


def test_v3_demographic_baselines():
    """Verify Demographic Baselines (N03) table and verify Age+Sex performance."""
    demo_path = os.path.join(V3_RESULTS_DIR, "demographic_baselines", "demographic_baselines_results.csv")
    assert os.path.exists(demo_path), "Demographic baselines CSV missing"

    df_demo = pd.read_csv(demo_path)
    assert len(df_demo) == 9, f"Expected 9 rows (3 models x 3 cohorts), got {len(df_demo)}"

    adni_age_sex = df_demo[(df_demo["Cohort"].str.contains("ADNI")) & (df_demo["Model"] == "Demographic: Age + Sex")]
    assert len(adni_age_sex) == 1, "ADNI Age + Sex row missing"
    auc_val = adni_age_sex["ROC_AUC"].values[0]
    assert 0.50 <= auc_val <= 0.65, f"Expected ADNI Age+Sex AUC around 0.567, got {auc_val}"


def test_v3_abstention_and_ood():
    """Verify Selective Prediction (N04) and OOD Detection (N07) outputs."""
    abst_path = os.path.join(V3_RESULTS_DIR, "abstention", "abstention_metrics.csv")
    ood_path = os.path.join(V3_RESULTS_DIR, "ood_detection", "ood_detection_evaluation.csv")

    assert os.path.exists(abst_path), "Abstention metrics missing"
    assert os.path.exists(ood_path), "OOD detection evaluation missing"

    df_ood = pd.read_csv(ood_path)
    for _, r in df_ood.iterrows():
        assert r["ood_roc_auc"] >= 0.95, f"Expected OOD AUC >= 0.95, got {r['ood_roc_auc']} for {r['method']}"


def test_v3_trustgate_framework():
    """Verify TrustGate benchmark, ablations, and scanner subgroup outputs."""
    tg_bench_path = os.path.join(V3_RESULTS_DIR, "trustgate", "trustgate_benchmark_results.csv")
    tg_sub_path = os.path.join(V3_RESULTS_DIR, "trustgate", "trustgate_scanner_subgroup_analysis.csv")
    tg_abl_path = os.path.join(V3_RESULTS_DIR, "trustgate_ablation", "trustgate_ablation_results.csv")

    assert os.path.exists(tg_bench_path), "TrustGate benchmark results missing"
    assert os.path.exists(tg_sub_path), "TrustGate scanner subgroup analysis missing"
    assert os.path.exists(tg_abl_path), "TrustGate ablation results missing"

    df_bench = pd.read_csv(tg_bench_path)
    assert "Classification Margin" in df_bench["Method"].values
    assert "Proposed TrustGate Composite" in df_bench["Method"].values

    df_abl = pd.read_csv(tg_abl_path)
    assert len(df_abl) == 5, f"Expected 5 ablation variants, got {len(df_abl)}"
    assert "Ablate C: No Disagreement (A + B + D)" in df_abl["Ablation_Variant"].values


def test_v3_documentation_and_reports_exist():
    """Verify that all core research reports, handoffs, and defense guides are present."""
    required_files = [
        os.path.join(V3_RESULTS_DIR, "SHIFT_MRI_FINAL_RESEARCH_REPORT.md"),
        os.path.join(V3_RESULTS_DIR, "SHIFT_MRI_EXECUTIVE_HANDOFF.md"),
        os.path.join(V3_DOCS_DIR, "v2_scientific_correction_audit.md"),
        os.path.join(V3_DOCS_DIR, "literature_review_and_novelty.md"),
        os.path.join(V3_DOCS_DIR, "new_dataset_discovery_report.md"),
        os.path.join(V3_DOCS_DIR, "dataset_license_and_access_matrix.csv"),
        os.path.join(V3_DOCS_DIR, "research_hypotheses.md"),
        os.path.join(V3_DOCS_DIR, "clinical_label_sensitivity_audit.md"),
        os.path.join(V3_DOCS_DIR, "trustgate_method_definition.md"),
        os.path.join(V3_DOCS_DIR, "trustgate_prior_art_comparison.md"),
        os.path.join(V3_DOCS_DIR, "trustgate_research_findings.md"),
        os.path.join(SUBMISSION_DIR, "SHIFT_MRI_NOVELTY_DEFENSE.md"),
        os.path.join(SUBMISSION_DIR, "SHIFT_MRI_PRESENTATION.md"),
        os.path.join(SUBMISSION_DIR, "FINAL_SUBMISSION_CHECKLIST.md"),
    ]

    for fpath in required_files:
        assert os.path.exists(fpath), f"Missing required document: {fpath}"
        assert os.path.getsize(fpath) > 100, f"File appears unexpectedly empty: {fpath}"
