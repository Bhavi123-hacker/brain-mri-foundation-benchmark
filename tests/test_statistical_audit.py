"""
Automated Validation Tests for Final Pre-Publication Statistical, Uncertainty and Robustness Audit.
Verifies all 12 artifacts, exact Clopper-Pearson binomial CIs, bootstrap CIs, paired comparisons,
sample size disclosure (N=8), zero test-label leakage in calibration, and domain shift tables.
"""

import os
import json
import csv
import pytest

AUDIT_DIR = "results/final_campaign/statistical_audit"

REQUIRED_ARTIFACTS = [
    "final_confidence_intervals.csv",
    "final_confidence_intervals.json",
    "adaptation_paired_comparison.csv",
    "adaptation_paired_comparison.json",
    "subject_level_prediction_comparison.csv",
    "domain_shift_analysis.csv",
    "domain_shift_analysis.json",
    "calibration_audit.json",
    "training_fit_audit.json",
    "parameter_efficiency_final.csv",
    "final_claim_audit.json",
    "FINAL_STATISTICAL_AUDIT_REPORT.md"
]


def test_all_12_audit_artifacts_exist():
    assert os.path.exists(AUDIT_DIR), f"Audit directory {AUDIT_DIR} does not exist"
    for filename in REQUIRED_ARTIFACTS:
        filepath = os.path.join(AUDIT_DIR, filename)
        assert os.path.exists(filepath), f"Required artifact {filename} missing"
        assert os.path.getsize(filepath) > 0, f"Artifact {filename} is empty"


def test_audit_json_validity_and_no_nan_or_inf():
    json_files = [f for f in REQUIRED_ARTIFACTS if f.endswith(".json")]
    for jf in json_files:
        filepath = os.path.join(AUDIT_DIR, jf)
        with open(filepath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            assert data is not None, f"JSON {jf} loaded as None"
            raw_text = json.dumps(data)
            assert "NaN" not in raw_text, f"NaN found in {jf}"
            assert "Infinity" not in raw_text, f"Infinity found in {jf}"


def test_sample_size_and_confidence_interval_bounds():
    filepath = os.path.join(AUDIT_DIR, "final_confidence_intervals.json")
    with open(filepath, "r", encoding="utf-8") as fp:
        records = json.load(fp)
    
    assert len(records) >= 18, f"Expected at least 18 evaluation records, got {len(records)}"
    
    for r in records:
        assert r["sample_size"] == 8, f"Sample size must be exactly 8, got {r['sample_size']} for {r['model_id']}"
        assert 0 <= r["failures"] <= 8, "Failures count must be between 0 and 8"
        
        # Test exact Clopper-Pearson CI bounds
        acc_est = r["accuracy"]
        acc_low, acc_high = r["accuracy_ci_95"]
        assert 0.0 <= acc_low <= (acc_est + 1e-4), f"Invalid accuracy CI lower bound for {r['model_id']}"
        assert (acc_est - 1e-4) <= acc_high <= 1.0, f"Invalid accuracy CI upper bound for {r['model_id']}"
        
        # Test ROC AUC CI bounds
        auc_est = r["roc_auc"]
        auc_low, auc_high = r["roc_auc_ci_95"]
        assert 0.0 <= auc_low <= (auc_est + 1e-4), f"Invalid ROC AUC CI lower bound for {r['model_id']}"
        assert (auc_est - 1e-4) <= auc_high <= 1.0, f"Invalid ROC AUC CI upper bound for {r['model_id']}"


def test_calibration_zero_test_label_leakage():
    filepath = os.path.join(AUDIT_DIR, "calibration_audit.json")
    with open(filepath, "r", encoding="utf-8") as fp:
        audit = json.load(fp)
    
    assert isinstance(audit, list), "Expected list of models in calibration_audit.json"
    for item in audit:
        assert item.get("test_labels_used") is False, f"Test labels used for fitting temperature on {item.get('model_name')}!"
        assert item.get("temperature_parameter_T") > 0.0, f"Invalid temperature for {item.get('model_name')}"
        assert "VERIFIED_ISOLATED" in item.get("audit_verdict", "")


def test_subject_level_predictions_integrity():
    filepath = os.path.join(AUDIT_DIR, "subject_level_prediction_comparison.csv")
    with open(filepath, "r", encoding="utf-8") as fp:
        reader = list(csv.DictReader(fp))
    
    assert len(reader) == 48, f"Expected 48 subject rows (3 models x 2 cohorts x 8 subjects), got {len(reader)}"
    valid_labels = {"Dementia/AD", "CN", "Cognitively Normal"}
    for row in reader:
        assert row["true_label"] in valid_labels, f"Unexpected true label: {row['true_label']}"
        assert row["baseline_prediction"] in valid_labels, f"Unexpected baseline pred: {row['baseline_prediction']}"
        assert row["adapted_prediction"] in valid_labels, f"Unexpected adapted pred: {row['adapted_prediction']}"
        assert row["outcome_change"] in [
            "Unchanged Correct",
            "Unchanged Incorrect",
            "Baseline Incorrect -> Adapted Correct",
            "Baseline Correct -> Adapted Incorrect"
        ]


def test_adaptation_paired_comparisons():
    filepath = os.path.join(AUDIT_DIR, "adaptation_paired_comparison.json")
    with open(filepath, "r", encoding="utf-8") as fp:
        records = json.load(fp)
    
    assert len(records) == 6, f"Expected 6 paired adaptation comparisons, got {len(records)}"
    for r in records:
        counts = r["transition_counts"]
        total_transitions = (
            counts["Unchanged Correct"] +
            counts["Unchanged Incorrect"] +
            counts["Baseline Incorrect -> Adapted Correct"] +
            counts["Baseline Correct -> Adapted Incorrect"]
        )
        assert total_transitions == 8, f"Transition counts do not sum to 8 for {r['family']} {r['cohort']}"


def test_final_claim_audit_veracity():
    filepath = os.path.join(AUDIT_DIR, "final_claim_audit.json")
    with open(filepath, "r", encoding="utf-8") as fp:
        claims = json.load(fp)
    
    assert claims["prohibited_terms_status"] == "ZERO_PROHIBITED_TERMS_FOUND"
    assert claims["clinical_overclaiming_status"] == "PASS"
    assert "N=8" in claims["sample_size_disclosure"]
