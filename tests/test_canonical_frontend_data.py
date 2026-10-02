"""
Validation Tests for Canonical Frontend Data Package.
Verifies all 19 JSON files, data integrity, constraints, and contract compliance.
"""

import os
import json
import pytest

CANONICAL_DIR = "results/final_campaign/canonical_frontend_data"

EXPECTED_FILES = [
    "study_metadata.json",
    "models.json",
    "leaderboard.json",
    "data_efficiency.json",
    "label_efficiency.json",
    "ssap.json",
    "parameter_efficiency.json",
    "compute_efficiency.json",
    "calibration.json",
    "domain_shift.json",
    "fusion.json",
    "representation_analysis.json",
    "error_analysis.json",
    "dataset_characteristics.json",
    "statistics.json",
    "interpretability_index.json",
    "methodology.json",
    "limitations.json",
    "audit_status.json",
    "training_diagnostics.json",
    "foundation_adaptation.json",
    "statistical_uncertainty.json",
    "adaptation_comparison.json"
]


def test_canonical_directory_and_files_exist():
    assert os.path.exists(CANONICAL_DIR), f"Directory {CANONICAL_DIR} does not exist"
    for f in EXPECTED_FILES:
        path = os.path.join(CANONICAL_DIR, f)
        assert os.path.exists(path), f"Required file {f} missing from canonical data package"
        assert os.path.getsize(path) > 0, f"File {f} is empty"


def test_canonical_json_validity_and_no_nan_inf():
    for f in EXPECTED_FILES:
        path = os.path.join(CANONICAL_DIR, f)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            assert data is not None, f"{f} loaded as None"
            # Text check for raw NaN or Infinity
            text = json.dumps(data)
            assert "NaN" not in text, f"NaN found in {f}"
            assert "Infinity" not in text, f"Infinity found in {f}"


def test_models_json_schema_and_uniqueness():
    path = os.path.join(CANONICAL_DIR, "models.json")
    with open(path, "r", encoding="utf-8") as fp:
        models = json.load(fp)
    assert len(models) == 6
    ids = [m["id"] for m in models]
    assert len(ids) == len(set(ids)), "Model IDs must be unique"
    for m in models:
        assert "display_name" in m
        assert "total_parameters" in m
        assert "trainable_parameters" in m
        assert "locked_test_auc" in m
        assert 0.0 <= m["locked_test_auc"] <= 1.0


def test_leaderboard_json_consistency():
    path = os.path.join(CANONICAL_DIR, "leaderboard.json")
    with open(path, "r", encoding="utf-8") as fp:
        lb = json.load(fp)
    assert "entries" in lb
    assert len(lb["entries"]) == 6
    ranks = [e["rank"] for e in lb["entries"]]
    assert ranks == list(range(1, 7)), "Leaderboard ranks must be 1 through 6 sequential"


def test_interpretability_index_portable_paths():
    path = os.path.join(CANONICAL_DIR, "interpretability_index.json")
    with open(path, "r", encoding="utf-8") as fp:
        cases = json.load(fp)
    assert len(cases) > 0
    for c in cases:
        assert not c["composite_image"].startswith("C:"), "No absolute Windows drive letters allowed"
        assert not c["composite_image"].startswith("D:"), "No absolute Windows drive letters allowed"
        assert "\\" not in c["composite_image"], "Image paths must use forward slashes"
        assert os.path.exists(c["composite_image"]), f"Image file {c['composite_image']} must exist"


def test_audit_status_is_ready():
    path = os.path.join(CANONICAL_DIR, "audit_status.json")
    with open(path, "r", encoding="utf-8") as fp:
        status = json.load(fp)
    assert status["overall_status"] == "PASS_WITH_LIMITATIONS"
    assert status["frontend_readiness"] == "READY"
    assert status["data_leakage_status"] == "PASS"
