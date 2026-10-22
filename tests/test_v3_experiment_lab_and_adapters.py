"""
tests/test_v3_experiment_lab_and_adapters.py
Rigorous verification for:
1. Experiment Lab execution engine and SQLite persistence.
2. Reusable Dataset and Model Adapters contracts.
3. Model-specific validation-locked threshold dictionary.
4. Health endpoint reporting canonical cohort counts (OASIS-1 N=198, ADNI N=500).
5. Server-side scan pagination with multi-attribute filtering.
6. Experiment comparison with cross-cohort diagnostic label warnings.
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from experiments.experiment_lab import (
    init_experiment_db,
    get_all_experiment_runs,
    execute_evaluation_experiment,
    compare_experiment_runs,
    compute_metrics_and_ci,
    DB_PATH
)

from adapters import (
    OASIS1Adapter,
    ADNIAdapter,
    OASIS2Adapter,
    FutureDatasetAdapterTemplate,
    ResNet10Adapter,
    DenseNet121Adapter,
    MultiView25DAdapter,
    DualBranchHybridAdapter
)


def test_experiment_lab_db_and_execution():
    """Verify SQLite database initialization and authentic evaluation execution."""
    init_experiment_db()
    assert DB_PATH.exists(), f"SQLite database missing at {DB_PATH}"

    runs = get_all_experiment_runs()
    assert len(runs) >= 5, f"Expected canonical historical runs in SQLite, got {len(runs)}"

    # Execute a live evaluation experiment on ADNI
    res = execute_evaluation_experiment(
        model_name="MONAI Swin UNETR SSL (Linear Probe)",
        dataset="ADNI",
        seed=42,
        custom_name="PyTest Swin UNETR External Evaluation"
    )

    assert res["status"] == "COMPLETED"
    assert "roc_auc" in res["metrics"]
    assert res["metrics"]["roc_auc"] == 0.5834
    assert abs(res["metrics"]["ci_95"][0] - 0.5331) < 0.01
    assert abs(res["metrics"]["ci_95"][1] - 0.6337) < 0.01
    assert res["metrics"]["sample_size"] == 500
    assert res["metrics"]["positives"] == 197
    assert res["metrics"]["negatives"] == 303


def test_experiment_comparator_and_cross_cohort_warning():
    """Verify multi-run comparison and cross-cohort diagnostic warning."""
    # Compare ADNI run against an OASIS-1 run
    res_oas = execute_evaluation_experiment(
        model_name="3D ResNet-10 (Pipeline B)",
        dataset="OASIS-1",
        seed=42,
        custom_name="PyTest OASIS-1 In-Domain Evaluation"
    )

    runs = get_all_experiment_runs()
    adni_run_id = [r["run_id"] for r in runs if r["dataset"] == "ADNI"][0]
    oasis_run_id = res_oas["run_id"]

    comp = compare_experiment_runs([adni_run_id, oasis_run_id])
    assert comp["status"] == "success"
    assert comp["runs_compared"] == 2
    assert comp["cross_cohort_warning"] is not None
    assert "Cross-cohort comparison detected" in comp["cross_cohort_warning"]
    assert "CDR" in comp["cross_cohort_warning"]


def test_dataset_adapters():
    """Verify Dataset Adapters metadata, clinical labels, and sample counts."""
    oas = OASIS1Adapter()
    meta_oas = oas.get_metadata()
    assert meta_oas["dataset_name"] == "OASIS-1"
    assert meta_oas["total_participants"] == 198
    assert meta_oas["splits"]["train"] == 138
    assert meta_oas["splits"]["val"] == 30
    assert meta_oas["splits"]["test"] == 30
    assert oas.check_data_availability() is True

    adni = ADNIAdapter()
    meta_adni = adni.get_metadata()
    assert meta_adni["dataset_name"] == "ADNI"
    assert meta_adni["total_participants"] == 500
    assert meta_adni["classes"] == {"CN": 303, "AD": 197}
    assert meta_adni["prevalence"] == 0.394
    assert adni.check_data_availability() is True

    future = FutureDatasetAdapterTemplate("AIBL")
    assert future.get_metadata()["is_authorized"] is False


def test_model_adapters():
    """Verify Model Adapters specification, parameter tracking, and checkpoint verification."""
    m_r10 = ResNet10Adapter()
    specs = m_r10.get_model_specs()
    assert specs["model_name"] == "3D ResNet-10 (Pipeline B)"
    assert specs["val_threshold"] == 0.6860
    assert specs["checkpoint_exists"] is True
    assert specs["checkpoint_hash"] is not None

    m_dense = DenseNet121Adapter()
    assert m_dense.get_model_specs()["val_threshold"] == 0.4651

    m_mv = MultiView25DAdapter()
    assert m_mv.get_model_specs()["val_threshold"] == 0.9741

    m_hyb = DualBranchHybridAdapter()
    assert m_hyb.get_model_specs()["val_threshold"] == 0.4763


def test_canonical_thresholds_in_registry():
    """Verify that canonical_registry.json contains distinct, validated thresholds for each model."""
    reg_path = PROJECT_ROOT / "results" / "final" / "canonical_registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    thresholds = reg["validation_derived_thresholds"]
    assert len(thresholds) == 10
    assert thresholds["MONAI Swin UNETR SSL (Linear Probe)"] == 0.5088
    assert thresholds["3D ResNet-10 (Pipeline B)"] == 0.6860
    assert thresholds["3D DenseNet-121 (Pipeline B)"] == 0.4651
    assert thresholds["2.5D Multi-View ResNet-18 (Pipeline B)"] == 0.9741
    assert thresholds["Dual-Branch Hybrid (ResNet-10 + Radiomics)"] == 0.4763
    assert thresholds["FOMO25 AMAES ResEnc-B (Linear Probe)"] == 0.5457


def test_health_endpoint_canonical_counts():
    """Verify that the backend server reports exact canonical cohort numbers."""
    from backend.server import SCAN_REPO

    # Check scan repository total
    assert len(SCAN_REPO.scans) == 848

    oasis_scans = [s for s in SCAN_REPO.scans if s["cohort"] == "OASIS-1"]
    adni_scans = [s for s in SCAN_REPO.scans if s["cohort"] == "ADNI"]
    oasis2_scans = [s for s in SCAN_REPO.scans if s["cohort"] == "OASIS-2"]

    assert len(oasis_scans) == 198
    assert len(adni_scans) == 500
    assert len(oasis2_scans) == 150
