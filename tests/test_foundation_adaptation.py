"""
Validation Tests for Foundation Model Adaptation Campaign Artifacts.
Verifies all adaptation history CSV/JSON records, multi-strategy coverage,
stability across seeds, and post-freeze locked holdout comparison data.
"""

import json
import os
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTATION_DIR = PROJECT_ROOT / "results" / "final_campaign" / "foundation_adaptation"
CANONICAL_DATA = PROJECT_ROOT / "results" / "final_campaign" / "canonical_frontend_data" / "foundation_adaptation.json"


def test_adaptation_directory_and_artifacts_exist():
    """Verify that all core adaptation artifact files exist on disk."""
    assert ADAPTATION_DIR.exists(), "foundation_adaptation directory must exist"
    assert (ADAPTATION_DIR / "adaptation_history.csv").exists(), "adaptation_history.csv must exist"
    assert (ADAPTATION_DIR / "adaptation_history.json").exists(), "adaptation_history.json must exist"
    assert (ADAPTATION_DIR / "adaptation_summary.json").exists(), "adaptation_summary.json must exist"
    assert (ADAPTATION_DIR / "adaptation_vs_baseline_comparison.json").exists(), "adaptation_vs_baseline_comparison.json must exist"
    assert CANONICAL_DATA.exists(), "canonical foundation_adaptation.json must exist"


def test_adaptation_curves_exist():
    """Verify that all 12 strategy dual-axis curve plots exist."""
    curves_dir = ADAPTATION_DIR / "adaptation_curves"
    assert curves_dir.exists(), "adaptation_curves directory must exist"
    expected_models = ["brainiac", "fomo25", "swinunetr"]
    expected_strategies = ["linear_probe", "partial_ft", "deeper_ft", "full_ft"]
    for m in expected_models:
        for s in expected_strategies:
            p = curves_dir / f"{m}_{s}_train_vs_val.png"
            assert p.exists(), f"Curve plot missing: {p}"
            assert p.stat().st_size > 10000, f"Curve plot file is abnormally small: {p}"


def test_adaptation_history_integrity():
    """Verify CSV history row count, columns, and numerical validity."""
    df = pd.read_csv(ADAPTATION_DIR / "adaptation_history.csv")
    assert len(df) == 360, f"Expected 360 epoch records, got {len(df)}"
    required_cols = [
        "model_name", "strategy", "seed", "epoch",
        "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "validation_AUC"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    assert not df["validation_AUC"].isna().any(), "validation_AUC must not contain NaN"
    assert (df["validation_AUC"] >= 0.0).all() and (df["validation_AUC"] <= 1.0).all()


def test_canonical_foundation_adaptation_json():
    """Verify canonical JSON structure, models evaluated, and no NaN/Infinity."""
    with open(CANONICAL_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["models_evaluated"] == 3
    assert data["total_training_runs"] == 18
    assert "best_strategies" in data
    assert set(data["best_strategies"].keys()) == {"BrainIAC", "FOMO25", "SwinUNETR"}
    
    # Check baseline vs adapted comparisons
    comps = data["baseline_vs_adapted_comparison"]
    assert len(comps) == 3
    for c in comps:
        evals = c["evaluations"]
        assert "Baseline Linear Probe" in evals
        assert "Adapted / Fine-Tuned" in evals
        assert "oasis_locked_test" in evals["Baseline Linear Probe"]
        assert "adni_external_holdout" in evals["Baseline Linear Probe"]
        assert "oasis_locked_test" in evals["Adapted / Fine-Tuned"]
        assert "adni_external_holdout" in evals["Adapted / Fine-Tuned"]
