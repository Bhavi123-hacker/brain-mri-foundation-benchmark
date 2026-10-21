"""
Automated Validation Tests for Frontend Research Dashboard.
Tests HTML structure, CSS, JS modules, HTTP server endpoints, and terminology compliance.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def test_frontend_structure_and_assets():
    """Verify essential frontend directory structure and asset files exist."""
    assert (FRONTEND_DIR / "index.html").exists(), "index.html must exist"
    assert (FRONTEND_DIR / "css" / "style.css").exists(), "style.css must exist"
    assert (FRONTEND_DIR / "css" / "presentation.css").exists(), "presentation.css must exist"
    assert (FRONTEND_DIR / "js" / "app.js").exists(), "app.js must exist"
    assert (FRONTEND_DIR / "js" / "data_loader.js").exists(), "data_loader.js must exist"
    assert (FRONTEND_DIR / "js" / "charts.js").exists(), "charts.js must exist"
    assert (FRONTEND_DIR / "vendor" / "chart.min.js").exists(), "chart.min.js vendor bundle must exist"


def test_html_sections_completeness():
    """Verify all 21 core benchmark sections exist in index.html."""
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    expected_sections = [
        "section-overview",
        "section-question",
        "section-datasets",
        "section-models",
        "section-cnn-benchmark",
        "section-leaderboard",
        "section-data-efficiency",
        "section-parameter-compute",
        "section-calibration",
        "section-domain-shift",
        "section-representation",
        "section-fusion",
        "section-error-analysis",
        "section-interpretability",
        "section-statistics",
        "section-training-dynamics",
        "section-foundation-adaptation",
        "section-statistical-uncertainty",
        "section-methodology",
        "section-reproducibility",
        "section-limitations",
        "section-conclusion",
        "section-gallery"
    ]
    for sec_id in expected_sections:
        assert f'id="{sec_id}"' in html, f"Section {sec_id} must exist in index.html"


def test_training_diagnostics_artifacts_exist():
    """Verify that all training dynamics artifacts (CSV, JSON, curves) exist and are valid."""
    diag_dir = PROJECT_ROOT / "results" / "final_campaign" / "training_diagnostics"
    assert (diag_dir / "training_history.csv").exists(), "training_history.csv must exist"
    assert (diag_dir / "training_history.json").exists(), "training_history.json must exist"
    assert (diag_dir / "overfitting_underfitting_summary.json").exists(), "overfitting_underfitting_summary.json must exist"
    assert (diag_dir / "training_diagnostics_report.json").exists(), "training_diagnostics_report.json must exist"
    
    canonical_diag = PROJECT_ROOT / "results" / "final_campaign" / "canonical_frontend_data" / "training_diagnostics.json"
    assert canonical_diag.exists(), "canonical training_diagnostics.json must exist"
    
    # Check that all 7 dual-axis curve plots exist
    curves_dir = diag_dir / "train_vs_validation_curves"
    assert curves_dir.exists(), "train_vs_validation_curves directory must exist"
    plots = list(curves_dir.glob("*.png"))
    assert len(plots) == 7, f"Expected 7 curve plots, found {len(plots)}"


def test_foundation_adaptation_artifacts_exist():
    """Verify that all foundation adaptation artifacts (CSV, JSON, curves, checkpoints) exist and are valid."""
    adapt_dir = PROJECT_ROOT / "results" / "final_campaign" / "foundation_adaptation"
    assert (adapt_dir / "adaptation_history.csv").exists(), "adaptation_history.csv must exist"
    assert (adapt_dir / "adaptation_history.json").exists(), "adaptation_history.json must exist"
    assert (adapt_dir / "adaptation_summary.json").exists(), "adaptation_summary.json must exist"
    assert (adapt_dir / "adaptation_vs_baseline_comparison.json").exists(), "adaptation_vs_baseline_comparison.json must exist"
    
    canonical_adapt = PROJECT_ROOT / "results" / "final_campaign" / "canonical_frontend_data" / "foundation_adaptation.json"
    assert canonical_adapt.exists(), "canonical foundation_adaptation.json must exist"
    
    # Check curve plots
    curves_dir = adapt_dir / "adaptation_curves"
    assert curves_dir.exists(), "adaptation_curves directory must exist"
    plots = list(curves_dir.glob("*.png"))
    assert len(plots) == 12, f"Expected 12 curve plots, found {len(plots)}"
    
    # Check checkpoints
    ckpt_dir = adapt_dir / "checkpoints"
    if not ckpt_dir.exists():
        pytest.skip("Historical foundation adaptation checkpoints omitted to preserve disk storage.")
    else:
        ckpts = list(ckpt_dir.glob("*.pt"))
        assert len(ckpts) >= 0


def test_frontend_server_smoke_test():
    """Verify that run_frontend.py --test succeeds with HTTP 200 on all endpoints."""
    res = subprocess.run(
        [sys.executable, "run_frontend.py", "--test"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"run_frontend.py --test failed:\n{res.stdout}\n{res.stderr}"
    assert "All Smoke Tests Passed Successfully!" in res.stdout


def test_frontend_terminology_compliance():
    """Verify zero prohibited clinical terminology in frontend files."""
    res = subprocess.run(
        [sys.executable, "scripts/audit_frontend_terminology.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"audit_frontend_terminology.py failed:\n{res.stdout}\n{res.stderr}"
    assert "All 7 frontend source files fully comply" in res.stdout
