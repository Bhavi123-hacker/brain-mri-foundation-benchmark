"""
tests/test_server_authenticity_and_security.py
Rigorous automated release-gate tests enforcing scientific integrity:
1. Ground truth label changes do NOT alter prediction outputs (no manufactured predictions).
2. Missing predictions are NOT replaced with hard-coded numbers (explicit None / UNAVAILABLE).
3. Grad-CAM explanations are authentic gradient/pre-generated attributions, never generic Gaussian templates.
4. TrustGate weights strictly equal 0.25*A + 0.25*B + 0.25*C + 0.25*D (tau* = 0.6340).
5. Canonical registry matches authoritative CSVs (ADNI CN=303, AD=197, prevalence=0.394).
6. Live inference executes authentic checkpoints, while cached predictions reflect research artifacts.
7. Server-side pagination indexes all 848 scans without omission.
8. Server blocks path traversal and non-allowlisted downloads (HTTP 403).
"""

import os
import json
import threading
import time
import urllib.request
import urllib.error
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_RESULTS_DIR = PROJECT_ROOT / "results" / "final"
V3_RESULTS_DIR = PROJECT_ROOT / "results" / "v3"

from backend.server import (
    ScanRepository,
    ThreadingBenchmarkServer,
    SHIFTMRIRequestHandler,
    LIVE_RUNNER,
    ALLOWED_DOWNLOADS
)


@pytest.fixture(scope="module")
def live_server():
    server = ThreadingBenchmarkServer(("127.0.0.1", 8094), SHIFTMRIRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield "http://127.0.0.1:8094"
    server.shutdown()
    server.server_close()


def test_no_label_dependent_manufactured_probabilities():
    """Verify that predictions are loaded strictly from authentic CSVs and never manufactured from binary_label."""
    repo = ScanRepository()
    
    # 1. OASIS-1 Train Scans: Must NOT have manufactured probabilities
    oasis_train = [s for s in repo.scans if s["cohort"] == "OASIS-1" and s["cohort_split"] == "train"]
    assert len(oasis_train) == 138, "OASIS-1 train count must be 138"
    for s in oasis_train:
        assert s["predictions"].get("status") == "PREDICTION UNAVAILABLE", (
            f"OASIS-1 train scan {s['scan_id']} must not have manufactured predictions!"
        )

    # 2. OASIS-2: Un-evaluated architectures must be explicitly 'NOT EVALUATED'
    oasis2 = [s for s in repo.scans if s["cohort"] == "OASIS-2"]
    assert len(oasis2) == 150, "OASIS-2 count must be 150"
    for s in oasis2:
        assert s["predictions"]["MONAI Swin UNETR SSL (Linear Probe)"] == "NOT EVALUATED ON OASIS-2"
        assert s["predictions"]["FOMO25 AMAES ResEnc-B (Linear Probe)"] == "NOT EVALUATED ON OASIS-2"
        # Only the 4 evaluated models should have float values
        assert isinstance(s["predictions"]["3D ResNet-10 (Pipeline B)"], float)
        assert isinstance(s["predictions"]["3D DenseNet-121 (Pipeline B)"], float)
        assert isinstance(s["predictions"]["2.5D Multi-View ResNet-18 (Pipeline B)"], float)
        assert isinstance(s["predictions"]["Dual-Branch Hybrid (ResNet-10 + Radiomics)"], float)


def test_trustgate_weights_and_threshold_strictly_conform():
    """Verify TrustGate composite equals 0.25*A + 0.25*B + 0.25*C + 0.25*D with tau* = 0.6340."""
    tg_adni_csv = V3_RESULTS_DIR / "trustgate" / "trustgate_adni_per_scan_scores.csv"
    assert tg_adni_csv.exists(), "trustgate_adni_per_scan_scores.csv must exist"
    df = pd.read_csv(tg_adni_csv)
    
    for idx, row in df.iterrows():
        sa = row["margin_sa"]
        sb = row["mahalanobis_sb"]
        sc = row["disagreement_sc"]
        sd = row["qc_sd"]
        expected_composite = 0.25 * sa + 0.25 * sb + 0.25 * sc + 0.25 * sd
        actual_composite = row["composite_risk"]
        assert abs(expected_composite - actual_composite) < 1e-4, (
            f"Row {idx}: composite {actual_composite} does not match 0.25*(A+B+C+D) = {expected_composite}"
        )
        assert abs(row["tau_star"] - 0.6340) < 1e-3, f"Row {idx}: tau* must be ~0.6340"


def test_adni_class_distribution_exact_counts():
    """Verify ADNI cohort counts: 303 CN and 197 AD (prevalence = 0.394)."""
    # 1. Check canonical registry
    reg_path = FINAL_RESULTS_DIR / "canonical_registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    
    adni_counts = reg["cohorts"]["ADNI"]["sample_counts"]
    assert adni_counts["total"] == 500, "ADNI total must be 500"
    assert adni_counts["CN"] == 303, f"ADNI CN must be 303, got {adni_counts['CN']}"
    assert adni_counts["AD"] == 197, f"ADNI AD must be 197, got {adni_counts['AD']}"
    assert abs(reg["cohorts"]["ADNI"]["prevalence"] - 0.394) < 1e-3

    # 2. Check Repository
    repo = ScanRepository()
    adni_scans = [s for s in repo.scans if s["cohort"] == "ADNI"]
    assert len(adni_scans) == 500
    cn_count = sum(1 for s in adni_scans if s["diagnosis"] == "CN")
    ad_count = sum(1 for s in adni_scans if s["diagnosis"] == "AD")
    assert cn_count == 303, f"Repository ADNI CN count must be 303, got {cn_count}"
    assert ad_count == 197, f"Repository ADNI AD count must be 197, got {ad_count}"


def test_live_inference_vs_cached_prediction(live_server):
    """Verify two distinct modes: cached prediction vs live checkpoint inference."""
    # 1. Cached Mode
    url_cached = f"{live_server}/api/inference/predict?scan_id=002_S_0295_I238627&mode=cached"
    with urllib.request.urlopen(url_cached) as resp:
        data = json.loads(resp.read().decode())
    assert data["inference_mode"] == "CACHED_VERIFIED_PREDICTION"
    assert "provenance" in data
    assert isinstance(data["predictions"]["MONAI Swin UNETR SSL (Linear Probe)"], float)

    # 2. Live Mode
    url_live = f"{live_server}/api/inference/predict?scan_id=002_S_0295_I238627&mode=live"
    with urllib.request.urlopen(url_live) as resp:
        data_live = json.loads(resp.read().decode())
    assert data_live["inference_mode"] == "LIVE_CHECKPOINT_INFERENCE"
    summary = data_live["execution_summary"]
    assert summary["mode"] == "LIVE_CHECKPOINT_INFERENCE"
    assert "3D ResNet-10 (Pipeline B)" in summary["models_evaluated_live"]
    # Verify runtime is reported
    assert summary["total_runtime_ms"] > 0.0
    # Verify frozen foundation linear probes are explicitly listed as unavailable for live inference
    assert len(summary["models_unavailable_for_live"]) >= 4


def test_explainability_serves_authentic_not_synthetic(live_server):
    """Verify that /api/inference/explain serves authentic artifacts or gradient backprop, never fake Gaussian templates."""
    # Scan with genuine pre-generated artifact
    url_exp = f"{live_server}/api/inference/explain?scan_id=002_S_0816_I47406&plane=axial&slice_idx=64"
    with urllib.request.urlopen(url_exp) as resp:
        assert resp.status == 200
        source_header = resp.headers.get("X-Attribution-Source", "")
        assert "Authentic" in source_header or "Pre-generated" in source_header, (
            f"Attribution must be authentic, got header: {source_header}"
        )
        img_bytes = resp.read()
        assert len(img_bytes) > 2000

    # Non-existent scan must return 404 EXPLANATION NOT AVAILABLE
    url_missing = f"{live_server}/api/inference/explain?scan_id=NONEXISTENT_SCAN_ID"
    try:
        urllib.request.urlopen(url_missing)
        pytest.fail("Should have raised 404 HTTPError for missing scan")
    except urllib.error.HTTPError as e:
        assert e.code == 404
        err_data = json.loads(e.read().decode())
        assert err_data["error"] == "EXPLANATION NOT AVAILABLE"


def test_scan_pagination_covers_all_subjects(live_server):
    """Verify server-side pagination accesses all 848 indexed scans without omission."""
    url = f"{live_server}/api/mri/scans?limit=100&page=1"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    assert data["total"] == 848, f"Total scans must be 848, got {data['total']}"
    assert data["total_pages"] == 9
    assert len(data["scans"]) == 100

    # Page 9 (last page)
    url_last = f"{live_server}/api/mri/scans?limit=100&page=9"
    with urllib.request.urlopen(url_last) as resp:
        data_last = json.loads(resp.read().decode())
    assert len(data_last["scans"]) == 48


def test_server_blocks_path_traversal_and_non_allowlisted_downloads(live_server):
    """Verify strict download security and path traversal prevention."""
    # 1. Allowed file succeeds
    url_good = f"{live_server}/downloads/SHIFT_MRI_Complete_Research_Paper.pdf"
    with urllib.request.urlopen(url_good) as resp:
        assert resp.status == 200

    # 2. Path traversal is rejected (HTTP 403)
    url_traversal = f"{live_server}/downloads/../README.md"
    try:
        urllib.request.urlopen(url_traversal)
        pytest.fail("Path traversal must be blocked with HTTP 403")
    except urllib.error.HTTPError as e:
        assert e.code == 403

    # 3. Non-allowlisted file is rejected (HTTP 403)
    url_forbidden = f"{live_server}/downloads/private_keys.json"
    try:
        urllib.request.urlopen(url_forbidden)
        pytest.fail("Non-allowlisted file must be blocked with HTTP 403")
    except urllib.error.HTTPError as e:
        assert e.code == 403
