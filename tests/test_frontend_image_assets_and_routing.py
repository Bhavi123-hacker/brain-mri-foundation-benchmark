"""
tests/test_frontend_image_assets_and_routing.py
Verification of static asset routing, image MIME types, genuine 404 handling,
domain shift V3 cross-cohort integrity, and real browser DOM rendering.
"""

import io
import os
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_URL = "http://127.0.0.1:8000"


def test_authorized_images_served_with_correct_mime_and_decode():
    """Verify that all authorized image categories return HTTP 200, image/png, and valid PIL decode."""
    test_paths = [
        # Figures
        "/results/final_campaign/figures/01_model_auc_comparison.png",
        "/results/final_campaign/figures/07_domain_shift_comparison.png",
        # Training diagnostics curves
        "/results/final_campaign/training_diagnostics/train_vs_validation_curves/resnet18_oasis_train_vs_val.png",
        "/results/final_campaign/training_diagnostics/train_vs_validation_curves/resnet34_oasis_train_vs_val.png",
        # Frontend interpretability slices
        "/results/final_campaign/frontend_data/interpretability/adni/resnet18/002_S_0816/axial_overlay.png",
        "/results/final_campaign/frontend_data/interpretability/adni/resnet18/002_S_0816/axial_original.png",
        "/results/final_campaign/frontend_data/interpretability/adni/resnet18/002_S_0816/axial_cam.png",
        "/results/final_campaign/frontend_data/interpretability/oasis/resnet18/OAS1_0001_MR1/axial_overlay.png",
        # Composite Grad-CAM
        "/results/final_campaign/interpretability/adni_resnet18_gradcam/002_S_0816_false_negative.png",
        "/results/final_campaign/interpretability/oasis_resnet18_gradcam/OAS1_0001_MR1_true_negative.png",
        # Foundation adaptation curves
        "/results/final_campaign/foundation_adaptation/adaptation_curves/swinunetr_linear_probe_train_vs_val.png"
    ]

    for path in test_paths:
        url = SERVER_URL + path
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200, f"Expected 200 for {path}, got {resp.status}"
            content_type = resp.headers.get("Content-Type", "")
            assert "image/png" in content_type, f"Expected image/png for {path}, got {content_type}"
            body = resp.read()
            assert len(body) > 0, f"Empty body for {path}"
            # Verify image decodes cleanly
            img = Image.open(io.BytesIO(body))
            assert img.format == "PNG", f"Decoded format was {img.format}, expected PNG for {path}"
            assert img.size[0] > 0 and img.size[1] > 0


def test_missing_and_disallowed_assets_return_genuine_404():
    """Verify that missing files and disallowed file extensions return genuine HTTP 404 (NOT 200 HTML)."""
    invalid_requests = [
        "/results/final_campaign/figures/non_existent_figure_404.png",
        "/results/final_campaign/training_diagnostics/train_vs_validation_curves/missing_file.png",
        "/results/final_campaign/frontend_data/interpretability/missing_subject/axial_overlay.png",
        "/results/final_campaign/interpretability/missing_gradcam.png",
        "/results/final_campaign/training_diagnostics/train_vs_validation_curves/exploit.py",
        "/results/final_campaign/figures/model.pt",
        "/results/final_campaign/secret_manifest.csv",
        "/frontend/js/app.py",
        "/non_existent_page.html"
    ]

    for path in invalid_requests:
        url = SERVER_URL + path
        req = urllib.request.Request(url)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        
        err = exc_info.value
        assert err.code in (404, 403), f"Expected 404/403 for {path}, got {err.code}"
        # Crucially verify it did NOT return HTML with HTTP 200
        assert err.code != 200


def test_domain_shift_data_and_v3_reconciled_leaderboard():
    """Verify that domain shift displays V3 reconciled models and pilot archive has safe keys."""
    # 1. Check registry has 10 V3 models with valid OASIS test AUC and ADNI AUC
    url = SERVER_URL + "/api/registry"
    with urllib.request.urlopen(url, timeout=5) as resp:
        assert resp.status == 200
        reg = json.loads(resp.read().decode("utf-8"))

    rankings = [m for m in reg.get("model_rankings_adni", []) if isinstance(m.get("rank"), int)]
    assert len(rankings) == 10, f"Expected 10 benchmark models, got {len(rankings)}"

    for m in rankings:
        assert "oasis_test_auc" in m and isinstance(m["oasis_test_auc"], (int, float))
        assert "auc" in m and isinstance(m["auc"], (int, float))
        assert "delta_auc" in m and isinstance(m["delta_auc"], (int, float))
        assert "brier" in m and isinstance(m["brier"], (int, float))
        assert 0.0 <= m["oasis_test_auc"] <= 1.0
        assert 0.0 <= m["auc"] <= 1.0

    # 2. Check canonical domain_shift.json (pilot archive) has valid numerical fields
    domain_shift_file = PROJECT_ROOT / "results" / "final_campaign" / "canonical_frontend_data" / "domain_shift.json"
    assert domain_shift_file.exists()
    pilot_data = json.loads(domain_shift_file.read_text(encoding="utf-8"))
    assert len(pilot_data) >= 8

    for d in pilot_data:
        # Verify that safe accessors work without throwing
        model = d.get("model_name") or d.get("model")
        assert model is not None
        adni_auc = d.get("adni_roc_auc") if d.get("adni_roc_auc") is not None else d.get("auc")
        assert adni_auc is not None and isinstance(adni_auc, (int, float))


def test_header_badge_and_footer_metadata():
    """Verify header badge shows verified tests and footer references N=198, N=500, N=150 cohorts."""
    index_html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

    # Header badge in HTML and JS
    assert "117/117 Tests Passing" in index_html or "Tests Passing" in index_html
    assert "117/117 Tests Passing" in app_js
    assert "93/93 Tests Passing" not in app_js
    assert "93/93" not in index_html

    # Footer metadata in HTML
    assert "N=198" in index_html
    assert "N=500" in index_html
    assert "N=150" in index_html
    assert "small cohorts ($N=50$)" not in index_html


def test_real_browser_section_dom_rendering():
    """Launch headless browser and verify error-free DOM rendering across primary research sections."""
    browser_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(browser_exe):
        browser_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    if not os.path.exists(browser_exe):
        pytest.skip("Neither Chrome nor Edge is installed on this system.")

    sections = [
        ("overview", "Foundation Models vs. Supervised Baselines"),
        ("domain-shift", "Ten-Architecture Cross-Cohort Generalization Audit"),
        ("interpretability", "3D CNN Interpretability"),
        ("training-dynamics", "Historical Development Diagnostics Archive"),
        ("cnn-benchmark", "3D CNN Benchmark: OASIS-1 vs ADNI")
    ]

    for sec, expected_phrase in sections:
        url = f"{SERVER_URL}/#{sec}"
        cmd = [
            browser_exe,
            "--headless=new",
            "--disable-gpu",
            "--virtual-time-budget=4000",
            "--dump-dom",
            url
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        dom = res.stdout.decode("utf-8", errors="ignore")
        assert "fatal-error" not in dom, f"Fatal error occurred rendering #{sec}"
        assert expected_phrase in dom, f"Expected phrase '{expected_phrase}' not found in #{sec} DOM"
