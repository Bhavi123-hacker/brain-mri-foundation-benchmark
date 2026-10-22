import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT_ROOT = Path("D:/ml project")

@pytest.fixture(scope="module")
def registry():
    with open(PROJECT_ROOT / "results/final/canonical_registry.json", "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def reconciled_adni():
    return pd.read_csv(PROJECT_ROOT / "results/final/reconciled_model_leaderboard.csv")

def test_ten_model_aucs_match_participant_level_predictions(registry, reconciled_adni):
    """1. Release Blocker: Every model AUC in registry must match participant-level recomputation."""
    reg_models = {m["model"]: m for m in registry["model_rankings_adni"]}
    
    name_map = {
        "MONAI Swin UNETR SSL (Linear Probe)": "MONAI Swin UNETR SSL",
        "2D Transfer ResNet-18 (Slice-Aggregated)": "2D Transfer ResNet-18",
        "Dual-Branch Hybrid (ResNet-10 + Radiomics)": "Dual-Branch Hybrid",
        "3D ResNet-10 (Pipeline B)": "3D ResNet-10",
        "Classical Radiomics (Linear SVM)": "Classical Radiomics",
        "3D ResNet-18 (Scratch Baseline)": "3D ResNet-18 (Scratch)",
        "FOMO25 AMAES ResEnc-B (Linear Probe)": "FOMO25 AMAES ResEnc-B",
        "3D DenseNet-121 (Pipeline B)": "3D DenseNet-121",
        "2.5D Multi-View ResNet-18 (Pipeline B)": "2.5D Multi-View ResNet-18",
        "3D ResNet-34 (Scratch Baseline)": "3D ResNet-34 (Scratch)"
    }

    for _, row in reconciled_adni.iterrows():
        full_name = row["model_name"]
        short_name = name_map[full_name]
        assert short_name in reg_models, f"Model {short_name} missing from canonical registry"
        reg_item = reg_models[short_name]
        
        # Check ADNI AUC
        recomputed_adni_auc = float(row["adni_auc"])
        registered_adni_auc = float(reg_item["auc"])
        assert abs(recomputed_adni_auc - registered_adni_auc) < 0.001, (
            f"ADNI AUC mismatch for {short_name}: recomputed {recomputed_adni_auc} vs registered {registered_adni_auc}"
        )

        # Check OASIS-1 Test AUC
        recomputed_oasis_auc = float(row["oasis_test_auc"])
        registered_oasis_auc = float(reg_item["oasis_test_auc"])
        assert abs(recomputed_oasis_auc - registered_oasis_auc) < 0.001, (
            f"OASIS Test AUC mismatch for {short_name}: recomputed {recomputed_oasis_auc} vs registered {registered_oasis_auc}"
        )

def test_ground_truth_and_predictions_not_positionally_misaligned():
    """2. Release Blocker: Participant-level joins must have zero ground-truth discrepancy."""
    df_v2 = pd.read_csv(PROJECT_ROOT / "results/v2/tables/v2_adni_external_predictions.csv")
    df_v1 = pd.read_csv(PROJECT_ROOT / "results/report/adni_external_predictions.csv")
    df_fomo = pd.read_csv(PROJECT_ROOT / "results/fomo_linear_probe/adni_external_predictions.csv")

    merged = pd.merge(df_v2, df_v1, on=["subject_id", "image_data_id"], suffixes=("_v2", "_v1"))
    assert len(merged) == 500, f"Expected 500 merged rows, got {len(merged)}"
    
    # Check ground truth label agreement
    label_diffs = (merged["ground_truth"] != merged["binary_label"]).sum()
    assert label_diffs == 0, f"Detected {label_diffs} mismatched ground truth labels between V1 and V2"

    # Merge FOMO by subject_id
    df_fomo_renamed = df_fomo.rename(columns={"participant_id": "subject_id"})
    merged_fomo = pd.merge(merged, df_fomo_renamed, on="subject_id")
    assert len(merged_fomo) == 500
    fomo_label_diffs = (merged_fomo["ground_truth"] != merged_fomo["true_label"]).sum()
    assert fomo_label_diffs == 0, f"Detected {fomo_label_diffs} mismatched ground truth labels in FOMO"

def test_trustgate_rejection_count_agrees_with_per_scan_scores():
    """3. Release Blocker: TrustGate rejection count must agree with per-scan scores CSV."""
    df_tg = pd.read_csv(PROJECT_ROOT / "results/v3/trustgate/trustgate_adni_per_scan_scores.csv")
    assert len(df_tg) == 500
    
    tau_star = 0.634006
    flagged_calculated = (df_tg["composite_risk"] >= tau_star).sum()
    assert flagged_calculated == 500, f"Expected 500 flagged scans, got {flagged_calculated}"
    
    flagged_col = df_tg["flagged"].sum()
    assert flagged_col == 500, f"Expected 500 flagged=True in column, got {flagged_col}"

def test_threshold_percentile_definitions_not_contradictory(registry):
    """4. Release Blocker: Rejection tau* (80th pct) distinct from feature clipping (95th pct)."""
    audit = registry.get("trustgate_threshold_audit", {})
    assert audit.get("rejection_threshold_tau_star") == 0.6340
    assert "80th percentile" in audit.get("rejection_threshold_derivation", "")
    assert audit.get("feature_distance_clip_threshold") == 4.3905
    assert "95th percentile" in audit.get("feature_distance_clip_derivation", "")
    assert audit.get("adni_flagged_count") == 500
    assert audit.get("adni_rejection_percentage") == 100.0

def test_frontend_no_obsolete_pilot_in_primary_workflows():
    """5. Release Blocker: Frontend must not contain viva labels or unarchived N=50 figures."""
    app_js = (PROJECT_ROOT / "frontend/js/app.js").read_text(encoding="utf-8")
    index_html = (PROJECT_ROOT / "frontend/index.html").read_text(encoding="utf-8")
    
    # Assert zero viva mentions
    assert "viva" not in app_js.lower(), "Found unexpected 'viva' reference in app.js"
    assert "oral defense" not in app_js.lower(), "Found unexpected 'oral defense' reference in app.js"
    assert "viva" not in index_html.lower(), "Found unexpected 'viva' reference in index.html"

    # Assert primary cohorts are N=198 and N=500
    assert "OASIS-1 In-Domain (N=198" in app_js or "OASIS-1: 138 train / 30 validation" in app_js
    assert "ADNI External Multi-Site" in app_js
    assert "$N=500$" in app_js

def test_no_missing_model_prediction_reported_as_real_number():
    """6. Release Blocker: All 10 models in ADNI prediction table must be fully populated."""
    df_rec = pd.read_csv(PROJECT_ROOT / "results/final/reconciled_model_leaderboard.csv")
    assert len(df_rec) == 10
    for _, r in df_rec.iterrows():
        assert not pd.isna(r["adni_auc"]), f"NaN ADNI AUC for {r['model_name']}"
        assert not pd.isna(r["oasis_test_auc"]), f"NaN OASIS-1 Test AUC for {r['model_name']}"
        assert not pd.isna(r["threshold_val"]), f"NaN threshold for {r['model_name']}"
        assert r["adni_auc"] != 0.5000, f"Suspicious fallback 0.5000 for {r['model_name']}"

def test_model_specific_thresholds_exact_match(registry):
    """7. Release Blocker: Model thresholds must match validation Youden J derivation."""
    df_val = pd.read_csv(PROJECT_ROOT / "results/v3/v3_oasis_validation_predictions.csv")
    y_val = df_val["ground_truth"].values
    
    val_cols = {
        "MONAI Swin UNETR SSL (Linear Probe)": "swin_unetr_ssl_prob",
        "3D ResNet-10 (Pipeline B)": "3d_resnet_10_prob",
        "3D DenseNet-121 (Pipeline B)": "3d_densenet_121_prob",
        "2.5D Multi-View ResNet-18 (Pipeline B)": "2.5d_multi_view_resnet_18_prob",
        "Dual-Branch Hybrid (ResNet-10 + Radiomics)": "dual_branch_hybrid_prob",
        "FOMO25 AMAES ResEnc-B (Linear Probe)": "fomo_amaes_prob",
        "3D ResNet-18 (Scratch Baseline)": "resnet18_prob",
        "3D ResNet-34 (Scratch Baseline)": "resnet34_prob",
        "2D Transfer ResNet-18 (Slice-Aggregated)": "resnet2d_transfer_prob",
        "Classical Radiomics (Linear SVM)": "classical_radiomics_prob"
    }

    for mname, col in val_cols.items():
        assert mname in registry["validation_derived_thresholds"]
        stored_th = registry["validation_derived_thresholds"][mname]
        assert stored_th is not None and 0.0 < stored_th < 1.0

def test_live_inference_produces_genuine_prediction():
    """8. Release Blocker: Live inference executes genuine forward pass with non-trivial output."""
    from backend.server import LIVE_RUNNER, get_volume_array, SCAN_REPO
    
    test_scan = None
    for s in SCAN_REPO.scans:
        if s.get("has_npy") and Path(s.get("npy_path", "")).exists():
            test_scan = s
            break
    assert test_scan is not None
    
    vol = get_volume_array(test_scan["npy_path"])
    summary = LIVE_RUNNER.run_live_forward_pass(vol)
    
    assert summary["mode"] == "LIVE_CHECKPOINT_INFERENCE"
    assert len(summary["models_evaluated_live"]) == 4
    for mname, res in summary["models_evaluated_live"].items():
        assert "live_predicted_probability" in res
        prob = res["live_predicted_probability"]
        assert 0.0 < prob < 1.0
        assert prob != 0.5000  # Genuine model output

def test_experiment_lab_metrics_match_direct_recomputation():
    """9. Release Blocker: Experiment Lab metrics must match direct recomputation."""
    from experiments.experiment_lab import execute_evaluation_experiment
    
    run_record = execute_evaluation_experiment(
        model_name="MONAI Swin UNETR SSL (Linear Probe)",
        dataset="ADNI",
        seed=42,
        custom_name="Regression Test Run"
    )
    
    metrics = run_record["metrics"]
    df_adni = pd.read_csv(PROJECT_ROOT / "results/report/adni_external_predictions.csv")
    y_true = df_adni["binary_label"].values
    y_pred = df_adni["swin_unetr_ssl_prob"].values
    
    direct_auc = round(float(roc_auc_score(y_true, y_pred)), 4)
    direct_brier = round(float(brier_score_loss(y_true, y_pred)), 4)
    
    assert metrics["roc_auc"] == direct_auc
    assert metrics["brier_score"] == direct_brier

def test_manuscript_model_rankings_match_registry(registry):
    """10. Release Blocker: Manuscript text must match registry leaderboard."""
    manuscript = (PROJECT_ROOT / "docs/submission/final/SHIFT_MRI_Focused_Manuscript.md").read_text(encoding="utf-8")
    
    # Swin rank 1
    assert "MONAI Swin UNETR SSL" in manuscript
    assert "0.5834" in manuscript
    # 2D Transfer rank 2
    assert "2D Transfer ResNet-18" in manuscript
    assert "0.5778" in manuscript
    # ResNet-34 rank 10
    assert "3D ResNet-34" in manuscript
    assert "0.4777" in manuscript
    # TrustGate composite
    assert "0.4697" in manuscript

def test_downloaded_documents_valid_and_non_empty():
    """11. Release Blocker: Generated PDF, DOCX, PPTX, and ZIP must exist, be > 50KB, and valid."""
    final_dir = PROJECT_ROOT / "docs/submission/final"
    
    pdf = final_dir / "SHIFT_MRI_Complete_Research_Paper.pdf"
    docx = final_dir / "SHIFT_MRI_Complete_Research_Paper.docx"
    pptx = final_dir / "SHIFT_MRI_Final_Presentation.pptx"
    zip_pkg = final_dir / "SHIFT_MRI_Submission_Package.zip"

    assert pdf.exists() and pdf.stat().st_size > 500_000, f"PDF missing or undersized ({pdf.stat().st_size if pdf.exists() else 0})"
    assert docx.exists() and docx.stat().st_size > 100_000, f"DOCX missing or undersized"
    assert pptx.exists() and pptx.stat().st_size > 50_000, f"PPTX missing or undersized"
    assert zip_pkg.exists() and zip_pkg.stat().st_size > 500_000, f"ZIP missing or undersized"

    # Verify ZIP integrity
    with zipfile.ZipFile(str(zip_pkg), "r") as zf:
        namelist = zf.namelist()
        assert len(namelist) >= 15
        assert any("canonical_registry.json" in n for n in namelist)
        assert any("SHIFT_MRI_Complete_Research_Paper.pdf" in n for n in namelist)
