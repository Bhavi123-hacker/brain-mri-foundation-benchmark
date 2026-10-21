"""
scripts/run_v3_trustgate_campaign.py
Implementation, empirical evaluation, and ablation study for SHIFT-MRI TrustGate:
Source-locked failure-risk estimation for brain MRI classification under distribution shift.

Urgent Audit Corrections Incorporated:
1. Reconstruct TrustGate validation scores strictly from authentic validation data
   (validation predictions, validation latent features, validation multi-model disagreement,
   and validation QC metrics). Zero test data used for rejection threshold calibration.
2. Assert participant-ID alignment across all predictions, FOMO embeddings, and QC metadata.
3. Diagnose and document FOMO Mahalanobis distance saturation on ADNI (all samples exceed q95).
4. Explicitly document mathematical equivalence of Margin, MSP, and Entropy for binary probabilities.
5. Accurately report Partial AURC (pAURC_[0.5, 1.0]) and Full AURC over [0.0, 1.0].
6. Honestly present TrustGate composite vs simple Margin parity/degradation findings.
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.v3.shift_mri import compute_stratified_auc_ci, find_optimal_youden_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory setup
TRUSTGATE_DIR = os.path.join(PROJECT_ROOT, "results", "v3", "trustgate")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "results", "v3", "trustgate_ablation")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "v3", "figures")
DOCS_V3_DIR = os.path.join(PROJECT_ROOT, "docs", "research", "v3")

for d in [TRUSTGATE_DIR, ABLATION_DIR, FIGURES_DIR, DOCS_V3_DIR]:
    os.makedirs(d, exist_ok=True)


def compute_risk_coverage_metrics(
    risks: List[float], coverages: List[float], y_true: np.ndarray, preds: np.ndarray, sorted_indices: np.ndarray, threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute Partial Area Under Risk-Coverage Curve (pAURC_[0.5, 1.0]),
    normalized pAURC, and complete Full AURC over [0.0, 1.0].
    """
    r_arr = np.array(risks)
    c_arr = np.array(coverages)
    cov_span = float(abs(c_arr[0] - c_arr[-1]))
    paurc = float(np.sum(0.5 * (r_arr[:-1] + r_arr[1:]) * np.abs(np.diff(c_arr))))
    norm_paurc = float(paurc / cov_span) if cov_span > 0 else paurc

    # Full AURC over [0.0, 1.0]
    n = len(sorted_indices)
    grid_covs = np.linspace(1.0, 0.0, 101)
    grid_risks = []
    for fc in grid_covs:
        if fc == 0.0:
            grid_risks.append(0.0)
        else:
            k = max(1, int(round(fc * n)))
            ret = sorted_indices[:k]
            grid_risks.append(float(np.mean(preds[ret] != y_true[ret])))
    grid_risks_arr = np.array(grid_risks)
    full_aurc = float(np.sum(0.5 * (grid_risks_arr[:-1] + grid_risks_arr[1:]) * np.abs(np.diff(grid_covs))))

    return paurc, norm_paurc, full_aurc


def main():
    print("=" * 80, flush=True)
    print("STARTING SHIFT-MRI TRUSTGATE EVALUATION CAMPAIGN (AUDITED & RECONSTRUCTED)", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 80, flush=True)

    # --------------------------------------------------------------------------
    # 1. Load All Source & External Data with Exact ID Alignment Assertions
    # --------------------------------------------------------------------------
    # Authentic OASIS-1 Validation Data
    val_preds = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "v3", "v3_oasis_validation_predictions.csv"))

    # OASIS-1 Test Data
    v1_o = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "report", "oasis_locked_test_predictions.csv"))
    v2_o = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "v2", "tables", "v2_oasis_locked_test_predictions.csv"))
    fomo_o = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "oasis_test_predictions.csv"))
    f_o_raw = torch.load(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "oasis_fomo_features.pt"), map_location="cpu", weights_only=False)

    # External ADNI Data
    v1_a = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "report", "adni_external_predictions.csv"))
    v2_a = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "v2", "tables", "v2_adni_external_predictions.csv"))
    fomo_a = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "adni_external_predictions.csv"))
    f_a_raw = torch.load(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "adni_fomo_features.pt"), map_location="cpu", weights_only=False)

    # Manifests with QC metrics
    adni_manifest = pd.read_csv("D:/brain_mri_data/ADNI/manifests/adni_full_preprocessed_manifest.csv")
    oasis_cache = pd.read_csv("D:/brain_mri_data/OASIS1/manifests/oasis1_preprocessed_cache_manifest.csv")

    # --- ID Alignment Verification (Fix 3) ---
    print("\n>>> Verifying Exact Participant-ID Alignment Across Splits...")
    splits = np.array(f_o_raw["splits"])
    tr_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    # OASIS-1 Validation ID checks
    assert len(val_preds) == 30, f"Expected 30 val predictions, got {len(val_preds)}"
    assert val_preds["session_id"].nunique() == 30, "Validation session IDs are not unique"
    val_fomo_ids = np.array([f"{p}_MR1" for p in np.array(f_o_raw["participant_ids"])[val_mask]])
    assert (val_fomo_ids == val_preds["session_id"].values).all(), "Validation FOMO ID mismatch!"
    oasis_val_cache = oasis_cache[oasis_cache["split"] == "val"].reset_index(drop=True)
    assert (oasis_val_cache["session_id"].values == val_preds["session_id"].values).all(), "Validation QC cache ID mismatch!"

    # OASIS-1 Test ID checks
    assert len(v1_o) == 30, f"Expected 30 test predictions, got {len(v1_o)}"
    assert v1_o["session_id"].nunique() == 30, "Test session IDs are not unique"
    assert (v1_o["session_id"].values == v2_o["session_id"].values).all(), "Test V1 vs V2 session ID mismatch!"
    test_fomo_ids = np.array([f"{p}_MR1" for p in np.array(f_o_raw["participant_ids"])[test_mask]])
    assert (test_fomo_ids == v1_o["session_id"].values).all(), "Test FOMO ID mismatch!"
    oasis_test_cache = oasis_cache[oasis_cache["split"] == "test"].reset_index(drop=True)
    assert (oasis_test_cache["session_id"].values == v1_o["session_id"].values).all(), "Test QC cache ID mismatch!"

    # ADNI External ID checks
    assert len(v1_a) == 500, f"Expected 500 ADNI predictions, got {len(v1_a)}"
    assert v1_a["subject_id"].nunique() == 500, "ADNI subject IDs are not unique"
    assert (v1_a["subject_id"].values == v2_a["subject_id"].values).all(), "ADNI V1 vs V2 subject ID mismatch!"
    assert (v1_a["image_data_id"].values == v2_a["image_data_id"].values).all(), "ADNI image data ID mismatch!"
    assert (v1_a["subject_id"].values == np.array(f_a_raw["participant_ids"])).all(), "ADNI FOMO participant ID mismatch!"
    assert (v1_a["image_data_id"].values == adni_manifest["image_data_id"].values).all(), "ADNI manifest ID mismatch!"
    print("  All participant and session IDs perfectly aligned and verified unique!")

    # --------------------------------------------------------------------------
    # 2. Extract Latent Manifold Parameters strictly from OASIS-1 Train Split (N=138)
    # --------------------------------------------------------------------------
    tr_feats = f_o_raw["features"][tr_mask]   # (138, 320)
    val_feats = f_o_raw["features"][val_mask] # (30, 320)
    test_feats = f_o_raw["features"][test_mask] # (30, 320)
    adni_feats = f_a_raw["features"]          # (500, 320)

    # Source manifold center and regularized covariance
    mu_source = np.mean(tr_feats, axis=0)
    tr_diff = tr_feats - mu_source
    cov_empirical = (tr_diff.T @ tr_diff) / (len(tr_feats) - 1)
    lambda_ridge = 0.10 * np.trace(cov_empirical) / 320.0
    cov_reg = cov_empirical + lambda_ridge * np.eye(320)
    inv_cov = np.linalg.pinv(cov_reg)

    def compute_reg_mahalanobis(feats):
        diff = feats - mu_source
        return np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))

    d_val_mahal = compute_reg_mahalanobis(val_feats)
    d_test_mahal = compute_reg_mahalanobis(test_feats)
    d_adni_mahal = compute_reg_mahalanobis(adni_feats)

    # Component B: Source-locked percentile standardization on validation
    q05_d = float(np.percentile(d_val_mahal, 5.0))
    q95_d = float(np.percentile(d_val_mahal, 95.0))
    if q95_d <= q05_d: q95_d = q05_d + 1.0

    sb_val = np.clip((d_val_mahal - q05_d) / (q95_d - q05_d), 0.0, 1.0)
    sb_test = np.clip((d_test_mahal - q05_d) / (q95_d - q05_d), 0.0, 1.0)
    sb_adni = np.clip((d_adni_mahal - q05_d) / (q95_d - q05_d), 0.0, 1.0)

    # Diagnose Component B saturation (Fix 6)
    adni_saturated_frac = float(np.mean(sb_adni == 1.0))
    print(f"\nComponent B (Feature Distance) Saturation Audit:")
    print(f"  Source Val Mahalanobis Range: [{d_val_mahal.min():.2f}, {d_val_mahal.max():.2f}], q95={q95_d:.2f}")
    print(f"  External ADNI Mahalanobis Range: [{d_adni_mahal.min():.2f}, {d_adni_mahal.max():.2f}]")
    print(f"  ADNI Scans Exceeding Source q95: {adni_saturated_frac*100:.1f}% ({int(adni_saturated_frac*len(sb_adni))}/500)")
    if adni_saturated_frac == 1.0:
        print("  WARNING: 100% of ADNI samples saturate sb_adni to 1.0000. Variance is 0.0 -> uninformative AUC 0.5000.")

    # --------------------------------------------------------------------------
    # 3. Component C: Multi-Model Disagreement across 6 Diverse Architectures
    # --------------------------------------------------------------------------
    def get_disagreement(p1, p2, p3, p4, p5, p6):
        pred_mat = np.column_stack([p1, p2, p3, p4, p5, p6])
        return np.clip(np.std(pred_mat, axis=1) / 0.50, 0.0, 1.0)

    # Disagreement on validation (strictly from authentic validation predictions)
    sc_val = get_disagreement(
        val_preds["swin_unetr_ssl_prob"].values,
        val_preds["fomo_amaes_prob"].values,
        val_preds["dual_branch_hybrid_prob"].values,
        val_preds["3d_resnet_10_prob"].values,
        val_preds["2.5d_multi_view_resnet_18_prob"].values,
        val_preds["resnet2d_transfer_prob"].values,
    )

    # Disagreement on test
    sc_test = get_disagreement(
        v1_o["swin_unetr_ssl_prob"].values,
        fomo_o["fomo_amaes_prob"].values,
        v2_o["dual_branch_hybrid_prob"].values,
        v2_o["3d_resnet_10_prob"].values,
        v2_o["2.5d_multi_view_resnet_18_prob"].values,
        v1_o["resnet2d_transfer_prob"].values,
    )

    # Disagreement on ADNI
    sc_adni = get_disagreement(
        v1_a["swin_unetr_ssl_prob"].values,
        fomo_a["fomo_amaes_prob"].values,
        v2_a["dual_branch_hybrid_prob"].values,
        v2_a["3d_resnet_10_prob"].values,
        v2_a["2.5d_multi_view_resnet_18_prob"].values,
        v1_a["resnet2d_transfer_prob"].values,
    )

    # --------------------------------------------------------------------------
    # 4. Component D: Acquisition / Image QC Anomaly
    # --------------------------------------------------------------------------
    oasis_tr_cache = oasis_cache[oasis_cache["split"] == "train"]
    mu_std = float(oasis_tr_cache["val_std"].mean())
    sig_std = float(oasis_tr_cache["val_std"].std()) if float(oasis_tr_cache["val_std"].std()) > 0 else 1.0
    mu_nz = float(oasis_tr_cache["nonzero_fraction"].mean())
    sig_nz = float(oasis_tr_cache["nonzero_fraction"].std()) if float(oasis_tr_cache["nonzero_fraction"].std()) > 0 else 1.0

    def get_qc_anomaly(std_vals, nz_vals):
        dev_std = np.abs(std_vals - mu_std) / (3.0 * sig_std)
        dev_nz = np.abs(nz_vals - mu_nz) / (3.0 * sig_nz)
        return np.clip(0.5 * (dev_std + dev_nz), 0.0, 1.0)

    sd_val = get_qc_anomaly(oasis_val_cache["val_std"].values, oasis_val_cache["nonzero_fraction"].values)
    sd_test = get_qc_anomaly(oasis_test_cache["val_std"].values, oasis_test_cache["nonzero_fraction"].values)
    sd_adni = get_qc_anomaly(adni_manifest["std_val"].values, adni_manifest["nonzero_pct"].values / 100.0)

    # --------------------------------------------------------------------------
    # 5. Component A & Baselines using Primary Model (MONAI Swin UNETR SSL)
    # --------------------------------------------------------------------------
    y_val_gt = val_preds["ground_truth"].astype(int).values
    y_test_gt = v1_o["true_label"].astype(int).values
    y_adni_gt = v1_a["binary_label"].astype(int).values

    p_val = val_preds["swin_unetr_ssl_prob"].values
    p_test = v1_o["swin_unetr_ssl_prob"].values
    p_adni = v1_a["swin_unetr_ssl_prob"].values

    # Derive primary model decision threshold t* STRICTLY on validation predictions (Fix 1)
    t_opt, _ = find_optimal_youden_threshold(y_val_gt, p_val)
    print(f"\nPrimary Model (MONAI Swin UNETR SSL) Source Decision Threshold:")
    print(f"  t* derived strictly from validation split (N=30): {t_opt:.4f}")

    # Margin Risk (Component A)
    sa_val = 1.0 - 2.0 * np.abs(p_val - 0.50)
    sa_test = 1.0 - 2.0 * np.abs(p_test - 0.50)
    sa_adni = 1.0 - 2.0 * np.abs(p_adni - 0.50)

    # Baseline 1: Maximum Softmax Probability (MSP) Risk: 1 - max(p, 1-p)
    msp_val = 1.0 - np.maximum(p_val, 1.0 - p_val)
    msp_test = 1.0 - np.maximum(p_test, 1.0 - p_test)
    msp_adni = 1.0 - np.maximum(p_adni, 1.0 - p_adni)

    # Baseline 2: Predictive Entropy
    def get_entropy(p_arr):
        pc = np.clip(p_arr, 1e-6, 1.0 - 1e-6)
        return - (pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))

    ent_val = get_entropy(p_val)
    ent_test = get_entropy(p_test)
    ent_adni = get_entropy(p_adni)

    # Baseline 6: Simple Average Confidence Aggregation
    agg_val = 0.5 * sa_val + 0.5 * ent_val
    agg_test = 0.5 * sa_test + 0.5 * ent_test
    agg_adni = 0.5 * sa_adni + 0.5 * ent_adni

    # --------------------------------------------------------------------------
    # 6. Reconstruct TrustGate Composite & Derive tau* (Fix 2)
    # --------------------------------------------------------------------------
    # Reconstructed STRICTLY on validation inputs:
    tg_val = 0.25 * sa_val + 0.25 * sb_val + 0.25 * sc_val + 0.25 * sd_val
    tg_test = 0.25 * sa_test + 0.25 * sb_test + 0.25 * sc_test + 0.25 * sd_test
    tg_adni = 0.25 * sa_adni + 0.25 * sb_adni + 0.25 * sc_adni + 0.25 * sd_adni

    # Binary Error Target: Did primary model make a classification error at t*?
    pred_val = (p_val >= t_opt).astype(int)
    pred_test = (p_test >= t_opt).astype(int)
    pred_adni = (p_adni >= t_opt).astype(int)

    err_val = (pred_val != y_val_gt).astype(int)
    err_test = (pred_test != y_test_gt).astype(int)
    err_adni = (pred_adni != y_adni_gt).astype(int)

    print(f"\nClassification Error Rates under Validation Threshold t*={t_opt:.4f}:")
    print(f"  OASIS-1 Val (N=30):   {err_val.sum()}/30 errors ({err_val.mean()*100:.1f}%)")
    print(f"  OASIS-1 Test (N=30):  {err_test.sum()}/30 errors ({err_test.mean()*100:.1f}%)")
    print(f"  ADNI External (N=500): {err_adni.sum()}/500 errors ({err_adni.mean()*100:.1f}%)")

    # --------------------------------------------------------------------------
    # 7. Comparative Evaluation: Error-Detection ROC-AUC & Monotonicity (Fix 8)
    # --------------------------------------------------------------------------
    methods = [
        ("Maximum Probability (MSP) Risk", msp_adni, msp_test, "Rank-equivalent to Margin and Entropy under binary classification (|p - 0.5| monotonic)"),
        ("Predictive Entropy", ent_adni, ent_test, "Rank-equivalent to Margin and MSP under binary classification (symmetric concave around 0.5)"),
        ("Classification Margin Risk", sa_adni, sa_test, "Primary confidence baseline 1 - 2*|p - 0.5|"),
        ("Regularized Feature Distance (Raw Mahalanobis)", d_adni_mahal, d_test_mahal, "Unclipped latent Mahalanobis distance from source manifold"),
        ("Regularized Feature Distance (Clipped / Saturated)", sb_adni, sb_test, "Saturated to 1.0000 across 100% of ADNI scans (zero variance -> AUC 0.5000)"),
        ("Model Disagreement (6 Architectures)", sc_adni, sc_test, "Normalized inter-model standard deviation across diverse architectures"),
        ("Acquisition / Image QC Anomaly", sd_adni, sd_test, "Foreground intensity std and nonzero brain fraction deviations"),
        ("Simple Confidence Aggregation", agg_adni, agg_test, "Average of margin risk and entropy"),
        ("Proposed TrustGate Composite", tg_adni, tg_test, "Prespecified equal-weighted composite (0.25*A + 0.25*B + 0.25*C + 0.25*D)"),
    ]

    benchmark_rows = []
    print("\n>>> Comparing TrustGate Against Baseline Risk Estimators (results/v3/trustgate/)", flush=True)
    for name, r_adni, r_test, note in methods:
        auc_adni, ci_adni = compute_stratified_auc_ci(err_adni, r_adni, n_bootstraps=2000)
        auc_test, ci_test = compute_stratified_auc_ci(err_test, r_test, n_bootstraps=2000)

        benchmark_rows.append({
            "Method": name,
            "ADNI_Error_Detection_AUC": round(auc_adni, 4),
            "ADNI_AUC_95CI": f"[{ci_adni[0]:.4f}, {ci_adni[1]:.4f}]",
            "OASIS_Test_Error_Detection_AUC": round(auc_test, 4),
            "OASIS_AUC_95CI": f"[{ci_test[0]:.4f}, {ci_test[1]:.4f}]",
            "Theoretical_Empirical_Note": note,
        })
        print(f"  {name:50s} | ADNI AUC={auc_adni:.4f} {ci_adni} | OASIS Test AUC={auc_test:.4f}", flush=True)

    bench_df = pd.DataFrame(benchmark_rows)
    bench_csv = os.path.join(TRUSTGATE_DIR, "trustgate_benchmark_results.csv")
    bench_df.to_csv(bench_csv, index=False)
    print(f"Saved TrustGate Benchmark Results to: {bench_csv}", flush=True)

    # --------------------------------------------------------------------------
    # 8. Selective Prediction & Partial AURC / Full AURC Evaluation (Fix 5)
    # --------------------------------------------------------------------------
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    tg_sorted_idx = np.argsort(tg_adni)
    dis_sorted_idx = np.argsort(sc_adni)
    mar_sorted_idx = np.argsort(sa_adni)

    def evaluate_coverage_curve(sorted_indices, risk_name):
        risks, covs = [], []
        curve_records = []
        for cov in coverages:
            k = max(1, int(round(cov * len(sorted_indices))))
            retained = sorted_indices[:k]
            y_sub = y_adni_gt[retained]
            pred_sub = pred_adni[retained]

            err_rate = float(np.mean(pred_sub != y_sub))
            acc = float(np.mean(pred_sub == y_sub))

            fn_count = int(np.sum((pred_sub == 0) & (y_sub == 1)))
            total_ad = int(np.sum(y_sub == 1))
            fn_retention = float(fn_count / total_ad) if total_ad > 0 else 0.0

            curve_records.append({
                "Risk_Estimator": risk_name,
                "Coverage": cov,
                "Retained_N": k,
                "Abstained_N": len(sorted_indices) - k,
                "Selective_Error_Rate": round(err_rate, 4),
                "Selective_Accuracy": round(acc, 4),
                "False_Negative_Retention_Rate": round(fn_retention, 4),
            })
            risks.append(err_rate)
            covs.append(cov)

        paurc, norm_paurc, full_aurc = compute_risk_coverage_metrics(risks, covs, y_adni_gt, pred_adni, sorted_indices, t_opt)
        for r in curve_records:
            r["pAURC_[0.5, 1.0]"] = round(paurc, 4)
            r["Normalized_pAURC"] = round(norm_paurc, 4)
            r["Full_AURC"] = round(full_aurc, 4)
            r["AURC"] = round(paurc, 4)  # For backwards compatibility
        return curve_records, paurc, norm_paurc, full_aurc, risks

    rec_tg, paurc_tg, norm_tg, full_tg, risks_tg = evaluate_coverage_curve(tg_sorted_idx, "Proposed TrustGate Composite")
    rec_dis, paurc_dis, norm_dis, full_dis, risks_dis = evaluate_coverage_curve(dis_sorted_idx, "Model Disagreement")
    rec_mar, paurc_mar, norm_mar, full_mar, risks_mar = evaluate_coverage_curve(mar_sorted_idx, "Classification Margin")

    all_curve_records = rec_tg + rec_dis + rec_mar
    curve_df = pd.DataFrame(all_curve_records)
    curve_csv = os.path.join(TRUSTGATE_DIR, "trustgate_risk_coverage_table.csv")
    curve_df.to_csv(curve_csv, index=False)
    print(f"Saved Risk-Coverage Profile to: {curve_csv}")
    print(f"  TrustGate: pAURC_[0.5, 1.0]={paurc_tg:.4f}, Full AURC={full_tg:.4f}")
    print(f"  Margin:    pAURC_[0.5, 1.0]={paurc_mar:.4f}, Full AURC={full_mar:.4f}")

    # --------------------------------------------------------------------------
    # 9. Source-Calibrated Thresholding & Scanner Subgroup Analysis
    # --------------------------------------------------------------------------
    # Threshold calibrated strictly on OASIS-1 validation to flag top 20% on source
    tau_star_source = float(np.percentile(tg_val, 80.0))
    print(f"\nSource-Calibrated Flagging Threshold tau* (80th pct of validation): {tau_star_source:.4f}", flush=True)

    flagged_mask = (tg_adni > tau_star_source)
    overall_flag_rate = float(np.mean(flagged_mask))
    print(f"Overall ADNI Flagged Fraction: {overall_flag_rate*100:.1f}% ({flagged_mask.sum()}/500 scans)", flush=True)

    # Export authentic per-scan TrustGate scores for ADNI (N=500)
    adni_scores_df = pd.DataFrame({
        "subject_id": adni_manifest["subject_id"].values,
        "image_data_id": adni_manifest["image_data_id"].values,
        "scan_id": [f"{s}_{i}" for s, i in zip(adni_manifest["subject_id"].values, adni_manifest["image_data_id"].values)],
        "margin_sa": sa_adni,
        "mahalanobis_sb": sb_adni,
        "raw_mahalanobis": d_adni_mahal,
        "disagreement_sc": sc_adni,
        "qc_sd": sd_adni,
        "composite_risk": tg_adni,
        "tau_star": tau_star_source,
        "flagged": flagged_mask
    })
    adni_scores_path = os.path.join(TRUSTGATE_DIR, "trustgate_adni_per_scan_scores.csv")
    adni_scores_df.to_csv(adni_scores_path, index=False)
    print(f"Saved authentic ADNI per-scan TrustGate scores to: {adni_scores_path}", flush=True)

    # Export authentic per-scan TrustGate scores for OASIS-1 Test (N=30)
    test_flagged_mask = (tg_test > tau_star_source)
    oasis_test_scores_df = pd.DataFrame({
        "session_id": oasis_test_cache["session_id"].values,
        "scan_id": oasis_test_cache["session_id"].values,
        "margin_sa": sa_test,
        "mahalanobis_sb": sb_test,
        "raw_mahalanobis": d_test_mahal,
        "disagreement_sc": sc_test,
        "qc_sd": sd_test,
        "composite_risk": tg_test,
        "tau_star": tau_star_source,
        "flagged": test_flagged_mask
    })
    oasis_scores_path = os.path.join(TRUSTGATE_DIR, "trustgate_oasis1_test_per_scan_scores.csv")
    oasis_test_scores_df.to_csv(oasis_scores_path, index=False)
    print(f"Saved authentic OASIS-1 Test per-scan TrustGate scores to: {oasis_scores_path}", flush=True)

    # Subgroup breakdown
    subgroups = {
        "Siemens (N=370)": (adni_manifest["manufacturer"].astype(str).str.upper() == "SIEMENS").values,
        "Philips (N=130)": (adni_manifest["manufacturer"].astype(str).str.upper() == "PHILIPS").values,
        "1.5 Tesla (N=165)": (adni_manifest["field_strength"] < 2.0).values,
        "3.0 Tesla (N=335)": (adni_manifest["field_strength"] >= 2.0).values,
    }

    subgroup_records = []
    for sg_name, sg_mask in subgroups.items():
        sg_err = err_adni[sg_mask]
        sg_tg = tg_adni[sg_mask]
        sg_flag = flagged_mask[sg_mask]

        auc_sg, _ = compute_stratified_auc_ci(sg_err, sg_tg, n_bootstraps=1000)
        flag_rate_sg = float(np.mean(sg_flag))

        err_retained = float(np.mean(sg_err[~sg_flag])) if np.sum(~sg_flag) > 0 else float("nan")
        err_flagged = float(np.mean(sg_err[sg_flag])) if np.sum(sg_flag) > 0 else float("nan")

        subgroup_records.append({
            "Scanner_Subgroup": sg_name,
            "Sample_N": int(np.sum(sg_mask)),
            "Baseline_Error_Rate": round(float(np.mean(sg_err)), 4),
            "Flagged_Fraction": round(flag_rate_sg, 4),
            "Error_Detection_AUC": round(auc_sg, 4),
            "Error_Rate_Retained": round(err_retained, 4) if not np.isnan(err_retained) else None,
            "Error_Rate_Flagged": round(err_flagged, 4) if not np.isnan(err_flagged) else None,
        })
        print(f"  Subgroup {sg_name:18s} | Flag Rate={flag_rate_sg*100:.1f}% | Retained Err={err_retained*100:.1f}% vs Flagged Err={err_flagged*100:.1f}%", flush=True)

    sg_df = pd.DataFrame(subgroup_records)
    sg_csv = os.path.join(TRUSTGATE_DIR, "trustgate_scanner_subgroup_analysis.csv")
    sg_df.to_csv(sg_csv, index=False)
    print(f"Saved Scanner Subgroup Analysis to: {sg_csv}", flush=True)

    # --------------------------------------------------------------------------
    # 10. Ablation Study: Individual Component Removal
    # --------------------------------------------------------------------------
    print("\n>>> Running TrustGate Ablation Experiments (results/v3/trustgate_ablation/)", flush=True)
    ablations = [
        ("Full TrustGate (A + B + C + D)", tg_adni),
        ("Ablate A: No Margin (B + C + D)", (sb_adni + sc_adni + sd_adni) / 3.0),
        ("Ablate B: No Feature Distance (A + C + D)", (sa_adni + sc_adni + sd_adni) / 3.0),
        ("Ablate C: No Disagreement (A + B + D)", (sa_adni + sb_adni + sd_adni) / 3.0),
        ("Ablate D: No Acquisition Anomaly (A + B + C)", (sa_adni + sb_adni + sc_adni) / 3.0),
    ]

    tg_full_adni_auc = benchmark_rows[8]["ADNI_Error_Detection_AUC"]  # TrustGate Composite row
    ablation_records = []
    for abl_name, abl_scores in ablations:
        auc_abl, ci_abl = compute_stratified_auc_ci(err_adni, abl_scores, n_bootstraps=2000)

        s_idx = np.argsort(abl_scores)
        _, paurc_abl, norm_abl, full_aurc_abl, _ = evaluate_coverage_curve(s_idx, abl_name)

        ablation_records.append({
            "Ablation_Variant": abl_name,
            "Error_Detection_AUC": round(auc_abl, 4),
            "AUC_95CI": f"[{ci_abl[0]:.4f}, {ci_abl[1]:.4f}]",
            "pAURC_[0.5, 1.0]": round(paurc_abl, 4),
            "Full_AURC": round(full_aurc_abl, 4),
            "AURC": round(paurc_abl, 4),
            "Delta_AUC_vs_Full": round(auc_abl - tg_full_adni_auc, 4),
        })
        print(f"  {abl_name:42s} | AUC={auc_abl:.4f} {ci_abl} | pAURC={paurc_abl:.4f} | Full AURC={full_aurc_abl:.4f}", flush=True)

    abl_df = pd.DataFrame(ablation_records)
    abl_csv = os.path.join(ABLATION_DIR, "trustgate_ablation_results.csv")
    abl_df.to_csv(abl_csv, index=False)
    print(f"Saved Ablation Results to: {abl_csv}", flush=True)

    # --------------------------------------------------------------------------
    # 11. Generate Updated Figures
    # --------------------------------------------------------------------------
    # Figure 1: TrustGate Risk-Coverage Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, risks_tg, "o-", color="#1f77b4", lw=2.5, label=f"TrustGate Composite (pAURC={paurc_tg:.3f})")
    ax.plot(coverages, risks_dis, "s--", color="#ff7f0e", lw=2, label=f"Model Disagreement (pAURC={paurc_dis:.3f})")
    ax.plot(coverages, risks_mar, "^-.", color="#2ca02c", lw=2, label=f"Classification Margin (pAURC={paurc_mar:.3f})")
    ax.set_title("TrustGate Selective Risk vs Coverage on External ADNI (N=500)", fontweight="bold")
    ax.set_xlabel("Coverage (Proportion of Accepted Scans)")
    ax.set_ylabel("Selective Diagnostic Error Rate")
    ax.set_ylim(0.40, 0.65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig_path1 = os.path.join(FIGURES_DIR, "trustgate_risk_coverage.png")
    fig.savefig(fig_path1, dpi=300)
    plt.close(fig)
    print(f"Saved Risk-Coverage plot to: {fig_path1}", flush=True)

    # Figure 2: Error-Detection ROC Curves
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr_tg, tpr_tg, _ = roc_curve(err_adni, tg_adni)
    fpr_mar, tpr_mar, _ = roc_curve(err_adni, sa_adni)
    fpr_dis, tpr_dis, _ = roc_curve(err_adni, sc_adni)
    fpr_sb, tpr_sb, _ = roc_curve(err_adni, sb_adni)

    ax.plot(fpr_tg, tpr_tg, color="#1f77b4", lw=2.5, label=f"TrustGate Composite (AUC={benchmark_rows[8]['ADNI_Error_Detection_AUC']:.3f})")
    ax.plot(fpr_mar, tpr_mar, color="#2ca02c", lw=2, label=f"Margin Alone (AUC={benchmark_rows[2]['ADNI_Error_Detection_AUC']:.3f})")
    ax.plot(fpr_dis, tpr_dis, color="#ff7f0e", lw=2, label=f"Disagreement Alone (AUC={benchmark_rows[5]['ADNI_Error_Detection_AUC']:.3f})")
    ax.plot(fpr_sb, tpr_sb, color="#9467bd", lw=1.5, ls=":", label=f"Feature Dist Clipped (AUC={benchmark_rows[4]['ADNI_Error_Detection_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)

    ax.set_title("Error Detection ROC Curves on External ADNI (N=500)", fontweight="bold")
    ax.set_xlabel("False Positive Rate (Correct Scans Flagged)")
    ax.set_ylabel("True Positive Rate (Errors Successfully Flagged)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig_path2 = os.path.join(FIGURES_DIR, "trustgate_error_detection_roc.png")
    fig.savefig(fig_path2, dpi=300)
    plt.close(fig)
    print(f"Saved Error-Detection ROC plot to: {fig_path2}", flush=True)

    print("=" * 80, flush=True)
    print("ALL TRUSTGATE EXPERIMENTS, ABLATIONS, AND FIGURES COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
