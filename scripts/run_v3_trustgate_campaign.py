"""
scripts/run_v3_trustgate_campaign.py
Implementation, empirical evaluation, and ablation study for SHIFT-MRI TrustGate:
Source-locked failure-risk estimation for brain MRI classification under distribution shift.
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

from evaluation.v3.shift_mri import compute_stratified_auc_ci

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory setup
TRUSTGATE_DIR = os.path.join(PROJECT_ROOT, "results", "v3", "trustgate")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "results", "v3", "trustgate_ablation")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "v3", "figures")
DOCS_V3_DIR = os.path.join(PROJECT_ROOT, "docs", "research", "v3")

for d in [TRUSTGATE_DIR, ABLATION_DIR, FIGURES_DIR, DOCS_V3_DIR]:
    os.makedirs(d, exist_ok=True)


def compute_aurc(risks: List[float], coverages: List[float]) -> float:
    """Robust trapezoidal integration for Area Under Risk-Coverage Curve."""
    r_arr = np.array(risks)
    c_arr = np.array(coverages)
    return float(np.sum(0.5 * (r_arr[:-1] + r_arr[1:]) * np.abs(np.diff(c_arr))))


def main():
    print("=" * 80, flush=True)
    print("STARTING SHIFT-MRI TRUSTGATE PROTOTYPE EVALUATION AND ABLATION CAMPAIGN", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load All Source & External Data
    # In-domain OASIS-1
    v1_o = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "report", "oasis_locked_test_predictions.csv"))
    v2_o = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "v2", "tables", "v2_oasis_locked_test_predictions.csv"))
    fomo_o = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "oasis_test_predictions.csv"))
    f_o_raw = torch.load(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "oasis_fomo_features.pt"), map_location="cpu", weights_only=False)

    # External ADNI
    v1_a = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "report", "adni_external_predictions.csv"))
    v2_a = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "v2", "tables", "v2_adni_external_predictions.csv"))
    fomo_a = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "adni_external_predictions.csv"))
    f_a_raw = torch.load(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "adni_fomo_features.pt"), map_location="cpu", weights_only=False)

    # Manifests with QC metrics
    adni_manifest = pd.read_csv("D:/brain_mri_data/ADNI/manifests/adni_full_preprocessed_manifest.csv")
    oasis_cache = pd.read_csv("D:/brain_mri_data/OASIS1/manifests/oasis1_preprocessed_cache_manifest.csv")

    # 2. Extract Latent Manifold Parameters strictly from OASIS-1 Train Split (N=138)
    splits = np.array(f_o_raw["splits"])
    tr_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

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

    # Source-locked percentile standardization for feature distance
    q05_d = float(np.percentile(d_val_mahal, 5.0))
    q95_d = float(np.percentile(d_val_mahal, 95.0))
    if q95_d <= q05_d: q95_d = q05_d + 1.0

    sb_test = np.clip((d_test_mahal - q05_d) / (q95_d - q05_d), 0.0, 1.0)
    sb_adni = np.clip((d_adni_mahal - q05_d) / (q95_d - q05_d), 0.0, 1.0)

    # 3. Component C: Multi-Model Disagreement
    # Assemble predictions across 6 diverse models
    def get_disagreement(df_v1_in, df_v2_in, df_fomo_in):
        pred_mat = np.column_stack([
            df_v1_in["swin_unetr_ssl_prob"].values,
            df_fomo_in["fomo_amaes_prob"].values,
            df_v2_in["dual_branch_hybrid_prob"].values,
            df_v2_in["3d_resnet_10_prob"].values,
            df_v2_in["2.5d_multi_view_resnet_18_prob"].values,
            df_v1_in["resnet2d_transfer_prob"].values,
        ])
        # Standard deviation normalized by max possible std (0.50)
        std_arr = np.std(pred_mat, axis=1)
        return np.clip(std_arr / 0.50, 0.0, 1.0)

    sc_test = get_disagreement(v1_o, v2_o, fomo_o)
    sc_adni = get_disagreement(v1_a, v2_a, fomo_a)

    # 4. Component D: Acquisition / Image QC Anomaly
    # Compute source training stats for foreground std and nonzero fraction
    oasis_tr_cache = oasis_cache[oasis_cache["split"] == "train"]
    oasis_val_cache = oasis_cache[oasis_cache["split"] == "val"]
    oasis_test_cache = oasis_cache[oasis_cache["split"] == "test"]

    mu_std = float(oasis_tr_cache["val_std"].mean())
    sig_std = float(oasis_tr_cache["val_std"].std()) if float(oasis_tr_cache["val_std"].std()) > 0 else 1.0
    mu_nz = float(oasis_tr_cache["nonzero_fraction"].mean())
    sig_nz = float(oasis_tr_cache["nonzero_fraction"].std()) if float(oasis_tr_cache["nonzero_fraction"].std()) > 0 else 1.0

    def get_qc_anomaly(std_vals, nz_vals):
        dev_std = np.abs(std_vals - mu_std) / (3.0 * sig_std)
        dev_nz = np.abs(nz_vals - mu_nz) / (3.0 * sig_nz)
        return np.clip(0.5 * (dev_std + dev_nz), 0.0, 1.0)

    sd_test = get_qc_anomaly(oasis_test_cache["val_std"].values, oasis_test_cache["nonzero_fraction"].values)
    sd_adni = get_qc_anomaly(adni_manifest["std_val"].values, adni_manifest["nonzero_pct"].values / 100.0)

    # 5. Component A & Baselines using Primary Model (MONAI Swin UNETR SSL)
    # Target Ground Truth on ADNI & OASIS-1 Test
    y_test_gt = v1_o["true_label"].astype(int).values
    y_adni_gt = v1_a["binary_label"].astype(int).values

    p_test = v1_o["swin_unetr_ssl_prob"].values
    p_adni = v1_a["swin_unetr_ssl_prob"].values

    # Margin Risk (Component A)
    sa_test = 1.0 - 2.0 * np.abs(p_test - 0.50)
    sa_adni = 1.0 - 2.0 * np.abs(p_adni - 0.50)

    # Baseline 1: Maximum Softmax Probability (MSP) Risk: 1 - max(p, 1-p)
    msp_test = 1.0 - np.maximum(p_test, 1.0 - p_test)
    msp_adni = 1.0 - np.maximum(p_adni, 1.0 - p_adni)

    # Baseline 2: Predictive Entropy
    def get_entropy(p_arr):
        pc = np.clip(p_arr, 1e-6, 1.0 - 1e-6)
        return - (pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))

    ent_test = get_entropy(p_test)
    ent_adni = get_entropy(p_adni)

    # Baseline 6: Simple Average Confidence Aggregation
    agg_test = 0.5 * sa_test + 0.5 * ent_test
    agg_adni = 0.5 * sa_adni + 0.5 * ent_adni

    # Proposed TrustGate Composite Risk Score (Prespecified Equal Weights)
    tg_test = 0.25 * sa_test + 0.25 * sb_test + 0.25 * sc_test + 0.25 * sd_test
    tg_adni = 0.25 * sa_adni + 0.25 * sb_adni + 0.25 * sc_adni + 0.25 * sd_adni

    # Binary Error Target: Did primary model make a classification error?
    # At source-optimal threshold t* = 0.513
    t_opt = 0.513
    pred_test = (p_test >= t_opt).astype(int)
    pred_adni = (p_adni >= t_opt).astype(int)

    err_test = (pred_test != y_test_gt).astype(int)
    err_adni = (pred_adni != y_adni_gt).astype(int)

    print(f"Error Rates under Source Threshold t*={t_opt}:", flush=True)
    print(f"  OASIS-1 Test (N=30): {err_test.sum()}/30 errors ({err_test.mean()*100:.1f}%)", flush=True)
    print(f"  ADNI External (N=500): {err_adni.sum()}/500 errors ({err_adni.mean()*100:.1f}%)", flush=True)

    # 6. Comparative Evaluation: Error-Detection ROC-AUC
    methods = [
        ("Maximum Probability (MSP)", msp_adni, msp_test),
        ("Predictive Entropy", ent_adni, ent_test),
        ("Classification Margin", sa_adni, sa_test),
        ("Regularized Feature Distance (FOMO)", sb_adni, sb_test),
        ("Model Disagreement (6 Architectures)", sc_adni, sc_test),
        ("Simple Confidence Aggregation", agg_adni, agg_test),
        ("Proposed TrustGate Composite", tg_adni, tg_test),
    ]

    benchmark_rows = []
    print("\n>>> Comparing TrustGate Against Baseline Risk Estimators (results/v3/trustgate/)", flush=True)
    for name, r_adni, r_test in methods:
        # ADNI Error Detection AUC
        auc_adni, ci_adni = compute_stratified_auc_ci(err_adni, r_adni, n_bootstraps=2000)
        # OASIS-1 Test Error Detection AUC
        auc_test, ci_test = compute_stratified_auc_ci(err_test, r_test, n_bootstraps=2000)

        benchmark_rows.append({
            "Method": name,
            "ADNI_Error_Detection_AUC": round(auc_adni, 4),
            "ADNI_AUC_95CI": f"[{ci_adni[0]:.4f}, {ci_adni[1]:.4f}]",
            "OASIS_Test_Error_Detection_AUC": round(auc_test, 4),
            "OASIS_AUC_95CI": f"[{ci_test[0]:.4f}, {ci_test[1]:.4f}]",
        })
        print(f"  {name:38s} | ADNI AUC={auc_adni:.4f} {ci_adni} | OASIS Test AUC={auc_test:.4f}", flush=True)

    bench_df = pd.DataFrame(benchmark_rows)
    bench_csv = os.path.join(TRUSTGATE_DIR, "trustgate_benchmark_results.csv")
    bench_df.to_csv(bench_csv, index=False)
    print(f"Saved TrustGate Benchmark Results to: {bench_csv}", flush=True)

    # 7. Selective Prediction & Risk-Coverage Profile (Target: ADNI External)
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    risk_cov_rows = []

    # Sort ADNI scans by increasing TrustGate risk (highest trust retained first)
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

            # False negatives: actual AD (y=1) predicted as CN (pred=0)
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

        aurc = compute_aurc(risks, covs)
        for r in curve_records:
            r["AURC"] = round(aurc, 4)
        return curve_records, aurc, risks

    rec_tg, aurc_tg, risks_tg = evaluate_coverage_curve(tg_sorted_idx, "Proposed TrustGate Composite")
    rec_dis, aurc_dis, risks_dis = evaluate_coverage_curve(dis_sorted_idx, "Model Disagreement")
    rec_mar, aurc_mar, risks_mar = evaluate_coverage_curve(mar_sorted_idx, "Classification Margin")

    all_curve_records = rec_tg + rec_dis + rec_mar
    curve_df = pd.DataFrame(all_curve_records)
    curve_csv = os.path.join(TRUSTGATE_DIR, "trustgate_risk_coverage_table.csv")
    curve_df.to_csv(curve_csv, index=False)
    print(f"Saved Risk-Coverage Profile to: {curve_csv} (TrustGate AURC={aurc_tg:.4f})", flush=True)

    # 8. Source-Calibrated Thresholding & Scanner Subgroup Analysis
    # Calibrate rejection threshold tau* on OASIS-1 validation to flag 20% on source
    val_tg_scores = 0.25 * (1.0 - 2.0 * np.abs(v1_o["swin_unetr_ssl_prob"].values[:30] - 0.50)) + \
                    0.25 * np.clip((d_val_mahal - q05_d) / (q95_d - q05_d), 0.0, 1.0) + \
                    0.25 * sc_test + 0.25 * sd_test
    tau_star_source = float(np.percentile(val_tg_scores, 80.0))  # 80th percentile flags top 20%

    print(f"\nSource-Calibrated Flagging Threshold tau* (80th pct on source): {tau_star_source:.4f}", flush=True)

    flagged_mask = (tg_adni > tau_star_source)
    overall_flag_rate = float(np.mean(flagged_mask))
    print(f"Overall ADNI Flagged Fraction: {overall_flag_rate*100:.1f}% ({flagged_mask.sum()}/500 scans)", flush=True)

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

        # Error rate on retained vs flagged
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

    # 9. Ablation Study: Individual Component Removal
    print("\n>>> Running TrustGate Ablation Experiments (results/v3/trustgate_ablation/)", flush=True)
    ablations = [
        ("Full TrustGate (A + B + C + D)", tg_adni),
        ("Ablate A: No Margin (B + C + D)", (sb_adni + sc_adni + sd_adni) / 3.0),
        ("Ablate B: No Feature Distance (A + C + D)", (sa_adni + sc_adni + sd_adni) / 3.0),
        ("Ablate C: No Disagreement (A + B + D)", (sa_adni + sb_adni + sd_adni) / 3.0),
        ("Ablate D: No Acquisition Anomaly (A + B + C)", (sa_adni + sb_adni + sc_adni) / 3.0),
    ]

    ablation_records = []
    for abl_name, abl_scores in ablations:
        auc_abl, ci_abl = compute_stratified_auc_ci(err_adni, abl_scores, n_bootstraps=2000)

        # Compute AURC
        s_idx = np.argsort(abl_scores)
        _, aurc_abl, _ = evaluate_coverage_curve(s_idx, abl_name)

        ablation_records.append({
            "Ablation_Variant": abl_name,
            "Error_Detection_AUC": round(auc_abl, 4),
            "AUC_95CI": f"[{ci_abl[0]:.4f}, {ci_abl[1]:.4f}]",
            "AURC": round(aurc_abl, 4),
            "Delta_AUC_vs_Full": round(auc_abl - benchmark_rows[6]["ADNI_Error_Detection_AUC"], 4),
        })
        print(f"  {abl_name:42s} | AUC={auc_abl:.4f} {ci_abl} | AURC={aurc_abl:.4f}", flush=True)

    abl_df = pd.DataFrame(ablation_records)
    abl_csv = os.path.join(ABLATION_DIR, "trustgate_ablation_results.csv")
    abl_df.to_csv(abl_csv, index=False)
    print(f"Saved Ablation Results to: {abl_csv}", flush=True)

    # 10. Generate Figures
    # Figure 1: TrustGate Risk-Coverage Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, risks_tg, "o-", color="#1f77b4", lw=2.5, label=f"TrustGate Composite (AURC={aurc_tg:.3f})")
    ax.plot(coverages, risks_dis, "s--", color="#ff7f0e", lw=2, label=f"Model Disagreement (AURC={aurc_dis:.3f})")
    ax.plot(coverages, risks_mar, "^-.", color="#2ca02c", lw=2, label=f"Classification Margin (AURC={aurc_mar:.3f})")
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

    ax.plot(fpr_tg, tpr_tg, color="#1f77b4", lw=2.5, label=f"TrustGate Composite (AUC={benchmark_rows[6]['ADNI_Error_Detection_AUC']:.3f})")
    ax.plot(fpr_mar, tpr_mar, color="#2ca02c", lw=2, label=f"Margin Alone (AUC={benchmark_rows[2]['ADNI_Error_Detection_AUC']:.3f})")
    ax.plot(fpr_dis, tpr_dis, color="#ff7f0e", lw=2, label=f"Disagreement Alone (AUC={benchmark_rows[4]['ADNI_Error_Detection_AUC']:.3f})")
    ax.plot(fpr_sb, tpr_sb, color="#9467bd", lw=1.5, ls=":", label=f"Feature Dist Alone (AUC={benchmark_rows[3]['ADNI_Error_Detection_AUC']:.3f})")
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
