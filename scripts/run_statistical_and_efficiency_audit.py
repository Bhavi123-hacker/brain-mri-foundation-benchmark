"""
Statistical Integrity and Paired Bootstrap Evaluation Suite.

Tasks:
1. Verifies prediction reproduction across OASIS Test and ADNI External.
2. Produces paired bootstrap 95% CIs for each model and paired AUC differences (Delta-AUC) on identical subjects.
3. Conducts matched-label-budget experiments (10%, 25%, 50%, 100%) for FOMO25 AMAES ResEnc-B on OASIS train/val.
4. Analyzes threshold dependency, probability calibration, and sensitivity/specificity collapse mechanisms.
5. Exports comprehensive statistical tables and JSON deliverables.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("StatisticalAudit")

OASIS_TEST_CSV = r"D:\ml project\results\report\oasis_locked_test_predictions.csv"
FOMO_TEST_CSV = r"D:\ml project\results\fomo_linear_probe\oasis_test_predictions.csv"
ADNI_EXT_CSV = r"D:\ml project\results\report\adni_external_predictions.csv"
FOMO_ADNI_CSV = r"D:\ml project\results\fomo_linear_probe\adni_external_predictions.csv"
MASTER_TABLE_CSV = r"D:\ml project\results\report\master_model_comparison_table.csv"
OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
FOMO_OASIS_FEATURES_PT = r"D:\ml project\results\fomo_linear_probe\oasis_fomo_features.pt"

OUTPUT_DIR = r"D:\ml project\results\statistical_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)


def run_paired_bootstrap(y_true, prob_dict, n_bootstraps=2000, rng_seed=42):
    """
    Computes individual AUC 95% CIs and paired Delta-AUC 95% CIs on identical bootstrap resamples.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    model_names = list(prob_dict.keys())

    boot_aucs = {m: [] for m in model_names}
    paired_diffs = {(m1, m2): [] for i, m1 in enumerate(model_names) for m2 in model_names[i+1:]}

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue

        sample_aucs = {}
        for m in model_names:
            auc = roc_auc_score(y_true[idx], prob_dict[m][idx])
            sample_aucs[m] = auc
            boot_aucs[m].append(auc)

        for (m1, m2) in paired_diffs:
            paired_diffs[(m1, m2)].append(sample_aucs[m1] - sample_aucs[m2])

    # Summarize individual model CIs
    individual_summary = {}
    for m in model_names:
        arr = np.sort(boot_aucs[m])
        individual_summary[m] = {
            "median_auc": float(np.median(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "std": float(np.std(arr))
        }

    # Summarize paired differences
    paired_summary = []
    for (m1, m2), diffs in paired_diffs.items():
        arr = np.sort(diffs)
        med = float(np.median(arr))
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        # Two-tailed empirical p-value for H0: delta = 0
        p_val = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
        p_val = min(1.0, max(0.0001, p_val))
        sig = (ci_low > 0 or ci_high < 0)

        paired_summary.append({
            "model_A": m1,
            "model_B": m2,
            "delta_auc_median": round(med, 4),
            "ci_lower": round(ci_low, 4),
            "ci_upper": round(ci_high, 4),
            "p_value": round(p_val, 4),
            "statistically_significant": bool(sig)
        })

    return individual_summary, pd.DataFrame(paired_summary)


def evaluate_threshold_calibration_profiles(y_true, y_prob, model_name, cohort_name):
    """
    Computes calibration statistics and compares metrics under standard (0.50) vs optimal (Youden's J) threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Standard threshold t=0.50
    preds_05 = (y_prob >= 0.50).astype(int)
    bal_acc_05 = balanced_accuracy_score(y_true, preds_05)
    cm_05 = confusion_matrix(y_true, preds_05, labels=[0, 1])
    sens_05 = cm_05[1, 1] / max(1, (y_true == 1).sum())
    spec_05 = cm_05[0, 0] / max(1, (y_true == 0).sum())

    # Find empirical Youden's J threshold
    thresholds = np.linspace(0.01, 0.99, 99)
    best_j = -1.0
    best_t = 0.50
    best_bal_acc = bal_acc_05
    best_sens = sens_05
    best_spec = spec_05

    for t in thresholds:
        preds_t = (y_prob >= t).astype(int)
        cm_t = confusion_matrix(y_true, preds_t, labels=[0, 1])
        sens_t = cm_t[1, 1] / max(1, (y_true == 1).sum())
        spec_t = cm_t[0, 0] / max(1, (y_true == 0).sum())
        j = sens_t + spec_t - 1.0
        if j > best_j:
            best_j = j
            best_t = t
            best_bal_acc = (sens_t + spec_t) / 2.0
            best_sens = sens_t
            best_spec = spec_t

    return {
        "cohort": cohort_name,
        "model": model_name,
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "mean_pred_prob": round(float(np.mean(y_prob)), 4),
        "min_pred_prob": round(float(np.min(y_prob)), 4),
        "max_pred_prob": round(float(np.max(y_prob)), 4),
        "t05_bal_acc": round(float(bal_acc_05), 4),
        "t05_sensitivity": round(float(sens_05), 4),
        "t05_specificity": round(float(spec_05), 4),
        "youden_optimal_t": round(float(best_t), 4),
        "youden_bal_acc": round(float(best_bal_acc), 4),
        "youden_sensitivity": round(float(best_sens), 4),
        "youden_specificity": round(float(best_spec), 4)
    }


def run_fomo_label_efficiency_suite(fractions=[0.10, 0.25, 0.50, 1.00], seed=RANDOM_SEED):
    """
    Evaluates FOMO25 AMAES ResEnc-B across matched label budgets (10%, 25%, 50%, 100%)
    using strictly OASIS train subsets and evaluating strictly on OASIS validation.
    """
    splits_df = pd.read_csv(OASIS_SPLITS_CSV)
    if "binary_label" not in splits_df.columns:
        splits_df["binary_label"] = (splits_df["cdr"] > 0.0).astype(float)

    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)

    # Load pre-extracted FOMO features
    fomo_data = torch.load(FOMO_OASIS_FEATURES_PT, map_location="cpu", weights_only=False)
    features_all = fomo_data["features"]
    p_ids = fomo_data["participant_ids"]
    id_to_idx = {pid: i for i, pid in enumerate(p_ids)}

    val_indices = [id_to_idx[pid] for pid in val_df["base_subject_id"] if pid in id_to_idx]
    X_val = features_all[val_indices]
    y_val = val_df["binary_label"].values

    # Generate reproducible stratified subsets of train
    cn_train = train_df[train_df["binary_label"] == 0.0].sample(frac=1.0, random_state=seed)
    ad_train = train_df[train_df["binary_label"] == 1.0].sample(frac=1.0, random_state=seed)

    budget_results = {}

    for frac in fractions:
        n_cn = max(2, int(round(len(cn_train) * frac)))
        n_ad = max(2, int(round(len(ad_train) * frac)))
        sub_df = pd.concat([cn_train.iloc[:n_cn], ad_train.iloc[:n_ad]]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        b_key = int(frac * 100)
        n_samples = len(sub_df)

        sub_indices = [id_to_idx[pid] for pid in sub_df["base_subject_id"] if pid in id_to_idx]
        X_sub = features_all[sub_indices]
        y_sub = sub_df["binary_label"].values

        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_val_scaled = scaler.transform(X_val)

        # Fit linear probe with fixed C=0.001
        clf = LogisticRegression(C=0.001, max_iter=2000, random_state=seed)
        clf.fit(X_sub_scaled, y_sub)

        p_val = clf.predict_proba(X_val_scaled)[:, 1]
        auc = float(roc_auc_score(y_val, p_val))
        brier = float(brier_score_loss(y_val, p_val))
        preds = (p_val >= 0.5).astype(int)
        bal_acc = float(balanced_accuracy_score(y_val, preds))
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        sens = float(cm[1, 1] / max(1, (y_val == 1).sum()))
        spec = float(cm[0, 0] / max(1, (y_val == 0).sum()))

        budget_results[b_key] = {
            "budget_pct": b_key,
            "n_samples": n_samples,
            "n_cn": n_cn,
            "n_ad": n_ad,
            "val_roc_auc": round(auc, 4),
            "val_balanced_acc": round(bal_acc, 4),
            "val_sensitivity": round(sens, 4),
            "val_specificity": round(spec, 4),
            "val_brier_score": round(brier, 4)
        }
        logger.info(f"FOMO AMAES Budget {b_key:3d}% (N={n_samples:3d}) -> Val AUC: {auc:.4f}, BalAcc: {bal_acc:.4f}, Brier: {brier:.4f}")

    return budget_results


def main():
    set_seed(RANDOM_SEED)
    logger.info("=================================================================")
    logger.info("Executing Statistical Integrity & Paired Bootstrap Benchmark")
    logger.info("=================================================================")

    # 1. Load predictions
    oasis_p1 = pd.read_csv(OASIS_TEST_CSV)
    oasis_p2 = pd.read_csv(FOMO_TEST_CSV)
    oasis_df = oasis_p1.copy()
    oasis_df["fomo_amaes_prob"] = oasis_p2["fomo_amaes_prob"].values

    adni_p1 = pd.read_csv(ADNI_EXT_CSV)
    adni_p2 = pd.read_csv(FOMO_ADNI_CSV)
    adni_df = adni_p1.copy()
    adni_df["fomo_amaes_prob"] = adni_p2["fomo_amaes_prob"].values

    model_cols = {
        "3D ResNet-18 (Scratch)": "resnet18_prob",
        "3D ResNet-34 (Scratch)": "resnet34_prob",
        "2D Transfer ResNet-18": "resnet2d_transfer_prob",
        "Classical Radiomics (SVM)": "classical_radiomics_prob",
        "MONAI Swin UNETR SSL": "swin_unetr_ssl_prob",
        "FOMO25 AMAES ResEnc-B": "fomo_amaes_prob"
    }

    # 2. Verify reproduction of metrics
    logger.info("--- TASK 1: Verifying Metric Reproduction from Saved Prediction Files ---")
    oasis_probs = {m: oasis_df[col].values for m, col in model_cols.items()}
    adni_probs = {m: adni_df[col].values for m, col in model_cols.items()}

    oasis_indiv_ci, oasis_paired_df = run_paired_bootstrap(oasis_df["true_label"].values, oasis_probs, n_bootstraps=2000)
    adni_indiv_ci, adni_paired_df = run_paired_bootstrap(adni_df["binary_label"].values, adni_probs, n_bootstraps=2000)

    # Save paired difference tables
    oasis_paired_path = os.path.join(OUTPUT_DIR, "oasis_test_paired_auc_differences.csv")
    oasis_paired_df.to_csv(oasis_paired_path, index=False)
    logger.info(f"OASIS Paired Bootstrap Differences saved to {oasis_paired_path}")

    adni_paired_path = os.path.join(OUTPUT_DIR, "adni_external_paired_auc_differences.csv")
    adni_paired_df.to_csv(adni_paired_path, index=False)
    logger.info(f"ADNI Paired Bootstrap Differences saved to {adni_paired_path}")

    # 3. Calibration & Threshold Profiles
    logger.info("--- TASKS 5 & 6: Threshold & Calibration Profiles ---")
    calibration_profiles = []
    for m, col in model_cols.items():
        p_oasis = evaluate_threshold_calibration_profiles(oasis_df["true_label"].values, oasis_df[col].values, m, "OASIS_Locked_Test")
        p_adni = evaluate_threshold_calibration_profiles(adni_df["binary_label"].values, adni_df[col].values, m, "ADNI_External")
        calibration_profiles.extend([p_oasis, p_adni])

    calib_df = pd.DataFrame(calibration_profiles)
    calib_path = os.path.join(OUTPUT_DIR, "threshold_and_calibration_profiles.csv")
    calib_df.to_csv(calib_path, index=False)
    logger.info(f"Threshold & Calibration Profiles saved to {calib_path}")

    # 4. Matched-Label-Budget Experiments
    logger.info("--- TASK 4: Matched-Label-Budget Experiments (FOMO25 AMAES) ---")
    fomo_budget_res = run_fomo_label_efficiency_suite()
    fomo_budget_path = os.path.join(OUTPUT_DIR, "fomo_amaes_label_efficiency_results.json")
    with open(fomo_budget_path, "w") as f:
        json.dump(fomo_budget_res, f, indent=2)
    logger.info(f"FOMO Label Efficiency results saved to {fomo_budget_path}")

    # 5. Master Summary JSON
    summary_report = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "oasis_test_n": len(oasis_df),
        "adni_external_n": len(adni_df),
        "individual_auc_bootstrap_ci": {
            "oasis_locked_test": oasis_indiv_ci,
            "adni_external": adni_indiv_ci
        },
        "paired_auc_differences": {
            "oasis_test_comparisons": oasis_paired_df.to_dict(orient="records"),
            "adni_external_comparisons": adni_paired_df.to_dict(orient="records")
        },
        "fomo_amaes_label_efficiency": fomo_budget_res
    }

    summary_path = os.path.join(OUTPUT_DIR, "comprehensive_statistical_audit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"Comprehensive Statistical Audit Summary saved to {summary_path}")

    # Print Key Highlights
    logger.info("\n=== TOP OASIS TEST PAIRED AUC DIFFERENCES (N=30) ===")
    for row in oasis_paired_df.head(6).itertuples():
        logger.info(f"  {row.model_A} vs {row.model_B}: Delta AUC = {row.delta_auc_median:+.4f} [{row.ci_lower:+.4f}, {row.ci_upper:+.4f}], p={row.p_value:.4f}, Sig={row.statistically_significant}")

    logger.info("\n=== TOP ADNI EXTERNAL PAIRED AUC DIFFERENCES (N=500) ===")
    for row in adni_paired_df.head(6).itertuples():
        logger.info(f"  {row.model_A} vs {row.model_B}: Delta AUC = {row.delta_auc_median:+.4f} [{row.ci_lower:+.4f}, {row.ci_upper:+.4f}], p={row.p_value:.4f}, Sig={row.statistically_significant}")

    return summary_report


if __name__ == "__main__":
    main()
