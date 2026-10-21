"""
evaluation/v3/shift_mri.py
Core analytical and statistical library for the SHIFT-MRI benchmark:
Cross-Cohort Reliability Matrix, Collapse Analysis, Demographic Baselines,
Confidence-Aware Abstention, and Perturbation Robustness.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit


def compute_stratified_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 2000, seed: int = 42
) -> Tuple[float, List[float]]:
    """Compute empirical mean ROC-AUC and 95% stratified bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_aucs = []
    for _ in range(n_bootstraps):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        try:
            auc = roc_auc_score(y_true[b_idx], y_prob[b_idx])
            boot_aucs.append(auc)
        except ValueError:
            continue

    if not boot_aucs:
        base_auc = roc_auc_score(y_true, y_prob)
        return float(base_auc), [float(base_auc), float(base_auc)]

    boot_aucs = np.array(boot_aucs)
    return float(np.mean(boot_aucs)), [
        float(np.percentile(boot_aucs, 2.5)),
        float(np.percentile(boot_aucs, 97.5)),
    ]


def compute_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE), calibration slope, and calibration intercept.
    - ECE: Expected Calibration Error over uniform bins.
    - Calibration Slope & Intercept: Logit recalibration model logit(p_recal) = a + b * logit(p).
      Perfect calibration corresponds to slope=1.0 and intercept=0.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)

    # 1. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        b_min = bin_boundaries[i]
        b_max = bin_boundaries[i + 1]
        mask = (y_prob >= b_min) & (y_prob < b_max if i < n_bins - 1 else y_prob <= b_max)
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (n_in_bin / len(y_prob)) * abs(bin_acc - bin_conf)

    # 2. Logistic Calibration Slope & Intercept
    log_odds = logit(y_prob_clipped).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
        lr.fit(log_odds, y_true)
        cal_slope = float(lr.coef_[0, 0])
        cal_intercept = float(lr.intercept_[0])
    except Exception:
        cal_slope = float("nan")
        cal_intercept = float("nan")

    return {
        "ece": float(ece),
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
    }


def find_optimal_youden_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Find threshold t* maximizing Youden's J = TPR + TNR - 1 = Sensitivity + Specificity - 1.
    Returns (optimal_threshold, max_youden_j).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    # Handle infinite threshold boundary from roc_curve
    if np.isinf(best_thresh) or best_thresh > 1.0:
        best_thresh = 0.50
    return float(best_thresh), float(j_scores[best_idx])


def compute_reliability_matrix_row(
    cohort_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    source_optimal_threshold: float = 0.50,
    source_auc: Optional[float] = None,
    source_brier: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for one row of the Cross-Cohort Reliability Matrix (N01).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_total = len(y_true)
    prevalence = float(np.mean(y_true))

    # Discrimination
    auc_mean, auc_ci = compute_stratified_auc_ci(y_true, y_prob, n_bootstraps=2000)
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Threshold t = 0.50
    pred_05 = (y_prob >= 0.50).astype(int)
    bal_acc_05 = float(balanced_accuracy_score(y_true, pred_05))
    cm_05 = confusion_matrix(y_true, pred_05, labels=[0, 1])
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()
    sens_05 = float(tp_05 / (tp_05 + fn_05)) if (tp_05 + fn_05) > 0 else 0.0
    spec_05 = float(tn_05 / (tn_05 + fp_05)) if (tn_05 + fp_05) > 0 else 0.0
    pos_rate_05 = float(np.mean(pred_05))

    # Source-locked optimal threshold t*
    pred_star = (y_prob >= source_optimal_threshold).astype(int)
    bal_acc_star = float(balanced_accuracy_score(y_true, pred_star))
    cm_star = confusion_matrix(y_true, pred_star, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_star.ravel()
    sens_star = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else 0.0
    spec_star = float(tn_s / (tn_s + fp_s)) if (tn_s + fp_s) > 0 else 0.0
    pos_rate_star = float(np.mean(pred_star))

    # Calibration
    brier = float(brier_score_loss(y_true, y_prob))
    calib = compute_calibration_metrics(y_true, y_prob, n_bins=10)

    # Generalization degradation
    delta_auc = float(source_auc - auc_mean) if source_auc is not None else 0.0
    delta_brier = float(brier - source_brier) if source_brier is not None else 0.0

    # Decision collapse indicator
    is_collapsed_05 = bool(pos_rate_05 <= 0.05 or pos_rate_05 >= 0.95)

    return {
        "Cohort": cohort_name,
        "Model": model_name,
        "N": n_total,
        "Prevalence": round(prevalence, 4),
        "ROC_AUC": round(auc_mean, 4),
        "ROC_AUC_95CI": f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
        "PR_AUC": round(pr_auc, 4),
        "BalAcc_t05": round(bal_acc_05, 4),
        "Sensitivity_t05": round(sens_05, 4),
        "Specificity_t05": round(spec_05, 4),
        "PosRate_t05": round(pos_rate_05, 4),
        "Collapse_t05": is_collapsed_05,
        "Source_Optimal_t": round(source_optimal_threshold, 4),
        "BalAcc_t_star": round(bal_acc_star, 4),
        "Sensitivity_t_star": round(sens_star, 4),
        "Specificity_t_star": round(spec_star, 4),
        "PosRate_t_star": round(pos_rate_star, 4),
        "Brier_Score": round(brier, 4),
        "ECE": round(calib["ece"], 4),
        "Calibration_Slope": round(calib["calibration_slope"], 4),
        "Calibration_Intercept": round(calib["calibration_intercept"], 4),
        "AUC_Generalization_Gap": round(delta_auc, 4),
        "Brier_Degradation": round(delta_brier, 4),
    }


def compute_selective_prediction_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    coverages: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Confidence-Aware Selective Prediction / Abstention Framework (N04).
    Confidence is defined as distance from decision boundary: c(x) = |p(x) - threshold|.
    At coverage kappa, retains top kappa instances by confidence and evaluates selective risk (error rate).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    confidence = np.abs(y_prob - threshold)
    entropy = - (np.clip(y_prob, 1e-6, 1.0 - 1e-6) * np.log2(np.clip(y_prob, 1e-6, 1.0 - 1e-6)) +
                 np.clip(1.0 - y_prob, 1e-6, 1.0 - 1e-6) * np.log2(np.clip(1.0 - y_prob, 1e-6, 1.0 - 1e-6)))

    sorted_indices = np.argsort(-confidence)  # Most confident first

    records = []
    risks = []
    covs = []

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        retained = sorted_indices[:k]
        y_sub = y_true[retained]
        p_sub = y_prob[retained]
        preds_sub = (p_sub >= threshold).astype(int)

        error_rate = float(1.0 - accuracy_score(y_sub, preds_sub))
        bal_acc = float(balanced_accuracy_score(y_sub, preds_sub))
        auc = float(roc_auc_score(y_sub, p_sub)) if len(np.unique(y_sub)) > 1 else float("nan")

        records.append({
            "coverage": cov,
            "retained_n": k,
            "abstain_n": n - k,
            "selective_error_rate": round(error_rate, 4),
            "selective_accuracy": round(1.0 - error_rate, 4),
            "selective_bal_acc": round(bal_acc, 4),
            "selective_auc": round(auc, 4) if not np.isnan(auc) else None,
            "min_confidence": round(float(confidence[retained[-1]]), 4),
        })
        covs.append(cov)
        risks.append(error_rate)

    # Compute Area Under Risk-Coverage Curve (AURC) via robust trapezoidal integration
    risks_arr = np.array(risks)
    covs_arr = np.array(covs)
    aurc = float(np.sum(0.5 * (risks_arr[:-1] + risks_arr[1:]) * np.abs(np.diff(covs_arr))))

    return {
        "curve_records": records,
        "aurc": round(aurc, 4),
    }


def fit_demographic_baselines(
    train_df: pd.DataFrame,
    test_cohorts: Dict[str, pd.DataFrame],
    age_col: str = "age",
    sex_col: str = "sex",
    label_col: str = "label",
) -> List[Dict[str, Any]]:
    """
    Demographic Shortcut Baselines (N03).
    Fits standard Logistic Regression on OASIS-1 train split using:
      1. Age alone
      2. Sex alone
      3. Age + Sex
    Evaluates on test cohorts (OASIS-1 test, ADNI external, OASIS-2 external).
    """
    # Clean train
    tr_df = train_df.dropna(subset=[age_col, sex_col, label_col]).copy()
    tr_y = tr_df[label_col].astype(int).values
    tr_age = tr_df[age_col].astype(float).values
    tr_sex = (tr_df[sex_col].astype(str).str.upper().str.startswith("M")).astype(float).values

    age_mean = float(np.mean(tr_age))
    age_std = float(np.std(tr_age)) if np.std(tr_age) > 0 else 1.0

    tr_age_norm = (tr_age - age_mean) / age_std

    # Define model configurations
    configs = [
        ("Demographic: Age Alone", np.column_stack([tr_age_norm])),
        ("Demographic: Sex Alone", np.column_stack([tr_sex])),
        ("Demographic: Age + Sex", np.column_stack([tr_age_norm, tr_sex])),
    ]

    fitted_models = []
    for name, X_tr in configs:
        clf = LogisticRegression(solver="lbfgs", max_iter=500, penalty=None)
        clf.fit(X_tr, tr_y)
        fitted_models.append((name, clf))

    results = []
    for cohort_name, eval_df in test_cohorts.items():
        ev_df = eval_df.dropna(subset=[age_col, sex_col, label_col]).copy()
        ev_y = ev_df[label_col].astype(int).values
        ev_age = ev_df[age_col].astype(float).values
        ev_sex = (ev_df[sex_col].astype(str).str.upper().str.startswith("M")).astype(float).values
        ev_age_norm = (ev_age - age_mean) / age_std

        eval_features = {
            "Demographic: Age Alone": np.column_stack([ev_age_norm]),
            "Demographic: Sex Alone": np.column_stack([ev_sex]),
            "Demographic: Age + Sex": np.column_stack([ev_age_norm, ev_sex]),
        }

        for name, clf in fitted_models:
            X_ev = eval_features[name]
            p_ev = clf.predict_proba(X_ev)[:, 1]

            auc_mean, auc_ci = compute_stratified_auc_ci(ev_y, p_ev, n_bootstraps=2000)
            preds_05 = (p_ev >= 0.50).astype(int)
            bal_acc = float(balanced_accuracy_score(ev_y, preds_05))
            brier = float(brier_score_loss(ev_y, p_ev))
            calib = compute_calibration_metrics(ev_y, p_ev)

            results.append({
                "Cohort": cohort_name,
                "Model": name,
                "N": len(ev_y),
                "ROC_AUC": round(auc_mean, 4),
                "ROC_AUC_95CI": f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
                "Balanced_Accuracy": round(bal_acc, 4),
                "Brier_Score": round(brier, 4),
                "ECE": round(calib["ece"], 4),
                "Calibration_Slope": round(calib["calibration_slope"], 4),
                "Calibration_Intercept": round(calib["calibration_intercept"], 4),
            })

    return results
