"""
Standardized Medical Imaging Evaluation Metrics.
Supports Binary Classification and Continuous Regression with exact statistical definitions.
"""

import numpy as np
import torch
from typing import Dict, Any, Union
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate full suite of clinical classification metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().flatten()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy().flatten()

    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    # Edge cases: single class in targets
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix for Sensitivity and Specificity
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "brier_score": round(brier, 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Calculate continuous regression metrics (Brain-Age).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().flatten()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().flatten()

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # R2 Score
    if len(y_true) > 1 and np.var(y_true) > 1e-8:
        r2 = float(r2_score(y_true, y_pred))
        pr, _ = pearsonr(y_true, y_pred)
        sr, _ = spearmanr(y_true, y_pred)
        p_val = float(pr) if not np.isnan(pr) else 0.0
        s_val = float(sr) if not np.isnan(sr) else 0.0
    else:
        r2 = 0.0
        p_val = 0.0
        s_val = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "pearson_r": round(p_val, 4),
        "spearman_rho": round(s_val, 4),
    }
