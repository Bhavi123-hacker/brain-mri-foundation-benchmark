"""
Publication-Quality Visualization Tools for Medical ML Benchmarks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: str = "results/plots/roc_curve.png"):
    """Plot and save ROC curve."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_label_efficiency_curve(leaderboard_df: pd.DataFrame, metric: str = "ROC_AUC", save_path: str = "results/plots/label_efficiency.png"):
    """
    Plot performance vs label budget (10%, 25%, 50%, 100%) across models.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if leaderboard_df.empty or metric not in leaderboard_df.columns:
        return

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=leaderboard_df,
        x="Label_Budget",
        y=metric,
        hue="Model",
        marker="o",
        linewidth=2.5,
    )
    plt.title(f"Label Efficiency Curve ({metric} vs Budget)")
    plt.xlabel("Training Label Budget")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model Architecture")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str = "results/plots/regression_scatter.png"):
    """
    Plot predicted vs actual continuous values for Brain-Age regression.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color="teal", edgecolors="none")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Identity (Perfect Prediction)")
    plt.xlabel("Chronological Age (Years)")
    plt.ylabel("Predicted Brain-Age (Years)")
    plt.title(f"Brain-Age Scatter — {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
