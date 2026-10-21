"""
Generates publication-quality benchmark figures comparing:
1. Multi-paradigm ROC curves (Locked Test Holdout & External ADNI)
2. Label Efficiency Curves across budgets (10%, 25%, 50%, 100%)
3. Preprocessing Robustness & Mask Erosion analysis
4. Subgroup Generalization across Scanner Manufacturers & Field Strengths
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

RUNS_DIR = r"D:\brain_mri_data\benchmark_runs"
FIGURES_DIR = os.path.join(RUNS_DIR, "figures")
TABLES_DIR = os.path.join(RUNS_DIR, "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_multiparadigm_comparison():
    test_preds_path = os.path.join(TABLES_DIR, "oasis_locked_test_predictions.csv")
    adni_preds_path = os.path.join(TABLES_DIR, "adni_external_predictions.csv")

    if not (os.path.exists(test_preds_path) and os.path.exists(adni_preds_path)):
        print("Predictions not ready yet.")
        return

    test_df = pd.read_csv(test_preds_path)
    adni_df = pd.read_csv(adni_preds_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models_info = [
        ("resnet18_prob", "3D ResNet-18 (Scratch)", "#1f77b4", "-"),
        ("resnet34_prob", "3D ResNet-34 (Scratch)", "#aec7e8", "--"),
        ("resnet2d_transfer_prob", "2D Transfer ResNet-18 (ImageNet)", "#ff7f0e", "-"),
        ("classical_radiomics_prob", "Classical Radiomics (L2 LR)", "#2ca02c", "-."),
        ("swin_unetr_ssl_prob", "MONAI Swin UNETR SSL (Linear Probe)", "#9467bd", ":")
    ]

    # Panel A: OASIS-1 Locked Test Holdout (N=30)
    y_test = test_df["true_label"].values if "true_label" in test_df.columns else test_df["binary_label"].values
    ax = axes[0]
    for col, label, color, ls in models_info:
        if col in test_df.columns:
            fpr, tpr, _ = roc_curve(y_test, test_df[col].values)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
    ax.set_title("A. OASIS-1 Locked Test Holdout (N=30, In-Domain)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel B: External ADNI Benchmark (N=500, Out-of-Domain)
    y_adni = (adni_df["group"] == "AD").astype(float).values
    ax = axes[1]
    for col, label, color, ls in models_info:
        if col in adni_df.columns:
            fpr, tpr, _ = roc_curve(y_adni, adni_df[col].values)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
    ax.set_title("B. External ADNI Zero-Shot Generalization (N=500)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "multiparadigm_benchmark_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved multiparadigm comparison figure to {out_path}")


def plot_updated_label_efficiency():
    eff_path = os.path.join(TABLES_DIR, "phase6_multiparadigm_label_efficiency_results.json")
    if not os.path.exists(eff_path):
        print(f"Label efficiency results not found: {eff_path}")
        return

    with open(eff_path, "r") as f:
        data = json.load(f)

    budgets = data["budgets"] # [10, 25, 50, 100]
    paradigms = data["paradigms"]

    fig, ax = plt.subplots(figsize=(8, 6))

    paradigm_labels = {
        "3D_ResNet18": ("3D ResNet-18 (Scratch)", "#1f77b4", "o-"),
        "Classical_Radiomics_LR": ("Classical Radiomics (L2 LR)", "#2ca02c", "s-."),
        "MONAI_SwinUNETR_LinearProbe": ("MONAI Swin UNETR SSL (LP)", "#9467bd", "^:")
    }

    for p_key, (p_name, color, fmt) in paradigm_labels.items():
        if p_key in paradigms and len(paradigms[p_key]) > 0:
            aucs = [paradigms[p_key][str(b)]["roc_auc"] for b in budgets if str(b) in paradigms[p_key]]
            if len(aucs) == len(budgets):
                ax.plot(budgets, aucs, fmt, color=color, linewidth=2, markersize=8, label=p_name)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Chance Level (0.50)")
    ax.set_title("Multi-Paradigm Sample Efficiency Comparison (OASIS-1)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Data Budget (% of N=138)", fontsize=11)
    ax.set_ylabel("Validation ROC-AUC", fontsize=11)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["10%\n(N=14)", "25%\n(N=34)", "50%\n(N=69)", "100%\n(N=138)"])
    ax.set_ylim([0.40, 0.85])
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "label_efficiency_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Updated label efficiency figure saved to {out_path}")


if __name__ == "__main__":
    plot_multiparadigm_comparison()
    plot_updated_label_efficiency()
