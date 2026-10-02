"""
Generates Figures DE1 - DE5 for the Priority 1 Data-Efficiency Benchmark.
Saves to results/extensions/figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_de_figures(
    results_csv: str = "results/extensions/data_efficiency/oasis_data_efficiency_results.csv",
    output_dir: str = "results/extensions/figures",
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    models = ["Supervised 3D ResNet-18", "BrainIAC", "FOMO25 Winner", "MONAI Swin UNETR SSL"]
    colors = {
        "Supervised 3D ResNet-18": "#1f77b4",
        "BrainIAC": "#ff7f0e",
        "FOMO25 Winner": "#2ca02c",
        "MONAI Swin UNETR SSL": "#d62728",
    }
    markers = {
        "Supervised 3D ResNet-18": "s",
        "BrainIAC": "o",
        "FOMO25 Winner": "^",
        "MONAI Swin UNETR SSL": "D",
    }
    frac_order = ["10%", "25%", "50%", "100%"]
    frac_nums = [10, 25, 50, 100]

    # -------------------------------------------------------------------------
    # FIGURE DE1: Overall Performance (AUC, Acc, F1) vs Training Fraction
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    metrics = [("test_roc_auc", "ROC-AUC", axes[0]), ("test_accuracy", "Accuracy", axes[1]), ("test_f1", "F1-Score", axes[2])]

    for metric_col, title, ax in metrics:
        for m in models:
            sub = df[df["model"] == m]
            means = [sub[sub["fraction"] == f][metric_col].mean() for f in frac_order]
            stds = [sub[sub["fraction"] == f][metric_col].std() for f in frac_order]
            ax.errorbar(frac_nums, means, yerr=stds, label=m, color=colors[m], marker=markers[m], capsize=4, lw=2)
        ax.set_title(f"{title} vs Training Data Fraction", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Fraction (%)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_xticks(frac_nums)
        ax.set_xticklabels(["10% (N=4)", "25% (N=9)", "50% (N=18)", "100% (N=35)"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)

    axes[0].legend(fontsize=9, loc="lower right")
    fig1_path = os.path.join(output_dir, "figure_DE1_performance_vs_fraction.png")
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved Figure DE1 to {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURE DE2: Test ROC-AUC vs Training-Data Fraction (Detailed)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for m in models:
        sub = df[df["model"] == m]
        means = [sub[sub["fraction"] == f]["test_roc_auc"].mean() for f in frac_order]
        stds = [sub[sub["fraction"] == f]["test_roc_auc"].std() for f in frac_order]
        ax.plot(frac_nums, means, label=f"{m} (10%->100%: {means[0]:.2f}->{means[-1]:.2f})", color=colors[m], marker=markers[m], lw=2.5, markersize=8)
        ax.fill_between(frac_nums, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=colors[m], alpha=0.15)

    ax.axhline(0.5, color="gray", linestyle=":", label="Chance (0.50)")
    ax.set_title("Test ROC-AUC Scaling across Labeled Data Fractions (OASIS-1)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Data Fraction", fontsize=11)
    ax.set_ylabel("Test ROC-AUC (Mean ± SD across 5 Seeds)", fontsize=11)
    ax.set_xticks(frac_nums)
    ax.set_xticklabels(["10% (N=4)", "25% (N=9)", "50% (N=18)", "100% (N=35)"])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    fig2_path = os.path.join(output_dir, "figure_DE2_roc_auc_vs_fraction.png")
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved Figure DE2 to {fig2_path}")

    # -------------------------------------------------------------------------
    # FIGURE DE3: Test F1 vs Training-Data Fraction
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for m in models:
        sub = df[df["model"] == m]
        means = [sub[sub["fraction"] == f]["test_f1"].mean() for f in frac_order]
        stds = [sub[sub["fraction"] == f]["test_f1"].std() for f in frac_order]
        ax.errorbar(frac_nums, means, yerr=stds, label=m, color=colors[m], marker=markers[m], capsize=5, lw=2.2, markersize=8)

    ax.set_title("Test F1-Score Scaling across Labeled Data Fractions (OASIS-1)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Data Fraction", fontsize=11)
    ax.set_ylabel("Test F1-Score (Mean ± SD across 5 Seeds)", fontsize=11)
    ax.set_xticks(frac_nums)
    ax.set_xticklabels(["10% (N=4)", "25% (N=9)", "50% (N=18)", "100% (N=35)"])
    ax.set_ylim(-0.05, 0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)

    fig3_path = os.path.join(output_dir, "figure_DE3_f1_vs_fraction.png")
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved Figure DE3 to {fig3_path}")

    # -------------------------------------------------------------------------
    # FIGURE DE4: Parameter Efficiency vs Performance across Fractions
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for frac, style in zip(["10%", "100%"], ["--", "-"]):
        for m in models:
            sub = df[(df["model"] == m) & (df["fraction"] == frac)]
            n_params = sub["trainable_parameters"].iloc[0]
            auc_mean = sub["test_roc_auc"].mean()
            auc_std = sub["test_roc_auc"].std()
            lbl = f"{m} ({frac})" if style == "-" else None
            ax.errorbar(n_params, auc_mean, yerr=auc_std, fmt=markers[m], color=colors[m], markersize=10, capsize=5, label=lbl)

    ax.set_xscale("log")
    ax.axhline(0.5, color="gray", linestyle=":", label="Chance (0.50)")
    ax.set_title("Trainable Parameters vs Test ROC-AUC (10% vs 100% Data)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trainable Parameters (Log Scale)", fontsize=11)
    ax.set_ylabel("Test ROC-AUC", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)

    fig4_path = os.path.join(output_dir, "figure_DE4_parameter_efficiency_vs_performance.png")
    plt.tight_layout()
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved Figure DE4 to {fig4_path}")

    # -------------------------------------------------------------------------
    # FIGURE DE5: Performance Variability Across Random Seeds (Boxplot)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    data_by_group = []
    labels = []
    box_colors = []

    for frac in ["10%", "100%"]:
        for m in models:
            vals = df[(df["model"] == m) & (df["fraction"] == frac)]["test_roc_auc"].values
            data_by_group.append(vals)
            labels.append(f"{m.split()[0]}\n({frac})")
            box_colors.append(colors[m])

    bplot = ax.boxplot(data_by_group, patch_artist=True, tick_labels=labels)
    for patch, c in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for median in bplot["medians"]:
        median.set(color="black", linewidth=2)

    ax.set_title("Cross-Seed Performance Variability (10% vs 100% Training Data)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Test ROC-AUC across 5 Seeds", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", label="Chance (0.50)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig5_path = os.path.join(output_dir, "figure_DE5_performance_variability_across_seeds.png")
    plt.tight_layout()
    plt.savefig(fig5_path)
    plt.close()
    print(f"Saved Figure DE5 to {fig5_path}")

if __name__ == "__main__":
    generate_de_figures()
