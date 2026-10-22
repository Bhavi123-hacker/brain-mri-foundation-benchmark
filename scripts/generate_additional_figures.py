"""
scripts/generate_additional_figures.py
Generates Figures 7-10 for the IEEE Access manuscript and Teacher Report:
- Fig 7: Parameter Efficiency vs External Generalization Trade-Off (Pareto curve)
- Fig 8: Confusion Matrices for Primary Architectures on ADNI (N=500)
- Fig 9: Reliability and Calibration Curves (ECE)
- Fig 10: Three-Cohort Multi-Paradigm Benchmark Summary
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path("D:/ml project")
V4_DIR = PROJECT_ROOT / "results" / "v4"
FIGS_DIR = V4_DIR / "figures"
PREDS_DIR = V4_DIR / "predictions"
TABLES_DIR = V4_DIR / "tables"

FIGS_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

df_eval = pd.read_csv(V4_DIR / "FINAL_MODEL_EVALUATION.csv")
df_adni = pd.read_csv(PREDS_DIR / "adni_external_predictions.csv")
y_adni = df_adni["true_label"].values

# --------------------------------------------------------------------------
# Fig 7: Parameter Efficiency vs External Generalization Trade-Off
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

models_plot = [
    ("3D ResNet-10", "3d_resnet_10", "#2ca02c"),
    ("3D ResNet-18", "3d_resnet_18", "#1f77b4"),
    ("3D ResNet-34", "3d_resnet_34", "#17becf"),
    ("3D DenseNet-121", "3d_densenet_121", "#9467bd"),
    ("2.5D Multi-View ResNet-18", "2_5d_multi_view_resnet_18", "#8c564b"),
    ("Dual-Branch Hybrid", "dual_branch_hybrid", "#e377c2"),
    ("BrainIAC (Linear Probe)", "brainiac_vit_b16_linear_probe", "#d62728"),
    ("Swin UNETR SSL (Linear Probe)", "swin_unetr_ssl_linear_probe", "#ff7f0e"),
    ("FOMO25 AMAES (Linear Probe)", "fomo25_amaes_linear_probe", "#bcbd22"),
]

for dname, key, col in models_plot:
    r = df_eval[df_eval["model_key"] == key].iloc[0]
    p_count = r["total_parameters"] / 1e6
    auc = r["adni_auc"]
    ax.scatter(p_count, auc, color=col, s=120, edgecolors="black", zorder=4)
    # text placement offset
    dx = 1.5 if p_count < 50 else -15
    dy = 0.005 if "Swin" in dname or "ResNet-10" in dname else -0.008
    ax.annotate(dname, (p_count, auc), xytext=(p_count + dx, auc + dy),
                fontsize=9, fontweight="bold", zorder=5)

ax.set_xscale("log")
ax.set_xlabel("Total Model Parameters (Millions, Log Scale)")
ax.set_ylabel("ADNI External Generalization ROC-AUC")
ax.set_title("Parameter Efficiency vs Cross-Cohort Generalization (ADNI N=500)", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (AUC=0.50)")
ax.grid(True, linestyle="--", alpha=0.5, which="both")
ax.legend(loc="lower right")

plt.savefig(FIGS_DIR / "fig7_parameter_efficiency_tradeoff.png", dpi=300)
plt.close()

# --------------------------------------------------------------------------
# Fig 8: Confusion Matrices for Primary Architectures on ADNI (N=500)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
primary_cm = [
    ("3D ResNet-18", "3d_resnet_18"),
    ("MONAI Swin UNETR SSL", "swin_unetr_ssl_linear_probe"),
    ("BrainIAC ViT-B/16", "brainiac_vit_b16_linear_probe"),
    ("FOMO25 AMAES ResEnc-B", "fomo25_amaes_linear_probe")
]

for idx, (dname, key) in enumerate(primary_cm):
    p = df_adni[key].values
    tau = df_eval[df_eval["model_key"] == key]["val_optimal_threshold"].values[0]
    pred = (p >= tau).astype(int)
    cm = confusion_matrix(y_adni, pred, labels=[0, 1])

    ax = axes[idx]
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(f"{dname}\n(tau*={tau:.2f})", fontweight="bold", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["CN (0)", "AD (1)"])
    ax.set_yticklabels(["CN (0)", "AD (1)"])
    ax.set_xlabel("Predicted")
    if idx == 0:
        ax.set_ylabel("Ground Truth")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 150 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

plt.tight_layout()
plt.savefig(FIGS_DIR / "fig8_confusion_matrices.png", dpi=300)
plt.close()

# --------------------------------------------------------------------------
# Fig 9: Calibration Curves (Reliability Diagram)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
from sklearn.calibration import calibration_curve

for dname, key, col in [("3D ResNet-18", "3d_resnet_18", "#1f77b4"),
                        ("MONAI Swin UNETR", "swin_unetr_ssl_linear_probe", "#ff7f0e"),
                        ("BrainIAC ViT-B/16", "brainiac_vit_b16_linear_probe", "#d62728"),
                        ("Classical Radiomics", "classical_radiomics", "#2ca02c")]:
    p = df_adni[key].values
    prob_true, prob_pred = calibration_curve(y_adni, p, n_bins=8, strategy="uniform")
    ece = df_eval[df_eval["model_key"] == key]["adni_ece"].values[0]
    ax.plot(prob_pred, prob_true, marker="o", lw=2, label=f"{dname} (ECE={ece:.3f})", color=col)

ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Fraction of Positives (AD)")
ax.set_title("Reliability Diagram: Calibration on ADNI External Cohort (N=500)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

plt.savefig(FIGS_DIR / "fig9_calibration_curves.png", dpi=300)
plt.close()

# --------------------------------------------------------------------------
# Fig 10: Three-Cohort Multi-Paradigm Benchmark Summary
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

df_three = pd.read_csv(V4_DIR / "FINAL_THREE_COHORT_MODEL_COMPARISON.csv")
sub_three = df_three[df_three["model_key"].isin([
    "3d_resnet_10", "3d_resnet_18", "3d_resnet_34", "3d_densenet_121",
    "2_5d_multi_view_resnet_18", "dual_branch_hybrid", "classical_radiomics",
    "brainiac_vit_b16_linear_probe", "swin_unetr_ssl_linear_probe", "fomo25_amaes_linear_probe"
])].copy()

x_idx = np.arange(len(sub_three))
w = 0.25

ax.bar(x_idx - w, sub_three["oasis1_test_auc"], width=w, label="OASIS-1 Locked Test (N=30)", color="#2ecc71", edgecolor="black")
ax.bar(x_idx, sub_three["adni_auc"], width=w, label="ADNI External Test (N=500)", color="#3498db", edgecolor="black")
ax.bar(x_idx + w, sub_three["oasis2_auc"], width=w, label="OASIS-2 Exploratory (N=150)", color="#9b59b6", edgecolor="black")

ax.set_xticks(x_idx)
ax.set_xticklabels(sub_three["model_name"], rotation=35, ha="right", fontsize=9, fontweight="bold")
ax.set_ylabel("ROC-AUC")
ax.set_title("Three-Cohort Benchmark Summary Across 10 Representative Architectures", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend(loc="upper right")
ax.set_ylim(0.3, 0.8)

plt.tight_layout()
plt.savefig(FIGS_DIR / "fig10_three_cohort_summary.png", dpi=300)
plt.close()

print("Figures 7, 8, 9, 10 generated successfully.")
