import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score

PROJECT_ROOT = Path("D:/ml project")
V4_DIR = PROJECT_ROOT / "results" / "v4"
FIGS_DIR = V4_DIR / "figures"
PREDS_DIR = V4_DIR / "predictions"
SUBMISSION_FIGS_DIR = PROJECT_ROOT / "FINAL_SUBMISSION_23_OCT_2026" / "05_FIGURES"

FIGS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

df_eval = pd.read_csv(V4_DIR / "FINAL_MODEL_EVALUATION.csv")
test_preds = pd.read_csv(PREDS_DIR / "oasis1_locked_test_predictions.csv")
adni_preds = pd.read_csv(PREDS_DIR / "adni_external_predictions.csv")
o2_preds = pd.read_csv(PREDS_DIR / "oasis2_external_predictions.csv")

test_y = test_preds["true_label"].values
adni_y = adni_preds["true_label"].values
o2_y = o2_preds["true_label"].values

print("Regenerating all 10 figures from canonical results...")

# --------------------------------------------------------------------------
# Figure 1: Leaderboard ROC Curves (OASIS-1 Test vs ADNI)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
models_to_plot = [
    ("3D ResNet-18", "3d_resnet_18", "#1f77b4"),
    ("3D ResNet-10", "3d_resnet_10", "#2ca02c"),
    ("3D DenseNet-121", "3d_densenet_121", "#9467bd"),
    ("2D Transfer ResNet-18", "2d_transfer_resnet_18", "#e377c2"),
    ("Dual-Branch Hybrid", "dual_branch_hybrid", "#8c564b"),
    ("BrainIAC (Linear Probe)", "brainiac_vit_b16_linear_probe", "#d62728"),
    ("Swin UNETR SSL (Linear Probe)", "swin_unetr_ssl_linear_probe", "#ff7f0e"),
    ("FOMO25 AMAES (Linear Probe)", "fomo25_amaes_linear_probe", "#17becf"),
]

for dname, key, col in models_to_plot:
    fpr_o, tpr_o, _ = roc_curve(test_y, test_preds[key])
    auc_o = roc_auc_score(test_y, test_preds[key])
    axes[0].plot(fpr_o, tpr_o, label=f"{dname} ({auc_o:.3f})", color=col, lw=2)

    fpr_a, tpr_a, _ = roc_curve(adni_y, adni_preds[key])
    auc_a = roc_auc_score(adni_y, adni_preds[key])
    axes[1].plot(fpr_a, tpr_a, label=f"{dname} ({auc_a:.3f})", color=col, lw=2)

for i, title in enumerate(["OASIS-1 Locked Test (N=30, In-Distribution)", "ADNI Zero-Shot External (N=500, Domain Shift)"]):
    axes[i].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
    axes[i].set_title(title, fontweight="bold")
    axes[i].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[i].grid(True, linestyle="--", alpha=0.5)
    axes[i].legend(loc="lower right", fontsize=8)

axes[0].set_ylabel("True Positive Rate (Sensitivity)")
plt.savefig(FIGS_DIR / "fig1_leaderboard_roc_curves.png", dpi=300)
plt.close()
print("  - Saved Fig 1")

# --------------------------------------------------------------------------
# Figure 2: Generalization Gap (OASIS-1 Test vs ADNI)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df_eval.sort_values(by="generalization_gap_delta_auc", ascending=False)
y_pos = np.arange(len(df_sorted))
ax.barh(y_pos, df_sorted["generalization_gap_delta_auc"], color="#d9534f", alpha=0.85, edgecolor="black")
ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted["model_name"])
ax.invert_yaxis()
ax.set_xlabel("Generalization Gap Delta AUC (OASIS-1 Test AUC - ADNI AUC)")
ax.set_title("Cross-Cohort Generalization Collapse Across All Architectures", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5, axis="x")
for idx, v in enumerate(df_sorted["generalization_gap_delta_auc"]):
    ax.text(v + 0.005 if v >= 0 else v - 0.035, idx, f"{v:+.3f}", va="center", fontsize=9, fontweight="bold")
plt.savefig(FIGS_DIR / "fig2_cross_cohort_generalization_gap.png", dpi=300)
plt.close()
print("  - Saved Fig 2")

# --------------------------------------------------------------------------
# Figure 3: Foundation vs Supervised ResNets
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
comp_models = [
    "3D ResNet-10", "3D ResNet-18", "3D ResNet-34", "3D DenseNet-121",
    "BrainIAC ViT-B/16 (Linear Probe)", "MONAI Swin UNETR SSL (Linear Probe)", "FOMO25 AMAES ResEnc-B (Linear Probe)"
]
sub_df = df_eval[df_eval["model_name"].isin(comp_models)].copy()
sub_df = sub_df.sort_values(by="adni_auc", ascending=False)
colors = ["#2b5c8f" if "ResNet" in m or "DenseNet" in m else "#c0392b" for m in sub_df["model_name"]]
x_pos = np.arange(len(sub_df))
ax.bar(x_pos, sub_df["adni_auc"], yerr=[sub_df["adni_auc"] - sub_df["adni_ci_lower"], sub_df["adni_ci_upper"] - sub_df["adni_auc"]], capsize=5, color=colors, alpha=0.85, edgecolor="black")
ax.set_xticks(x_pos)
ax.set_xticklabels(sub_df["model_name"], rotation=30, ha="right")
ax.axhline(0.5, color="gray", linestyle="--", label="Chance (AUC=0.50)")
ax.set_ylabel("ADNI External ROC-AUC (95% CI)")
ax.set_title("Foundation Models vs Well-Tuned Supervised Baselines (ADNI N=500)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
plt.savefig(FIGS_DIR / "fig3_foundation_vs_supervised.png", dpi=300)
plt.close()
print("  - Saved Fig 3")

# --------------------------------------------------------------------------
# Figure 4: Fine-Tuning Impact (Linear Probe vs Partial FT)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ft_df = pd.read_csv(V4_DIR / "FINAL_FINE_TUNING_COMPARISON.csv")
x = np.arange(len(ft_df))
w = 0.35
ax.bar(x - w/2, ft_df["linear_probe_auc"], w, label="Linear Probe (Frozen Encoder)", color="#2ecc71", edgecolor="black")
ax.bar(x + w/2, ft_df["partial_ft_auc"], w, label="Partial Fine-Tuning (Top 2 Layers)", color="#e74c3c", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(ft_df["foundation_backbone"], fontsize=10)
ax.set_ylabel("ADNI External ROC-AUC")
ax.set_title("Impact of Fine-Tuning on Out-of-Distribution Transfer (ADNI N=500)", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend()
plt.savefig(FIGS_DIR / "fig4_fine_tuning_impact.png", dpi=300)
plt.close()
print("  - Saved Fig 4")

# --------------------------------------------------------------------------
# Figure 5: TrustGate Selective Prediction / Abstention Curves
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
p_r18 = adni_preds["3d_resnet_18"].values
tau_star = df_eval[df_eval["model_key"] == "3d_resnet_18"]["val_optimal_threshold"].values[0]
pred_r18 = (p_r18 >= tau_star).astype(int)
baseline_acc = accuracy_score(adni_y, pred_r18)

detectors = [
    ("Confidence Margin", 1.0 - np.abs(p_r18 - 0.5)),
    ("Predictive Entropy", -(p_r18 * np.log(p_r18 + 1e-8) + (1 - p_r18) * np.log(1 - p_r18 + 1e-8))),
    ("Ensemble Disagreement", np.std(np.stack([adni_preds[k].values for k in ["3d_resnet_10", "3d_resnet_18", "3d_resnet_34"]]), axis=0)),
    ("Composite TrustGate Risk", (1.0 - np.abs(p_r18 - 0.5)) * 0.5 + np.std(np.stack([adni_preds[k].values for k in ["3d_resnet_10", "3d_resnet_18", "3d_resnet_34"]]), axis=0) * 0.5),
]

rejection_rates = np.linspace(0.0, 0.50, 11)
for dname, score in detectors:
    accs = []
    for r_rate in rejection_rates:
        if r_rate == 0.0:
            accs.append(baseline_acc)
        else:
            cutoff = np.percentile(score, (1.0 - r_rate) * 100.0)
            mask = score <= cutoff
            accs.append(accuracy_score(adni_y[mask], pred_r18[mask]))
    ax.plot(rejection_rates * 100.0, accs, label=dname, lw=2, marker="o", markersize=4)

ax.set_xlabel("Cohort Rejection / Abstention Rate (%)")
ax.set_ylabel("Accuracy on Retained ADNI Cohort")
ax.set_title("SHIFT-MRI TrustGate: Selective Prediction under Domain Shift", fontweight="bold")
ax.axhline(baseline_acc, color="black", linestyle="--", label=f"Unretained Baseline ({baseline_acc:.1%})")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.savefig(FIGS_DIR / "fig5_trustgate_abstention_curves.png", dpi=300)
plt.close()
print("  - Saved Fig 5")

# --------------------------------------------------------------------------
# Figure 6: Subgroup Scanner Analysis (Siemens N=370 vs Philips N=130)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
x_sub = np.arange(len(sub_df))
width = 0.35
ax.bar(x_sub - width/2, sub_df["adni_siemens_auc"], width, label="Siemens (N=370)", color="#3498db", edgecolor="black")
ax.bar(x_sub + width/2, sub_df["adni_philips_auc"], width, label="Philips (N=130)", color="#e67e22", edgecolor="black")
ax.set_xticks(x_sub)
ax.set_xticklabels(sub_df["model_name"], rotation=30, ha="right")
ax.set_ylabel("Subgroup ROC-AUC")
ax.set_title("Cross-Scanner Subgroup Generalization on ADNI (Siemens vs Philips)", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend()
plt.savefig(FIGS_DIR / "fig6_subgroup_scanner_transfer.png", dpi=300)
plt.close()
print("  - Saved Fig 6")

# --------------------------------------------------------------------------
# Figure 7: Parameter Efficiency vs External Generalization Trade-Off
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
print("  - Saved Fig 7")

# --------------------------------------------------------------------------
# Figure 8: Confusion Matrices for Primary Architectures on ADNI (N=500)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
primary_cm = [
    ("3D ResNet-18", "3d_resnet_18"),
    ("MONAI Swin UNETR SSL", "swin_unetr_ssl_linear_probe"),
    ("BrainIAC ViT-B/16", "brainiac_vit_b16_linear_probe"),
    ("FOMO25 AMAES ResEnc-B", "fomo25_amaes_linear_probe")
]

for idx, (dname, key) in enumerate(primary_cm):
    p = adni_preds[key].values
    tau = df_eval[df_eval["model_key"] == key]["val_optimal_threshold"].values[0]
    pred = (p >= tau).astype(int)
    cm = confusion_matrix(adni_y, pred, labels=[0, 1])

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
            val = cm[i, j]
            ax.text(j, i, f"{val}", ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black", fontweight="bold")

plt.savefig(FIGS_DIR / "fig8_confusion_matrices.png", dpi=300)
plt.close()
print("  - Saved Fig 8")

# --------------------------------------------------------------------------
# Figure 9: Reliability and Calibration Curves (ECE)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
cal_models = [
    ("3D ResNet-18", "3d_resnet_18", "#1f77b4"),
    ("MONAI Swin UNETR SSL", "swin_unetr_ssl_linear_probe", "#ff7f0e"),
    ("BrainIAC ViT-B/16", "brainiac_vit_b16_linear_probe", "#d62728"),
    ("FOMO25 AMAES", "fomo25_amaes_linear_probe", "#17becf"),
]

for dname, key, col in cal_models:
    p = adni_preds[key].values
    ece = df_eval[df_eval["model_key"] == key]["adni_ece"].values[0]
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_confs = []
    bin_accs = []
    for i in range(10):
        idx_bin = (p >= bin_edges[i]) & (p < bin_edges[i+1])
        if idx_bin.sum() > 0:
            bin_confs.append(np.mean(p[idx_bin]))
            bin_accs.append(np.mean(adni_y[idx_bin]))
    ax.plot(bin_confs, bin_accs, marker="s", label=f"{dname} (ECE={ece:.3f})", color=col, lw=2)

ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability (Confidence)")
ax.set_ylabel("Empirical Accuracy")
ax.set_title("Calibration Reliability Curves on ADNI (N=500)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.savefig(FIGS_DIR / "fig9_calibration_curves.png", dpi=300)
plt.close()
print("  - Saved Fig 9")

# --------------------------------------------------------------------------
# Figure 10: Three-Cohort Summary Bar Chart
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
sub_models = [
    "3D ResNet-10", "3D ResNet-18", "3D ResNet-34", "3D DenseNet-121",
    "2.5D Multi-View ResNet-18", "Dual-Branch Hybrid", "Classical Radiomics (Linear SVM)",
    "BrainIAC ViT-B/16 (Linear Probe)", "MONAI Swin UNETR SSL (Linear Probe)", "FOMO25 AMAES ResEnc-B (Linear Probe)"
]
df_three = df_eval[df_eval["model_name"].isin(sub_models)].copy()
x_idx = np.arange(len(df_three))
width = 0.25

ax.bar(x_idx - width, df_three["oasis_test_auc"], width, label="OASIS-1 Test (N=30)", color="#2b5c8f", edgecolor="black")
ax.bar(x_idx, df_three["adni_auc"], width, label="ADNI External (N=500)", color="#d9534f", edgecolor="black")
ax.bar(x_idx + width, df_three["oasis2_auc"], width, label="OASIS-2 External (N=150)", color="#27ae60", edgecolor="black")

ax.set_xticks(x_idx)
ax.set_xticklabels(df_three["model_name"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("ROC-AUC")
ax.set_title("Three-Cohort Performance Across Evaluated Paradigms", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", label="Chance (AUC=0.50)")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend(loc="upper right")

plt.savefig(FIGS_DIR / "fig10_three_cohort_summary.png", dpi=300)
plt.close()
print("  - Saved Fig 10")

# Synchronize all 10 figures to Submission folder
for i in range(1, 11):
    f_patterns = list(FIGS_DIR.glob(f"fig{i}_*.png"))
    for f in f_patterns:
        shutil.copyfile(f, SUBMISSION_FIGS_DIR / f.name)
print(f"All 10 figures synchronized to: {SUBMISSION_FIGS_DIR}")
