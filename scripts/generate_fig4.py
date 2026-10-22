import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_path = r"D:\ml project\results\v4\FINAL_FINE_TUNING_COMPARISON.csv"
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(df))
width = 0.35

ax.bar(x - width/2, df["linear_probe_auc"], width, label="Linear Probe (Frozen Backbone)", color="#2980b9", edgecolor="black")
ax.bar(x + width/2, df["partial_ft_auc"], width, label="Partial Fine-Tuning (Top Stages)", color="#e74c3c", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels(df["foundation_backbone"], fontweight="bold")
ax.set_ylabel("ADNI External Transfer ROC-AUC")
ax.set_title("Impact of Fine-Tuning vs Linear Probing on Cross-Cohort Transfer", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", label="Chance (AUC=0.50)")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend(loc="lower left")

for i, (p, ft, d) in enumerate(zip(df["linear_probe_auc"], df["partial_ft_auc"], df["auc_delta_ft_minus_probe"])):
    ax.text(i - width/2, p + 0.01, f"{p:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + width/2, ft + 0.01, f"{ft:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i, max(p, ft) + 0.035, f"Δ = {d:+.3f}", ha="center", fontsize=10, color="#c0392b" if d < 0 else "#27ae60", fontweight="bold")

plt.ylim(0.4, 0.65)
plt.tight_layout()
fig_out = r"D:\ml project\results\v4\figures\fig4_fine_tuning_impact.png"
plt.savefig(fig_out, dpi=300)
plt.close()
print("Saved fig4 to", fig_out)
