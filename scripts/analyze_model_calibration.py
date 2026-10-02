"""
Priority 5: Model Calibration and Confidence Reliability Analysis.
Evaluates:
  - Brier Score
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Reliability Curves
  - Probability Confidence Distributions
Outputs:
  - results/extensions/calibration/calibration_results.csv
  - results/extensions/calibration/calibration_results.json
  - results/extensions/figures/figure_CAL1_reliability_curves.png
  - results/extensions/figures/figure_CAL2_probability_distributions.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5):
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            gap = np.abs(bin_acc - bin_conf)
            ece += (bin_count / n) * gap
            mce = max(mce, gap)

    return float(ece), float(mce)


def run_calibration_analysis(
    adni_test_json: str = "results/adni_locked_test_results.json",
    oasis_test_json: str = "results/extensions/supervised_baseline/model_report.json",
    output_dir: str = "results/extensions/calibration",
    figures_dir: str = "results/extensions/figures",
):
    logger = get_logger("CalibrationAnalysis")
    logger.info("=================================================================")
    logger.info("Starting Priority 5: Model Confidence Calibration Analysis")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    with open(adni_test_json, "r", encoding="utf-8") as f:
        adni_data = json.load(f)

    y_true_adni = np.array(adni_data["ground_truth_labels"])

    calib_records = []
    prob_dict = {}

    colors = {
        "Supervised 3D ResNet-18": "#1f77b4",
        "BrainIAC (Linear Probe)": "#ff7f0e",
        "FOMO25 Winner (Linear Probe)": "#2ca02c",
        "MONAI Swin UNETR SSL (Linear Probe)": "#d62728",
    }

    for model_name, m_dict in adni_data["models"].items():
        probs = np.array(m_dict["ensemble_evaluation"]["probabilities"])
        prob_dict[model_name] = probs

        brier = float(brier_score_loss(y_true_adni, probs))
        ece, mce = compute_ece(y_true_adni, probs, n_bins=4)

        rec = {
            "cohort": "ADNI MPRAGE (N=8)",
            "model": model_name,
            "brier_score": round(brier, 4),
            "expected_calibration_error_ece": round(ece, 4),
            "max_calibration_error_mce": round(mce, 4),
            "mean_predicted_probability": round(float(np.mean(probs)), 4),
            "std_predicted_probability": round(float(np.std(probs)), 4),
            "probabilities": [round(float(p), 4) for p in probs],
        }
        calib_records.append(rec)
        logger.info(f"  [{model_name}] Brier: {brier:.4f} | ECE: {ece:.4f} | MCE: {mce:.4f} | Mean Prob: {np.mean(probs):.4f}")

    # Save CSV
    calib_df = pd.DataFrame(calib_records).drop(columns=["probabilities"])
    csv_path = os.path.join(output_dir, "calibration_results.csv")
    calib_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved calibration results CSV to {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "calibration_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(calib_records, f, indent=2)
    logger.info(f"Saved calibration results JSON to {json_path}")

    # -------------------------------------------------------------------------
    # FIGURE CAL1: Reliability / Calibration Curves
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    for m_name, probs in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true_adni, probs, n_bins=4)
        c = colors.get(m_name, "#333333")
        ax.plot(mean_pred, frac_pos, marker="o", color=c, lw=2, label=f"{m_name.split()[0]} (Brier={brier_score_loss(y_true_adni, probs):.3f})")

    ax.set_title("Model Reliability Curves (Ensemble, ADNI Locked Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax.set_ylabel("Empirical True Fraction Positives", fontsize=11)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    fig1_path = os.path.join(figures_dir, "figure_CAL1_reliability_curves.png")
    plt.tight_layout(); plt.savefig(fig1_path); plt.close()
    print(f"Saved Figure CAL1 to {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURE CAL2: Probability Confidence Distributions
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    for i, (m_name, probs) in enumerate(prob_dict.items()):
        ax = axes[i]
        c = colors.get(m_name, "#1f77b4")
        ax.hist(probs[y_true_adni == 0], bins=np.linspace(0, 1, 6), alpha=0.7, color="#2ca02c", label="CN (True 0)", edgecolor="black")
        ax.hist(probs[y_true_adni == 1], bins=np.linspace(0, 1, 6), alpha=0.7, color="#d62728", label="AD (True 1)", edgecolor="black")
        ax.set_title(m_name.split("(")[0].strip(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Probability of AD", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig2_path = os.path.join(figures_dir, "figure_CAL2_probability_distributions.png")
    plt.tight_layout(); plt.savefig(fig2_path); plt.close()
    print(f"Saved Figure CAL2 to {fig2_path}")

    return calib_df

if __name__ == "__main__":
    run_calibration_analysis()
