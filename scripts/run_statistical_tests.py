"""
Priority 8 / Experiment 8: Statistical Analysis and Confidence Interval Estimation.
Computes:
- Non-parametric bootstrap 95% confidence intervals for ROC-AUC
- Summary statistics across seeds (Mean, SD, Median, IQR, 95% CI)
- Paired comparison tests between models across random seeds
- Explicit reporting of sample size limitations

Outputs:
- results/extensions/statistics/statistical_tests.csv
- results/extensions/statistics/statistical_tests.json
- results/extensions/figures/figure_STAT1_bootstrap_cis.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from utils.seed import set_seed


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Computes empirical percentile bootstrap confidence intervals for ROC-AUC."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        sample_prob = y_prob[idx]
        if len(np.unique(sample_y)) > 1:
            try:
                score = roc_auc_score(sample_y, sample_prob)
                boot_aucs.append(score)
            except Exception:
                pass

    if len(boot_aucs) < 50:
        return np.nan, np.nan, np.nan

    lower = float(np.percentile(boot_aucs, (alpha / 2.0) * 100))
    upper = float(np.percentile(boot_aucs, (1.0 - alpha / 2.0) * 100))
    point = float(roc_auc_score(y_true, y_prob))
    return point, lower, upper


def run_statistical_analysis():
    logger = get_logger("StatisticalAnalysis")
    logger.info("=================================================================")
    logger.info("Starting Priority 8: Statistical Uncertainty & Hypothesis Testing")
    logger.info("=================================================================")

    output_dir = "results/extensions/statistics"
    figures_dir = "results/extensions/figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    stat_records = []

    # 1. ADNI Locked Test Set Bootstraps
    with open("results/adni_locked_test_results.json", "r", encoding="utf-8") as f:
        adni_res = json.load(f)

    y_true_adni = np.array(adni_res["ground_truth_labels"])

    logger.info("--- ADNI Locked Test Set (N=8) Bootstrap CIs ---")
    for m_name, m_dict in adni_res["models"].items():
        probs = np.array(m_dict["ensemble_evaluation"]["probabilities"])
        point_auc, ci_lower, ci_upper = bootstrap_auc_ci(y_true_adni, probs)

        ci_width = round(ci_upper - ci_lower, 4) if not np.isnan(ci_lower) else None
        rec = {
            "experiment": "ADNI Locked Test Set (Ensemble)",
            "model": m_name,
            "sample_size": len(y_true_adni),
            "metric": "ROC-AUC",
            "point_estimate": round(point_auc, 4),
            "ci_95_lower": round(ci_lower, 4) if not np.isnan(ci_lower) else None,
            "ci_95_upper": round(ci_upper, 4) if not np.isnan(ci_upper) else None,
            "ci_width": ci_width,
            "statistical_note": "Confidence intervals are extremely wide because of the small test cohort (N=8). Differences must be interpreted cautiously without claims of definitive statistical superiority.",
        }
        stat_records.append(rec)
        logger.info(f"  {m_name}: Point AUC={point_auc:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}] (Width: {ci_width})")

    # 2. OASIS Data Efficiency Across-Seed Summary
    de_csv = "results/extensions/data_efficiency/oasis_data_efficiency_results.csv"
    if os.path.exists(de_csv):
        de_df = pd.read_csv(de_csv)
        logger.info("\n--- OASIS Data Efficiency Multi-Seed Distributions (5 Seeds) ---")
        for (m_name, frac), grp in de_df.groupby(["model", "fraction"]):
            aucs = grp["test_roc_auc"].values
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            median_auc = float(np.median(aucs))
            iqr_auc = float(stats.iqr(aucs))
            # Normal approximation 95% CI of the mean across seeds
            ci_half = 1.96 * (std_auc / np.sqrt(len(aucs))) if len(aucs) > 1 else 0.0

            rec = {
                "experiment": f"OASIS Data-Efficiency ({frac} Fraction)",
                "model": m_name,
                "sample_size": len(aucs),
                "metric": "Across-Seed Test ROC-AUC",
                "point_estimate": round(mean_auc, 4),
                "std": round(std_auc, 4),
                "median": round(median_auc, 4),
                "iqr": round(iqr_auc, 4),
                "ci_95_lower": round(mean_auc - ci_half, 4),
                "ci_95_upper": round(mean_auc + ci_half, 4),
                "ci_width": round(2 * ci_half, 4),
                "statistical_note": f"Evaluated across 5 random seeds [42, 43, 44, 45, 46]. Standard error of the mean across seeds reflects sensitivity to partition sampling.",
            }
            stat_records.append(rec)

    # 3. Model Pairwise Comparison (ResNet-34 vs ResNet-18 on 100% OASIS)
    r34_csv = "results/extensions/supervised_baseline/results.csv"
    if os.path.exists(r34_csv) and os.path.exists(de_csv):
        r34_df = pd.read_csv(r34_csv)
        r18_df = de_df[(de_df["slug"] == "resnet18") & (de_df["fraction"] == "100%")].sort_values("seed")
        r34_df = r34_df.sort_values("seed")

        diffs = r34_df["test_roc_auc"].values - r18_df["test_roc_auc"].values
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))

        # Wilcoxon signed-rank test or paired t-test
        t_stat, p_val = stats.ttest_rel(r34_df["test_roc_auc"].values, r18_df["test_roc_auc"].values)
        rec = {
            "experiment": "Paired Model Comparison: ResNet-34 vs ResNet-18 (100% OASIS)",
            "model": "ResNet-34 - ResNet-18",
            "sample_size": len(diffs),
            "metric": "Delta ROC-AUC",
            "point_estimate": round(mean_diff, 4),
            "std": round(std_diff, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 4),
            "statistical_note": "Paired across identical seeds. Given N=5 seeds and N=8 test set, p-values are descriptive and hypothesis tests have limited statistical power.",
        }
        stat_records.append(rec)
        logger.info(f"\nResNet-34 vs ResNet-18 Mean Delta AUC: {mean_diff:+.4f} (p={p_val:.4f})")

    # Save outputs
    stat_df = pd.DataFrame(stat_records)
    csv_path = os.path.join(output_dir, "statistical_tests.csv")
    stat_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved statistical tests CSV to {csv_path}")

    json_path = os.path.join(output_dir, "statistical_tests.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stat_records, f, indent=2)
    logger.info(f"Saved statistical tests JSON to {json_path}")

    # Plot Figure STAT1: Bootstrap Confidence Intervals on ADNI Locked Test Set
    adni_stats = [r for r in stat_records if r["experiment"] == "ADNI Locked Test Set (Ensemble)"]
    if adni_stats:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        models = [r["model"].split("(")[0].strip() for r in adni_stats]
        points = [r["point_estimate"] for r in adni_stats]
        lowers = [r["ci_95_lower"] for r in adni_stats]
        uppers = [r["ci_95_upper"] for r in adni_stats]
        y_pos = np.arange(len(models))

        xerr = [
            np.array(points) - np.array(lowers),
            np.array(uppers) - np.array(points)
        ]

        ax.errorbar(points, y_pos, xerr=xerr, fmt="o", color="#1f77b4", ecolor="#d62728", elinewidth=2, capsize=6, markersize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=10)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (AUC=0.5)")
        ax.set_xlabel("ROC-AUC (95% Bootstrap CI)", fontsize=11)
        ax.set_title("ADNI Locked Test Set: Model ROC-AUC with 95% Bootstrap CIs\n(Demonstrating High Uncertainty with N=8)", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        fig_path = os.path.join(figures_dir, "figure_STAT1_bootstrap_cis.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Saved Figure STAT1 to {fig_path}")

    return stat_df


if __name__ == "__main__":
    run_statistical_analysis()
