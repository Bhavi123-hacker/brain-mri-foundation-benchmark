"""
ADNI Locked Test Set Benchmark Evaluation Script (Day 4).
Evaluates all 20 trained models (4 model families x 5 seeds) on the isolated 8-subject holdout test cohort.
Models:
  1. Supervised 3D ResNet-18 Baseline (Seeds 42-46)
  2. BrainIAC Linear Probe (Seeds 42-46)
  3. FOMO25 Winner Linear Probe (Seeds 42-46)
  4. MONAI Swin UNETR SSL Linear Probe (Seeds 42-46)
Metrics:
  ROC-AUC, Accuracy, Balanced Accuracy, Sensitivity/Recall, Specificity, Precision, F1-Score, Brier Score, Confusion Matrix, 95% Bootstrap Confidence Intervals.
Outputs:
  - results/adni_locked_test_results.csv
  - results/adni_locked_test_results.json
  - results/adni_model_comparison.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from models.baseline.resnet3d import build_resnet18_3d
from models.foundation_brainiac.adapter import BrainIACAdapter
from models.foundation_fomo25.adapter import FOMO25Adapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from evaluation.metrics import compute_classification_metrics


def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42):
    """Compute 95% bootstrap confidence intervals for key classification metrics."""
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_metrics = {"roc_auc": [], "accuracy": [], "balanced_accuracy": [], "f1": [], "brier_score": []}

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]
        # Only evaluate if at least one sample from both classes exists
        if len(np.unique(sample_true)) < 2:
            continue
        m = compute_classification_metrics(sample_true, sample_prob)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    ci_results = {}
    for k, vals in boot_metrics.items():
        if len(vals) > 0:
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
            ci_results[f"{k}_ci_95"] = [round(low, 4), round(high, 4)]
        else:
            ci_results[f"{k}_ci_95"] = [None, None]
    return ci_results


def load_test_scans(splits_csv: str = "results/adni_cn_ad_split.csv", cache_dir: str = "data/cache/adni_cn_ad"):
    """Load preprocessed 3D scans and clinical metadata for the locked test cohort."""
    df = pd.read_csv(splits_csv)
    test_df = df[df["split"] == "test"].sort_values("subject_id").reset_index(drop=True)
    assert len(test_df) == 8, f"Expected 8 test subjects, got {len(test_df)}"

    tensors = []
    labels = []
    sids = []
    diagnoses = []
    sexes = []
    ages = []
    mfgs = []

    for _, row in test_df.iterrows():
        sid = str(row["subject_id"]).strip()
        npy_path = os.path.join(cache_dir, f"{sid}.npy")
        assert os.path.exists(npy_path), f"Missing cache: {npy_path}"
        arr = np.load(npy_path)
        tensors.append(torch.from_numpy(arr).to(torch.float32))
        labels.append(float(row["label"]))
        sids.append(sid)
        diagnoses.append(row["baseline_diagnosis"])
        sexes.append(row.get("sex", "UNKNOWN"))
        ages.append(float(row["age"]) if pd.notnull(row.get("age")) else None)
        mfgs.append(row.get("manufacturer", "UNKNOWN"))

    stacked_tensors = torch.stack(tensors)  # (8, 1, 128, 128, 128)
    labels_arr = np.array(labels, dtype=np.float32)

    metadata = {
        "subject_ids": sids,
        "diagnoses": diagnoses,
        "sexes": sexes,
        "ages": ages,
        "manufacturers": mfgs,
    }
    return stacked_tensors, labels_arr, metadata


def evaluate_adni_locked_test(
    splits_csv: str = "results/adni_cn_ad_split.csv",
    checkpoint_dir: str = "results/checkpoints",
    output_dir: str = "results",
    seeds: list = [42, 43, 44, 45, 46],
):
    logger = get_logger("ADNILockedSetEvaluation")
    logger.info("=================================================================")
    logger.info("Starting Day 4: ADNI Locked Test Set Benchmark Evaluation")
    logger.info("Evaluating 4 Model Architectures x 5 Random Seeds on 8-Subject Test Cohort")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference Device: {device}")

    # Load locked test cohort
    test_tensors, y_true, meta = load_test_scans(splits_csv)
    logger.info(f"Loaded {len(test_tensors)} locked test scans (Class Distribution: {dict(pd.Series(meta['diagnoses']).value_counts())})")
    logger.info(f"Test Subject IDs: {meta['subject_ids']}")

    model_configs = [
        {
            "family": "Supervised 3D ResNet-18",
            "slug": "resnet18",
            "type": "supervised_baseline",
            "builder": lambda: build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2),
            "ckpt_template": "adni_resnet18_seed{seed}_best.pt",
            "total_params": 33160513,
            "trainable_params": 33160513,
        },
        {
            "family": "BrainIAC (Linear Probe)",
            "slug": "brainiac",
            "type": "foundation_linear_probe",
            "builder": lambda: BrainIACAdapter(
                output_dim=1,
                weights_path="weights/brainiac_vit_base.pt",
                adaptation_mode="linear_probe",
                dropout=0.2,
            ),
            "ckpt_template": "adni_brainiac_linear_seed{seed}_best.pt",
            "total_params": 6538049,
            "trainable_params": 769,
        },
        {
            "family": "FOMO25 Winner (Linear Probe)",
            "slug": "fomo25",
            "type": "foundation_linear_probe",
            "builder": lambda: FOMO25Adapter(
                output_dim=1,
                weights_path="weights/fomo25_winner_encoder.pt",
                adaptation_mode="linear_probe",
                dropout=0.2,
            ),
            "ckpt_template": "adni_fomo25_linear_seed{seed}_best.pt",
            "total_params": 1296641,
            "trainable_params": 513,
        },
        {
            "family": "MONAI Swin UNETR SSL (Linear Probe)",
            "slug": "swin_unetr",
            "type": "foundation_linear_probe",
            "builder": lambda: SwinUNETRSSLAdapter(
                output_dim=1,
                weights_path="weights/model_swinvit.pt",
                adaptation_mode="linear_probe",
                dropout=0.2,
            ),
            "ckpt_template": "adni_swin_unetr_linear_seed{seed}_best.pt",
            "total_params": 774913,
            "trainable_params": 769,
        },
    ]

    all_seed_results = []
    detailed_reports = {}
    comparison_rows = []

    for cfg in model_configs:
        family_name = cfg["family"]
        logger.info(f"\nEvaluating Model Family: {family_name}")
        family_seed_metrics = []
        family_prob_predictions = []

        for s in seeds:
            ckpt_name = cfg["ckpt_template"].format(seed=s)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

            # Instantiate model
            model = cfg["builder"]().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            # Execute forward inference on locked test set
            with torch.no_grad():
                logits = model(test_tensors.to(device))
                probs = torch.sigmoid(logits).cpu().squeeze(-1).numpy()

            preds = (probs >= 0.5).astype(int)

            # Compute classification metrics
            metrics = compute_classification_metrics(y_true, probs)
            ci = compute_bootstrap_ci(y_true, probs, n_bootstraps=1000, seed=s)

            # Subject-level predictions
            subject_preds = []
            for i, sid in enumerate(meta["subject_ids"]):
                subject_preds.append({
                    "subject_id": sid,
                    "baseline_diagnosis": meta["diagnoses"][i],
                    "true_label": int(y_true[i]),
                    "predicted_prob": round(float(probs[i]), 4),
                    "predicted_class": int(preds[i]),
                    "correct": bool(int(preds[i]) == int(y_true[i])),
                })

            record = {
                "model_family": family_name,
                "model_slug": cfg["slug"],
                "model_type": cfg["type"],
                "seed": s,
                "checkpoint_path": ckpt_path,
                "total_params": cfg["total_params"],
                "trainable_params": cfg["trainable_params"],
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_roc_auc": metrics["roc_auc"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_specificity": metrics["specificity"],
                "test_brier_score": metrics["brier_score"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "roc_auc_ci_95": ci["roc_auc_ci_95"],
                "accuracy_ci_95": ci["accuracy_ci_95"],
                "f1_ci_95": ci["f1_ci_95"],
                "subject_predictions": subject_preds,
            }

            all_seed_results.append(record)
            family_seed_metrics.append(metrics)
            family_prob_predictions.append(probs)

            logger.info(
                f"  Seed {s:2d} -> Acc: {metrics['accuracy']:.4f} | Bal Acc: {metrics['balanced_accuracy']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | Brier: {metrics['brier_score']:.4f} | "
                f"CM: [TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}]"
            )

        # Ensemble prediction (average probability across the 5 seeds)
        ensemble_probs = np.mean(family_prob_predictions, axis=0)
        ensemble_metrics = compute_classification_metrics(y_true, ensemble_probs)
        ensemble_ci = compute_bootstrap_ci(y_true, ensemble_probs, n_bootstraps=1000, seed=42)

        # Aggregate across seeds
        accs = [m["accuracy"] for m in family_seed_metrics]
        bal_accs = [m["balanced_accuracy"] for m in family_seed_metrics]
        aucs = [m["roc_auc"] for m in family_seed_metrics]
        f1s = [m["f1"] for m in family_seed_metrics]
        briers = [m["brier_score"] for m in family_seed_metrics]
        recs = [m["recall"] for m in family_seed_metrics]
        specs = [m["specificity"] for m in family_seed_metrics]

        comp_row = {
            "model_family": family_name,
            "model_slug": cfg["slug"],
            "model_type": cfg["type"],
            "total_params": cfg["total_params"],
            "trainable_params": cfg["trainable_params"],
            "test_roc_auc_mean": round(float(np.mean(aucs)), 4),
            "test_roc_auc_std": round(float(np.std(aucs)), 4),
            "test_accuracy_mean": round(float(np.mean(accs)), 4),
            "test_accuracy_std": round(float(np.std(accs)), 4),
            "test_balanced_acc_mean": round(float(np.mean(bal_accs)), 4),
            "test_balanced_acc_std": round(float(np.std(bal_accs)), 4),
            "test_f1_mean": round(float(np.mean(f1s)), 4),
            "test_f1_std": round(float(np.std(f1s)), 4),
            "test_recall_mean": round(float(np.mean(recs)), 4),
            "test_specificity_mean": round(float(np.mean(specs)), 4),
            "test_brier_mean": round(float(np.mean(briers)), 4),
            "ensemble_roc_auc": ensemble_metrics["roc_auc"],
            "ensemble_accuracy": ensemble_metrics["accuracy"],
            "ensemble_f1": ensemble_metrics["f1"],
            "ensemble_brier": ensemble_metrics["brier_score"],
            "ensemble_roc_auc_ci_95": f"[{ensemble_ci['roc_auc_ci_95'][0]}, {ensemble_ci['roc_auc_ci_95'][1]}]",
        }
        comparison_rows.append(comp_row)

        detailed_reports[family_name] = {
            "slug": cfg["slug"],
            "total_params": cfg["total_params"],
            "trainable_params": cfg["trainable_params"],
            "cross_seed_summary": comp_row,
            "ensemble_evaluation": {
                "metrics": ensemble_metrics,
                "ci_95": ensemble_ci,
                "probabilities": [round(float(p), 4) for p in ensemble_probs],
            },
            "seeds": [r for r in all_seed_results if r["model_family"] == family_name],
        }

    # Save results CSV
    results_df = pd.DataFrame(all_seed_results)
    # Exclude subject_predictions list from flat CSV
    results_csv_df = results_df.drop(columns=["subject_predictions"])
    results_csv_path = os.path.join(output_dir, "adni_locked_test_results.csv")
    results_csv_df.to_csv(results_csv_path, index=False)
    logger.info(f"\nSaved locked test results CSV to {results_csv_path}")

    # Save model comparison CSV
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(output_dir, "adni_model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved model comparison CSV to {comp_csv_path}")

    # Save full JSON report
    full_report = {
        "benchmark_title": "Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Baselines (ADNI Locked Test Cohort)",
        "cohort": "ADNI MPRAGE (CN vs AD)",
        "test_subjects_count": len(meta["subject_ids"]),
        "test_subjects": meta["subject_ids"],
        "ground_truth_labels": [int(y) for y in y_true],
        "ground_truth_diagnoses": meta["diagnoses"],
        "class_balance": {
            "CN": int((y_true == 0.0).sum()),
            "AD": int((y_true == 1.0).sum()),
        },
        "model_comparison_table": comparison_rows,
        "models": detailed_reports,
    }

    report_json_path = os.path.join(output_dir, "adni_locked_test_results.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Saved full locked test JSON report to {report_json_path}")
    logger.info("Day 4 Benchmark Evaluation Complete!")

    return full_report


if __name__ == "__main__":
    evaluate_adni_locked_test()
