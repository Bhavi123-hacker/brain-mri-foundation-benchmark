"""
Priority 4: Bidirectional Cross-Dataset Generalization Benchmark.
Evaluates:
  1. Direction 1: OASIS-1 -> ADNI MPRAGE
  2. Direction 2: ADNI MPRAGE -> OASIS-1
Models:
  - 3D ResNet-18
  - BrainIAC Linear Probe
  - FOMO25 Winner Linear Probe
  - MONAI Swin UNETR SSL Linear Probe
Seeds: 42, 43, 44, 45, 46
Outputs:
  - results/extensions/cross_dataset/cross_dataset_results.csv
  - results/extensions/cross_dataset/cross_dataset_results.json
  - results/extensions/figures/figure_CD1_oasis_to_adni.png
  - results/extensions/figures/figure_CD2_adni_to_oasis.png
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


from preprocessing.cache_manager import PreprocessedCacheManager


def load_dataset_scans(splits_csv: str, cache_dir: str, dataset_name: str):
    """Load scans and labels for a dataset."""
    df = pd.read_csv(splits_csv)
    if "ID" in df.columns:
        sid_col = "ID"
        file_col = "mri_file"
        df = df[~df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)
    else:
        sid_col = "subject_id"
        file_col = "nifti_path"

    test_df = df[df["split"] == "test"].sort_values(sid_col).reset_index(drop=True)

    cache_mgr = PreprocessedCacheManager(cache_dir=cache_dir)
    tensors = []
    labels = []
    sids = []

    for _, row in test_df.iterrows():
        sid = str(row[sid_col]).strip()
        if dataset_name == "OASIS":
            fpath = os.path.join("D:/OASIS1_PILOT", row["mri_file"])
            tensor = cache_mgr.get(fpath, transform_id="oasis1_zscore_128")
        else:
            npy_path = os.path.join(cache_dir, f"{sid}.npy")
            if os.path.exists(npy_path):
                tensor = torch.from_numpy(np.load(npy_path))
            else:
                tensor = cache_mgr.get(row["nifti_path"], transform_id="adni_zscore_128")

        assert tensor is not None, f"Failed to load scan for {sid}"
        tensors.append(tensor.to(torch.float32))
        labels.append(float(row["label"]))
        sids.append(sid)

    stacked_tensors = torch.stack(tensors)  # (N, 1, 128, 128, 128)
    labels_arr = np.array(labels, dtype=np.float32)
    return stacked_tensors, labels_arr, sids


def run_cross_dataset_evaluation(
    oasis_splits: str = "D:/OASIS1_PILOT/pilot_splits.csv",
    oasis_cache: str = "data/cache/oasis1_pilot",
    adni_splits: str = "results/adni_cn_ad_split.csv",
    adni_cache: str = "data/cache/adni_cn_ad",
    output_dir: str = "results/extensions/cross_dataset",
    figures_dir: str = "results/extensions/figures",
    seeds: list = [42, 43, 44, 45, 46],
):
    logger = get_logger("CrossDatasetBenchmark")
    logger.info("=================================================================")
    logger.info("Starting Priority 4: Bidirectional Cross-Dataset Generalization")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load target test cohorts
    oasis_test_x, oasis_test_y, oasis_sids = load_dataset_scans(oasis_splits, oasis_cache, "OASIS")
    adni_test_x, adni_test_y, adni_sids = load_dataset_scans(adni_splits, adni_cache, "ADNI")

    logger.info(f"Loaded OASIS test cohort: {len(oasis_test_x)} subjects")
    logger.info(f"Loaded ADNI test cohort: {len(adni_test_x)} subjects")

    models = [
        {
            "name": "3D ResNet-18",
            "slug": "resnet18",
            "builder": lambda: build_resnet18_3d(output_dim=1),
            "oasis_ckpt_template": "results/checkpoints/oasis1_pilot_resnet18_best.pt",  # Seed 42 baseline
            "adni_ckpt_template": "results/checkpoints/adni_resnet18_seed{seed}_best.pt",
        },
        {
            "name": "BrainIAC",
            "slug": "brainiac",
            "builder": lambda: BrainIACAdapter(output_dim=1, weights_path="weights/brainiac_vit_base.pt", adaptation_mode="linear_probe"),
            "oasis_ckpt_template": "results/checkpoints/oasis1_pilot_brainiac_linear_probe_best.pt",
            "adni_ckpt_template": "results/checkpoints/adni_brainiac_linear_seed{seed}_best.pt",
        },
        {
            "name": "FOMO25 Winner",
            "slug": "fomo25",
            "builder": lambda: FOMO25Adapter(output_dim=1, weights_path="weights/fomo25_winner_encoder.pt", adaptation_mode="linear_probe"),
            "oasis_ckpt_template": "results/checkpoints/oasis1_pilot_fomo25_linear_probe_best.pt",
            "adni_ckpt_template": "results/checkpoints/adni_fomo25_linear_seed{seed}_best.pt",
        },
        {
            "name": "MONAI Swin UNETR SSL",
            "slug": "swin_unetr",
            "builder": lambda: SwinUNETRSSLAdapter(output_dim=1, weights_path="weights/model_swinvit.pt", adaptation_mode="linear_probe"),
            "oasis_ckpt_template": "results/checkpoints/oasis1_pilot_swin_unetr_linear_probe_best.pt",
            "adni_ckpt_template": "results/checkpoints/adni_swin_unetr_linear_seed{seed}_best.pt",
        },
    ]

    all_records = []

    # -------------------------------------------------------------------------
    # DIRECTION 1: ADNI -> OASIS (Evaluate models trained on ADNI against OASIS)
    # -------------------------------------------------------------------------
    logger.info("\n--- Direction 1: Trained on ADNI -> Evaluated on External OASIS ---")
    for m in models:
        for s in seeds:
            ckpt_path = m["adni_ckpt_template"].format(seed=s)
            if not os.path.exists(ckpt_path):
                continue
            model = m["builder"]().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(sd, strict=False)
            model.eval()

            with torch.no_grad():
                logits = model(oasis_test_x.to(device))
                probs = torch.sigmoid(logits).cpu().squeeze(-1).numpy()

            metrics = compute_classification_metrics(oasis_test_y, probs)
            rec = {
                "source_cohort": "ADNI MPRAGE (3.0T)",
                "target_cohort": "OASIS-1 Pilot (1.5T)",
                "direction": "ADNI -> OASIS",
                "model": m["name"],
                "slug": m["slug"],
                "seed": s,
                "target_test_roc_auc": metrics["roc_auc"],
                "target_test_accuracy": metrics["accuracy"],
                "target_test_balanced_acc": metrics["balanced_accuracy"],
                "target_test_f1": metrics["f1"],
                "target_test_brier": metrics["brier_score"],
            }
            all_records.append(rec)
            logger.info(f"  [ADNI->OASIS] {m['name']:22s} Seed {s:2d} -> Ext AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    # -------------------------------------------------------------------------
    # DIRECTION 2: OASIS -> ADNI (Evaluate models trained on OASIS against ADNI)
    # -------------------------------------------------------------------------
    logger.info("\n--- Direction 2: Trained on OASIS -> Evaluated on External ADNI ---")
    for m in models:
        ckpt_path = m["oasis_ckpt_template"]
        if not os.path.exists(ckpt_path):
            continue
        model = m["builder"]().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()

        with torch.no_grad():
            logits = model(adni_test_x.to(device))
            probs = torch.sigmoid(logits).cpu().squeeze(-1).numpy()

        metrics = compute_classification_metrics(adni_test_y, probs)
        rec = {
            "source_cohort": "OASIS-1 Pilot (1.5T)",
            "target_cohort": "ADNI MPRAGE (3.0T)",
            "direction": "OASIS -> ADNI",
            "model": m["name"],
            "slug": m["slug"],
            "seed": 42,
            "target_test_roc_auc": metrics["roc_auc"],
            "target_test_accuracy": metrics["accuracy"],
            "target_test_balanced_acc": metrics["balanced_accuracy"],
            "target_test_f1": metrics["f1"],
            "target_test_brier": metrics["brier_score"],
        }
        all_records.append(rec)
        logger.info(f"  [OASIS->ADNI] {m['name']:22s} -> Ext AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    results_df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "cross_dataset_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved cross-dataset results to {csv_path}")

    json_path = os.path.join(output_dir, "cross_dataset_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"Saved cross-dataset JSON to {json_path}")

    # -------------------------------------------------------------------------
    # GENERATE FIGURES CD1 & CD2
    # -------------------------------------------------------------------------
    colors = {"3D ResNet-18": "#1f77b4", "BrainIAC": "#ff7f0e", "FOMO25 Winner": "#2ca02c", "MONAI Swin UNETR SSL": "#d62728"}

    # Figure CD1: OASIS -> ADNI
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sub1 = results_df[results_df["direction"] == "OASIS -> ADNI"]
    bars1 = ax.bar(sub1["model"], sub1["target_test_roc_auc"], color=[colors[m] for m in sub1["model"]], edgecolor="black", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle=":", label="Chance (0.50)")
    ax.set_title("Cross-Dataset Generalization: OASIS-1 -> ADNI MPRAGE", fontsize=12, fontweight="bold")
    ax.set_ylabel("External Test ROC-AUC", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., h + 0.02, f"{h:.3f}", ha="center", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig1_path = os.path.join(figures_dir, "figure_CD1_oasis_to_adni.png")
    plt.tight_layout(); plt.savefig(fig1_path); plt.close()
    print(f"Saved Figure CD1 to {fig1_path}")

    # Figure CD2: ADNI -> OASIS (Mean ± SD across 5 seeds)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sub2 = results_df[results_df["direction"] == "ADNI -> OASIS"]
    m_names = ["3D ResNet-18", "BrainIAC", "FOMO25 Winner", "MONAI Swin UNETR SSL"]
    means = [sub2[sub2["model"] == m]["target_test_roc_auc"].mean() for m in m_names]
    stds = [sub2[sub2["model"] == m]["target_test_roc_auc"].std() for m in m_names]

    bars2 = ax.bar(m_names, means, yerr=stds, color=[colors[m] for m in m_names], capsize=5, edgecolor="black", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle=":", label="Chance (0.50)")
    ax.set_title("Cross-Dataset Generalization: ADNI MPRAGE -> OASIS-1", fontsize=12, fontweight="bold")
    ax.set_ylabel("External Test ROC-AUC (Mean ± SD)", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    for i, b in enumerate(bars2):
        h = means[i]
        ax.text(b.get_x() + b.get_width()/2., h + stds[i] + 0.02, f"{h:.3f}", ha="center", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig2_path = os.path.join(figures_dir, "figure_CD2_adni_to_oasis.png")
    plt.tight_layout(); plt.savefig(fig2_path); plt.close()
    print(f"Saved Figure CD2 to {fig2_path}")

    return results_df

if __name__ == "__main__":
    run_cross_dataset_evaluation()
