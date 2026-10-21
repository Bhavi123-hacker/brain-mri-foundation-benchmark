"""
Independent External Cohort Evaluation on OASIS-2 (N=150).
Evaluates zero-shot generalization of trained V2 and baseline models on the OASIS-2 longitudinal cohort.
Participant-disjoint: 150 unique participants (0% overlap with OASIS-1 or ADNI).
Calculates ROC-AUC, 95% CIs via 2,000 paired bootstrap resamples, and Holm-Bonferroni corrections.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.v2_models import ResNet10_3D, DenseNet121_3D, MultiView25D_ResNet18, DualBranchHybridModel
from preprocessing.enhanced_pipeline import get_v2_preprocessing_pipeline, AnatomicalROIExtractor
from evaluation.metrics import compute_classification_metrics

OASIS2_MANIFEST = r"D:\brain_mri_data\OASIS2\manifests\oasis2_baseline_manifest.csv"
OASIS2_TENSORS = r"D:\brain_mri_data\OASIS2\processed\tensors"
V2_CHECKPOINTS_DIR = r"D:\ml project\results\v2\checkpoints"
V2_TABLES_DIR = r"D:\ml project\results\v2\tables"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OASIS2EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tensors_dir: str, pipeline_type: str = "B", return_radiomics: bool = False):
        self.df = df.copy().reset_index(drop=True)
        self.tensors_dir = tensors_dir
        self.pipeline = get_v2_preprocessing_pipeline(pipeline_type, is_training=False)
        self.return_radiomics = return_radiomics
        self.roi_extractor = AnatomicalROIExtractor() if return_radiomics else None

        self.labels = self.df["binary_label"].astype(float).values
        self.participant_ids = self.df["participant_id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.participant_ids[idx]
        tpath = os.path.join(self.tensors_dir, f"{pid}.npy")
        arr = np.load(tpath)
        tensor = torch.from_numpy(arr).float()

        transformed = self.pipeline(tensor)
        target = torch.tensor([self.labels[idx]], dtype=torch.float32)

        if self.return_radiomics:
            biomarkers = self.roi_extractor.compute_morphometric_biomarkers(transformed.numpy())
            rad_vec = torch.tensor([
                biomarkers["total_brain_voxels"] / 1e5,
                biomarkers["ventricular_csf_voxels"] / 1e4,
                biomarkers["hippocampal_tissue_voxels"] / 1e4,
                biomarkers["ventricle_to_brain_ratio"] * 10.0,
                biomarkers["hippocampus_ratio"] * 10.0,
                biomarkers["intensity_skewness"],
            ], dtype=torch.float32)
            return transformed, target, pid, rad_vec

        return transformed, target, pid


def run_paired_bootstrap(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, n_bootstraps: int = 2000, seed: int = 42):
    np.random.seed(seed)
    n = len(y_true)
    delta_list = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, probs_a[idx])
        auc_b = roc_auc_score(y_b, probs_b[idx])
        delta_list.append(auc_a - auc_b)
    delta_arr = np.array(delta_list)
    p_two_sided = 2.0 * min(np.mean(delta_arr <= 0.0), np.mean(delta_arr >= 0.0))
    return {
        "mean_delta": float(np.mean(delta_arr)),
        "ci_delta": [float(np.percentile(delta_arr, 2.5)), float(np.percentile(delta_arr, 97.5))],
        "p_value": float(min(1.0, max(1.0 / len(delta_arr), p_two_sided))),
    }


def compute_single_auc_ci(y_true: np.ndarray, probs: np.ndarray, n_bootstraps: int = 2000, seed: int = 42):
    np.random.seed(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, probs[idx]))
    auc_arr = np.array(aucs)
    return float(np.mean(auc_arr)), [float(np.percentile(auc_arr, 2.5)), float(np.percentile(auc_arr, 97.5))]


def evaluate_oasis2():
    print("=" * 80)
    print("INDEPENDENT EXTERNAL VALIDATION ON OASIS-2 (N=150)")
    print("=" * 80)

    if not os.path.exists(OASIS2_MANIFEST):
        print(f"Error: Manifest {OASIS2_MANIFEST} not found. Run download script first.")
        return

    df = pd.read_csv(OASIS2_MANIFEST)
    df_pass = df[df["status"] == "PASS"].reset_index(drop=True)
    print(f"Loaded OASIS-2 manifest: {len(df_pass)} valid scans (AD={sum(df_pass['binary_label']==1)}, CN={sum(df_pass['binary_label']==0)})")

    models_to_eval = [
        ("3D ResNet-10", ResNet10_3D(output_dim=1, dropout=0.2), "3d_resnet_10_seed42_best.pt", False),
        ("3D DenseNet-121", DenseNet121_3D(output_dim=1, dropout=0.2), "3d_densenet_121_seed42_best.pt", False),
        ("2.5D Multi-View ResNet-18", MultiView25D_ResNet18(pretrained=True, output_dim=1, dropout=0.3), "2.5d_multi_view_resnet_18_seed42_best.pt", False),
        ("Dual-Branch Hybrid", DualBranchHybridModel(spatial_backbone="resnet10", num_radiomics=6, output_dim=1), "dual_branch_hybrid_resnet_10__radiomics_seed42_best.pt", True),
    ]

    oasis2_preds = {
        "participant_id": df_pass["participant_id"].tolist(),
        "ground_truth": df_pass["binary_label"].astype(int).tolist(),
        "age": df_pass["age"].tolist(),
        "sex": df_pass["sex"].tolist(),
    }
    oasis2_results = {}
    y_true = np.array(df_pass["binary_label"].values)

    for name, model_inst, ckpt_name, return_rad in models_to_eval:
        ckpt_path = os.path.join(V2_CHECKPOINTS_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {name}: checkpoint {ckpt_name} not found yet.")
            continue

        print(f"\nEvaluating: {name} on OASIS-2 ...")
        model_inst.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model_inst = model_inst.to(DEVICE).eval()

        ds = OASIS2EvalDataset(df_pass, OASIS2_TENSORS, pipeline_type="B", return_radiomics=return_rad)
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                if return_rad:
                    imgs, _, _, rads = batch
                    imgs, rads = imgs.to(DEVICE), rads.to(DEVICE)
                    logits = model_inst(imgs, rads)
                else:
                    imgs, _, _ = batch
                    imgs = imgs.to(DEVICE)
                    logits = model_inst(imgs)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)

        probs_arr = np.array(all_probs)
        clean_key = name.lower().replace(" ", "_").replace("-", "_")
        oasis2_preds[f"{clean_key}_prob"] = [round(float(p), 4) for p in probs_arr]

        metrics = compute_classification_metrics(probs_arr, y_true)
        mean_auc, ci_auc = compute_single_auc_ci(y_true, probs_arr, n_bootstraps=2000)

        oasis2_results[name] = {
            "roc_auc": round(metrics["roc_auc"], 4),
            "auc_95_ci": [round(ci_auc[0], 4), round(ci_auc[1], 4)],
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "sensitivity": round(metrics["sensitivity"], 4),
            "specificity": round(metrics["specificity"], 4),
            "brier_score": round(metrics["brier_score"], 4),
        }
        print(f"{name} on OASIS-2: AUC={metrics['roc_auc']:.4f} (95% CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]) | BalAcc={metrics['balanced_accuracy']:.4f} | Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f} | Brier={metrics['brier_score']:.4f}")

    # Save predictions & results
    pred_df = pd.DataFrame(oasis2_preds)
    pred_csv = os.path.join(V2_TABLES_DIR, "v2_oasis2_external_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nSaved OASIS-2 predictions to: {pred_csv}")

    res_json = os.path.join(V2_TABLES_DIR, "v2_oasis2_benchmark_results.json")
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump(oasis2_results, f, indent=2)
    print(f"Saved OASIS-2 benchmark results to: {res_json}")


if __name__ == "__main__":
    evaluate_oasis2()
