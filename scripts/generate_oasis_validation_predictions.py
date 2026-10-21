"""
scripts/generate_oasis_validation_predictions.py
Generates genuine participant-aligned predictions on the 30 OASIS-1 validation scans
from the authentic model checkpoints. Strictly enforces participant-ID alignment.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.baseline.resnet3d import build_resnet18_3d, build_resnet34_3d
from models.v2_models import ResNet10_3D, DenseNet121_3D, MultiView25D_ResNet18, DualBranchHybridModel
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from scripts.train_2d_slice_transfer_learning import TransferResNet2D, ParticipantSliceDataset, evaluate_participant_level
from scripts.run_v2_experiment_campaign import AnatomicalROIExtractor, evaluate_cohort_fast
from evaluation.metrics import compute_classification_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
RUNS_CKPT_DIR = r"D:\brain_mri_data\benchmark_runs\checkpoints"
V2_CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "v2", "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "v3")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 80)
    print("GENERATING GENUINE OASIS-1 VALIDATION PREDICTIONS FROM AUTHENTIC CHECKPOINTS")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # 1. Load validation split
    splits_df = pd.read_csv(OASIS_SPLITS_CSV)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
    val_sids = val_df["session_id"].tolist()
    val_targets = (val_df["cdr"] > 0).astype(np.float32).values
    val_xs = [np.load(os.path.join(OASIS_TENSORS_DIR, f"{sid}.npy")) for sid in val_sids]

    assert len(val_sids) == 30, f"Expected 30 validation samples, got {len(val_sids)}"
    assert pd.Series(val_sids).is_unique, "Validation session IDs must be unique"

    val_preds_dict = {
        "session_id": val_sids,
        "ground_truth": val_targets.astype(int),
    }

    # 2. ResNet-10 (V2)
    m_res10 = ResNet10_3D(output_dim=1).to(DEVICE)
    m_res10.load_state_dict(torch.load(os.path.join(V2_CKPT_DIR, "3d_resnet_10_seed42_best.pt"), map_location=DEVICE))
    p_res10, met10 = evaluate_cohort_fast(m_res10, val_xs, val_targets, DEVICE)
    val_preds_dict["3d_resnet_10_prob"] = [round(float(p), 4) for p in p_res10]
    print(f"3D ResNet-10 Val AUC: {met10['roc_auc']:.4f}")

    # 3. DenseNet-121 (V2)
    m_dense = DenseNet121_3D(output_dim=1).to(DEVICE)
    m_dense.load_state_dict(torch.load(os.path.join(V2_CKPT_DIR, "3d_densenet_121_seed42_best.pt"), map_location=DEVICE))
    p_dense, met_dense = evaluate_cohort_fast(m_dense, val_xs, val_targets, DEVICE)
    val_preds_dict["3d_densenet_121_prob"] = [round(float(p), 4) for p in p_dense]
    print(f"3D DenseNet-121 Val AUC: {met_dense['roc_auc']:.4f}")

    # 4. 2.5D Multi-View ResNet-18 (V2)
    m_mv = MultiView25D_ResNet18(pretrained=False, output_dim=1).to(DEVICE)
    m_mv.load_state_dict(torch.load(os.path.join(V2_CKPT_DIR, "2.5d_multi_view_resnet_18_seed42_best.pt"), map_location=DEVICE))
    p_mv, met_mv = evaluate_cohort_fast(m_mv, val_xs, val_targets, DEVICE)
    val_preds_dict["2.5d_multi_view_resnet_18_prob"] = [round(float(p), 4) for p in p_mv]
    print(f"2.5D Multi-View ResNet-18 Val AUC: {met_mv['roc_auc']:.4f}")

    # 5. Dual-Branch Hybrid (V2)
    extractor = AnatomicalROIExtractor()
    m_hyb = DualBranchHybridModel(spatial_backbone="resnet10", num_radiomics=6, output_dim=1).to(DEVICE)
    m_hyb.load_state_dict(torch.load(os.path.join(V2_CKPT_DIR, "dual_branch_hybrid_seed42_best.pt"), map_location=DEVICE))
    p_hyb, met_hyb = evaluate_cohort_fast(m_hyb, val_xs, val_targets, DEVICE, has_radiomics=True, roi_extractor=extractor)
    val_preds_dict["dual_branch_hybrid_prob"] = [round(float(p), 4) for p in p_hyb]
    print(f"Dual-Branch Hybrid Val AUC: {met_hyb['roc_auc']:.4f}")

    # 6. MONAI Swin UNETR SSL
    m_swin = SwinUNETRSSLAdapter(weights_path=os.path.join(PROJECT_ROOT, "weights", "model_swinvit.pt"), adaptation_mode="linear_probe", dropout=0.3).to(DEVICE)
    m_swin.load_state_dict(torch.load(os.path.join(RUNS_CKPT_DIR, "swin_unetr_linear_probe_best.pt"), map_location=DEVICE))
    m_swin.eval()
    p_swin = []
    with torch.no_grad():
        for i in range(0, len(val_xs), 4):
            batch = np.stack(val_xs[i:i+4])
            bt = torch.from_numpy(batch).to(DEVICE)
            with torch.amp.autocast("cuda"):
                logits = m_swin(bt)
            p_swin.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    val_preds_dict["swin_unetr_ssl_prob"] = [round(float(p), 4) for p in p_swin]
    met_swin = compute_classification_metrics(val_targets, np.array(p_swin))
    print(f"MONAI Swin UNETR SSL Val AUC: {met_swin['roc_auc']:.4f}")

    # 7. FOMO25 AMAES ResEnc-B (Load from authentic oasis_val_predictions.csv)
    fomo_val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "fomo_linear_probe", "oasis_val_predictions.csv"))
    # Align by participant_id
    val_pids = [s.split("_MR")[0] for s in val_sids]
    fomo_val_aligned = pd.DataFrame({"participant_id": val_pids}).merge(fomo_val_df, on="participant_id", how="left")
    assert not fomo_val_aligned["fomo_amaes_prob"].isnull().any(), "FOMO validation alignment failed"
    p_fomo = fomo_val_aligned["fomo_amaes_prob"].values
    val_preds_dict["fomo_amaes_prob"] = [round(float(p), 4) for p in p_fomo]
    met_fomo = compute_classification_metrics(val_targets, np.array(p_fomo))
    print(f"FOMO25 AMAES Val AUC: {met_fomo['roc_auc']:.4f}")

    # 8. 3D ResNet-18 (Baseline)
    m18 = build_resnet18_3d(output_dim=1).to(DEVICE)
    m18.load_state_dict(torch.load(os.path.join(RUNS_CKPT_DIR, "resnet18_seed42_best.pt"), map_location=DEVICE))
    m18.eval()
    p18 = []
    with torch.no_grad():
        for i in range(0, len(val_xs), 4):
            batch = np.stack(val_xs[i:i+4])
            bt = torch.from_numpy(batch).to(DEVICE)
            p18.extend(torch.sigmoid(m18(bt)).cpu().numpy().flatten())
    val_preds_dict["resnet18_prob"] = [round(float(p), 4) for p in p18]
    met18 = compute_classification_metrics(val_targets, np.array(p18))
    print(f"3D ResNet-18 Val AUC: {met18['roc_auc']:.4f}")

    # 9. 3D ResNet-34 (Baseline)
    m34 = build_resnet34_3d(output_dim=1).to(DEVICE)
    m34.load_state_dict(torch.load(os.path.join(RUNS_CKPT_DIR, "resnet34_seed42_best.pt"), map_location=DEVICE))
    m34.eval()
    p34 = []
    with torch.no_grad():
        for i in range(0, len(val_xs), 4):
            batch = np.stack(val_xs[i:i+4])
            bt = torch.from_numpy(batch).to(DEVICE)
            p34.extend(torch.sigmoid(m34(bt)).cpu().numpy().flatten())
    val_preds_dict["resnet34_prob"] = [round(float(p), 4) for p in p34]
    met34 = compute_classification_metrics(val_targets, np.array(p34))
    print(f"3D ResNet-34 Val AUC: {met34['roc_auc']:.4f}")

    # 10. 2D Transfer ResNet-18 (Slice-Aggregated)
    m2d = TransferResNet2D(adaptation_mode="partial_finetune").to(DEVICE)
    m2d.load_state_dict(torch.load(os.path.join(RUNS_CKPT_DIR, "resnet2d_partial_finetune_best.pt"), map_location=DEVICE))
    val_ds_2d = ParticipantSliceDataset(val_df, OASIS_TENSORS_DIR, is_train=False)
    met2d, agg_df_2d = evaluate_participant_level(m2d, val_ds_2d)
    # Align by session_id
    agg_df_2d = agg_df_2d.rename(columns={"subject_id": "session_id"})
    merged_2d = pd.DataFrame({"session_id": val_sids}).merge(agg_df_2d, on="session_id", how="left")
    assert not merged_2d["prob_mean"].isnull().any(), "2D validation alignment failed"
    p2d = merged_2d["prob_mean"].values
    val_preds_dict["resnet2d_transfer_prob"] = [round(float(p), 4) for p in p2d]
    print(f"2D Transfer ResNet-18 Val AUC: {met2d['roc_auc']:.4f}")

    # 11. Classical Radiomics (Linear SVM)
    with open(os.path.join(RUNS_CKPT_DIR, "classical_radiomics_best.pkl"), "rb") as f:
        clf_rad = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, "D:/brain_mri_data/benchmark_runs/tables/extracted_radiomics_features.pkl" if os.path.exists("D:/brain_mri_data/benchmark_runs/tables/extracted_radiomics_features.pkl") else "D:/brain_mri_data/benchmark_runs/tables/extracted_radiomics_features.pkl"), "rb") as f:
        rad_data = pickle.load(f)
    X_val, y_val_rad, sids_val_rad = rad_data["val"]
    # Verify alignment of sids
    assert list(sids_val_rad) == val_sids, "Radiomics validation IDs do not match val_sids"
    X_val_scaled = clf_rad["scaler"].transform(X_val)
    probs_rad = clf_rad["model"].predict_proba(X_val_scaled)[:, 1]
    val_preds_dict["classical_radiomics_prob"] = [round(float(p), 4) for p in probs_rad]
    met_rad = compute_classification_metrics(val_targets, probs_rad)
    print(f"Classical Radiomics Val AUC: {met_rad['roc_auc']:.4f}")

    # 12. Save DataFrame
    out_df = pd.DataFrame(val_preds_dict)
    out_csv = os.path.join(OUT_DIR, "v3_oasis_validation_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved authentic OASIS-1 validation predictions (N={len(out_df)}) to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
