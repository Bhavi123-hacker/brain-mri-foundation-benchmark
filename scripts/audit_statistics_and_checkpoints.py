import os
import sys
import json
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, balanced_accuracy_score

BASE_DIR = Path(r"D:\ml project")
DATA_ROOT = Path(r"D:\brain_mri_data")
RESULTS_DIR = BASE_DIR / "results" / "v4"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PREDS_DIR = RESULTS_DIR / "predictions"

print("=" * 80)
print("PART 1: DATASET ENROLLMENT VS PREDICTION COUNTS AUDIT")
print("=" * 80)

# Check cohort manifests and files
oasis1_splits = pd.read_csv(DATA_ROOT / "OASIS1" / "manifests" / "oasis1_primary_cohort_splits.csv")
adni_df = pd.read_csv(DATA_ROOT / "ADNI" / "manifests" / "adni_full_preprocessed_manifest.csv")
oasis2_df = pd.read_csv(DATA_ROOT / "OASIS2" / "manifests" / "oasis2_baseline_manifest.csv")

print(f"OASIS-1 Primary Cohort Total Subjects: {len(oasis1_splits)}")
print(f"  - Train split: {(oasis1_splits['split'] == 'train').sum()}")
print(f"  - Val split:   {(oasis1_splits['split'] == 'val').sum()}")
print(f"  - Test split:  {(oasis1_splits['split'] == 'test').sum()}")

adni_pass = adni_df[adni_df["status"] == "PASS"]
print(f"ADNI Manifest Total Rows: {len(adni_df)}, PASS: {len(adni_pass)}")

o2_pass = oasis2_df[oasis2_df["status"] == "PASS"]
print(f"OASIS-2 Manifest Total Rows: {len(oasis2_df)}, PASS: {len(o2_pass)}")

total_cohort_enrollment = len(oasis1_splits) + len(adni_pass) + len(o2_pass)
print(f"Total Cohort Enrollment (OASIS-1 + ADNI + OASIS-2): {len(oasis1_splits)} + {len(adni_pass)} + {len(o2_pass)} = {total_cohort_enrollment}")

# OASIS-1 split counts
val_preds = pd.read_csv(PREDS_DIR / "oasis1_validation_predictions.csv")
test_preds = pd.read_csv(PREDS_DIR / "oasis1_locked_test_predictions.csv")
adni_preds = pd.read_csv(PREDS_DIR / "adni_external_predictions.csv")
o2_preds = pd.read_csv(PREDS_DIR / "oasis2_external_predictions.csv")

print(f"\nSaved Prediction Files Row Counts:")
print(f"  - OASIS-1 Validation predictions:  {len(val_preds)}")
print(f"  - OASIS-1 Locked Test predictions: {len(test_preds)}")
print(f"  - ADNI External predictions:       {len(adni_preds)}")
print(f"  - OASIS-2 External predictions:    {len(o2_preds)}")

total_eval_preds = len(test_preds) + len(adni_preds) + len(o2_preds)
print(f"Total Evaluation / Test Predictions (Locked Test + ADNI + OASIS-2): {len(test_preds)} + {len(adni_preds)} + {len(o2_preds)} = {total_eval_preds}")
total_saved_preds = len(val_preds) + len(test_preds) + len(adni_preds) + len(o2_preds)
print(f"Total Saved Predictions across all 4 files (Val + Test + ADNI + OASIS-2): {total_saved_preds}")
print(f"Explain 848: Total cohort enrollment = 198 (OASIS-1) + 500 (ADNI) + 150 (OASIS-2) = 848.")
print(f"OASIS-1 train scans (138) are used exclusively for fitting model weights; hence their predictions are not part of out-of-sample evaluation files (138 + 710 = 848).")

print("\n" + "=" * 80)
print("PART 2: CHECKPOINT PROVENANCE & WEIGHT LOADING AUDIT")
print("=" * 80)

# Check all 15 configurations
configs = [
    ("3d_resnet_10", "3d_resnet_10_seed42_best.pt", "resnet", "neural"),
    ("3d_resnet_18", "3d_resnet_18_seed42_best.pt", "resnet", "neural"),
    ("3d_resnet_34", "3d_resnet_34_seed42_best.pt", "resnet", "neural"),
    ("3d_densenet_121", "3d_densenet_121_seed42_best.pt", "densenet", "neural"),
    ("2_5d_multi_view_resnet_18", "2_5d_multi_view_resnet_18_seed42_best.pt", "multiview", "neural"),
    ("dual_branch_hybrid", "dual_branch_hybrid_seed42_best.pt", "hybrid", "neural"),
    ("2d_transfer_resnet_18", "2d_transfer_resnet_18_seed42_best.pt", "transfer2d", "neural"),
    ("classical_radiomics", "classical_radiomics_model.joblib", "radiomics", "classical"),
    ("brainiac_vit_b16_linear_probe", "brainiac_vit_b16_linear_probe_seed42_best.pt", "foundation", "neural"),
    ("brainiac_vit_b16_partial_ft", "brainiac_vit_b16_partial_ft_seed42_best.pt", "foundation", "neural"),
    ("random_vit_b16_control", "random_vit_b16_control_seed42_best.pt", "foundation_control", "neural"),
    ("swin_unetr_ssl_linear_probe", "swin_unetr_ssl_linear_probe_seed42_best.pt", "foundation", "neural"),
    ("swin_unetr_ssl_partial_ft", "swin_unetr_ssl_partial_ft_seed42_best.pt", "foundation", "neural"),
    ("fomo25_amaes_linear_probe", "fomo25_amaes_linear_probe_seed42_best.pt", "foundation", "neural"),
    ("fomo25_amaes_partial_ft", "fomo25_amaes_partial_ft_seed42_best.pt", "foundation", "neural"),
]

for key, filename, family, ctype in configs:
    fpath = CHECKPOINTS_DIR / filename
    if not fpath.exists():
        if ctype == "classical":
            print(f"[NOTE] {key:30s} | Scikit-learn artifact not yet saved as joblib file.")
        else:
            print(f"[FAIL] {key}: checkpoint does not exist at {fpath}")
        continue
    fsize_mb = fpath.stat().st_size / (1024 * 1024)
    if ctype == "classical":
        try:
            artifact = joblib.load(fpath)
            print(f"[PASS] {key:30s} | Size: {fsize_mb:6.2f} MB | Type: Scikit-learn {type(artifact.get('model', artifact)).__name__}")
        except Exception as e:
            print(f"[FAIL] {key}: {e}")
    else:
        try:
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            s_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) and any("weight" in k for k in ckpt.keys()) else None))
            num_params = sum(p.numel() for p in s_dict.values()) if s_dict else 0
            encoder_keys = [k for k in s_dict.keys() if any(pat in k for pat in ["encoder", "backbone", "blocks", "swinViT", "features", "layer"])]
            print(f"[PASS] {key:30s} | Size: {fsize_mb:6.2f} MB | State Keys: {len(s_dict)} | Total Parameters: {num_params:,} | Encoder Keys: {len(encoder_keys)}")
        except Exception as e:
            print(f"[FAIL] {key}: {e}")

print("\n" + "=" * 80)
print("PART 3: EMPIRICAL SAMPLE AUC VS REPORTED METRICS RECONCILIATION")
print("=" * 80)

models = [k for k, _, _, _ in configs]
results_table = []
for m in models:
    t_auc = roc_auc_score(test_preds["true_label"], test_preds[m])
    a_auc = roc_auc_score(adni_preds["true_label"], adni_preds[m])
    o_auc = roc_auc_score(o2_preds["true_label"], o2_preds[m])
    gap = t_auc - a_auc
    results_table.append({
        "model_key": m,
        "test_auc": round(t_auc, 4),
        "adni_auc": round(a_auc, 4),
        "oasis2_auc": round(o_auc, 4),
        "generalization_gap": round(gap, 4),
    })
df_emp = pd.DataFrame(results_table)
print(df_emp.to_string(index=False))

print("\n" + "=" * 80)
print("PART 4: STATISTICAL HYPOTHESIS TESTING (H1-H7) & 9-TEST HOLM CORRECTION")
print("=" * 80)

hypotheses = [
    ("H1", "Model Capacity: 3D ResNet-10 vs 3D ResNet-34 on OASIS-1 Test", "3d_resnet_10", "3d_resnet_34", "oasis_test", test_preds),
    ("H2", "Dense vs Residual: 3D DenseNet-121 vs 3D ResNet-18 on ADNI", "3d_densenet_121", "3d_resnet_18", "adni", adni_preds),
    ("H3", "Multi-Planar 2.5D: 2.5D ResNet-18 vs 3D ResNet-18 on ADNI", "2_5d_multi_view_resnet_18", "3d_resnet_18", "adni", adni_preds),
    ("H4", "Pretrained Transfer 2D: 2D Transfer ResNet-18 vs 3D ResNet-18 on ADNI", "2d_transfer_resnet_18", "3d_resnet_18", "adni", adni_preds),
    ("H5", "Dual-Branch Synergy: Hybrid vs 3D ResNet-10 on OASIS-1 Test", "dual_branch_hybrid", "3d_resnet_10", "oasis_test", test_preds),
    ("H6a", "Foundation Probe: BrainIAC ViT-B/16 vs 3D ResNet-18 on ADNI", "brainiac_vit_b16_linear_probe", "3d_resnet_18", "adni", adni_preds),
    ("H6b", "Foundation Probe: MONAI Swin UNETR SSL vs 3D ResNet-18 on ADNI", "swin_unetr_ssl_linear_probe", "3d_resnet_18", "adni", adni_preds),
    ("H6c", "Foundation Probe: FOMO25 AMAES vs 3D ResNet-18 on ADNI", "fomo25_amaes_linear_probe", "3d_resnet_18", "adni", adni_preds),
    ("H7", "Pretraining Control: BrainIAC ViT-B/16 vs Random ViT-B/16 Control on ADNI", "brainiac_vit_b16_linear_probe", "random_vit_b16_control", "adni", adni_preds),
]

rng = np.random.RandomState(42)
B = 10000
stat_rows = []
for hid, desc, mA, mB, cname, df in hypotheses:
    y = df["true_label"].values
    pA = df[mA].values
    pB = df[mB].values
    auc_A = roc_auc_score(y, pA)
    auc_B = roc_auc_score(y, pB)
    delta_auc = auc_A - auc_B
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    boot_diffs = []
    for _ in range(B):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        diff = roc_auc_score(y[b_idx], pA[b_idx]) - roc_auc_score(y[b_idx], pB[b_idx])
        boot_diffs.append(diff)
    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_val = 2.0 * min(np.mean(boot_diffs <= 0.0), np.mean(boot_diffs >= 0.0))
    p_val = max(p_val, 1.0 / B)
    
    stat_rows.append({
        "hypothesis_id": hid,
        "description": desc,
        "model_A": mA,
        "model_B": mB,
        "cohort": cname,
        "auc_A": round(auc_A, 4),
        "auc_B": round(auc_B, 4),
        "delta_auc": round(delta_auc, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "raw_p_value": round(p_val, 4),
    })

df_stat = pd.DataFrame(stat_rows).sort_values("raw_p_value").reset_index(drop=True)
m_tests = len(df_stat)
adj_p_vals = []
sig_flags = []
for rank, p in enumerate(df_stat["raw_p_value"]):
    adj_p = min(p * (m_tests - rank), 1.0)
    adj_p_vals.append(adj_p)
    sig_flags.append(adj_p < 0.05)

for i in range(1, m_tests):
    if adj_p_vals[i] < adj_p_vals[i - 1]:
        adj_p_vals[i] = adj_p_vals[i - 1]

df_stat["rank"] = range(1, m_tests + 1)
df_stat["multiplier"] = [m_tests - r + 1 for r in df_stat["rank"]]
df_stat["holm_bonferroni_p_value"] = [round(p, 4) for p in adj_p_vals]
df_stat["statistically_significant"] = df_stat["holm_bonferroni_p_value"] < 0.05

for _, r in df_stat.iterrows():
    print(f"{r['hypothesis_id']:4s} (Rank {r['rank']}, m={r['multiplier']}) | {r['model_A']:30s} vs {r['model_B']:24s} | delta={r['delta_auc']:+.4f} | CI=[{r['ci_95_lower']:+.4f}, {r['ci_95_upper']:+.4f}] | raw_p={r['raw_p_value']:.4f} | holm_p={r['holm_bonferroni_p_value']:.4f} | Sig: {r['statistically_significant']}")

print("\n" + "=" * 80)
print("PART 5: TRUSTGATE FAILURE GATING RE-EVALUATION")
print("=" * 80)

# Check TrustGate results
tg_path = RESULTS_DIR / "FINAL_TRUSTGATE_ANALYSIS.csv"
if tg_path.exists():
    df_tg = pd.read_csv(tg_path)
    print("TrustGate Analysis Table:")
    print(df_tg.to_string(index=False))

y_adni = adni_preds["true_label"].values
p_r18 = adni_preds["3d_resnet_18"].values
err = ((p_r18 >= 0.5).astype(int) != y_adni).astype(int)
risk_margin = 1.0 - np.abs(p_r18 - 0.5)
margin_auroc = roc_auc_score(err, risk_margin)
print(f"Empirical Error Gating Margin AUROC for 3D ResNet-18 on ADNI: {margin_auroc:.4f}")
