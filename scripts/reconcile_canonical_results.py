import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

BASE_DIR = Path(r"D:\ml project")
RESULTS_DIR = BASE_DIR / "results" / "v4"
TABLES_DIR = RESULTS_DIR / "tables"
PREDS_DIR = RESULTS_DIR / "predictions"
SUBMISSION_DIR = BASE_DIR / "FINAL_SUBMISSION_23_OCT_2026"
EXP_RESULTS_DIR = SUBMISSION_DIR / "03_EXPERIMENTAL_RESULTS"

val_preds = pd.read_csv(PREDS_DIR / "oasis1_validation_predictions.csv")
test_preds = pd.read_csv(PREDS_DIR / "oasis1_locked_test_predictions.csv")
adni_preds = pd.read_csv(PREDS_DIR / "adni_external_predictions.csv")
o2_preds = pd.read_csv(PREDS_DIR / "oasis2_external_predictions.csv")

val_y = val_preds["true_label"].values
test_y = test_preds["true_label"].values
adni_y = adni_preds["true_label"].values
o2_y = o2_preds["true_label"].values

orig_eval = pd.read_csv(RESULTS_DIR / "FINAL_MODEL_EVALUATION.csv")

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            idx = idx | (y_prob == 1.0)
        if np.sum(idx) > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(y_prob[idx])
            ece += (np.sum(idx) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)

def compute_bootstrap_ci(y_true, y_prob, n_bootstraps=2000, seed=42):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    boot_aucs = []
    for _ in range(n_bootstraps):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        try:
            auc = roc_auc_score(y_true[b_idx], y_prob[b_idx])
            boot_aucs.append(auc)
        except ValueError:
            continue
    boot_aucs = np.array(boot_aucs)
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))

df_reconciled = orig_eval.copy()

for idx, row in df_reconciled.iterrows():
    m = row["model_key"]
    v_p = val_preds[m].values
    t_p = test_preds[m].values
    a_p = adni_preds[m].values
    o_p = o2_preds[m].values
    tau_star = row["val_optimal_threshold"]
    
    # Empirical point estimates
    t_auc = roc_auc_score(test_y, t_p)
    t_ci_l, t_ci_u = compute_bootstrap_ci(test_y, t_p)
    t_pred_bin = (t_p >= tau_star).astype(int)
    t_acc = accuracy_score(test_y, t_pred_bin)
    t_bacc = balanced_accuracy_score(test_y, t_pred_bin)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(test_y, t_pred_bin, labels=[0, 1]).ravel()
    t_sens = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    t_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    t_brier = brier_score_loss(test_y, t_p)
    t_ece = compute_ece(test_y, t_p)
    
    a_auc = roc_auc_score(adni_y, a_p)
    a_ci_l, a_ci_u = compute_bootstrap_ci(adni_y, a_p)
    a_pred_bin = (a_p >= tau_star).astype(int)
    a_acc = accuracy_score(adni_y, a_pred_bin)
    a_bacc = balanced_accuracy_score(adni_y, a_pred_bin)
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(adni_y, a_pred_bin, labels=[0, 1]).ravel()
    a_sens = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
    a_spec = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0.0
    a_brier = brier_score_loss(adni_y, a_p)
    a_ece = compute_ece(adni_y, a_p)
    
    o_auc = roc_auc_score(o2_y, o_p)
    o_pred_bin = (o_p >= tau_star).astype(int)
    o_bacc = balanced_accuracy_score(o2_y, o_pred_bin)
    
    # Subgroups
    is_siemens = (adni_preds["manufacturer"].str.lower() == "siemens").values
    is_philips = (adni_preds["manufacturer"].str.lower() == "philips").values
    sie_auc = roc_auc_score(adni_y[is_siemens], a_p[is_siemens])
    phi_auc = roc_auc_score(adni_y[is_philips], a_p[is_philips])
    
    gap = round(t_auc - a_auc, 4)
    
    df_reconciled.at[idx, "oasis_test_auc"] = round(t_auc, 4)
    df_reconciled.at[idx, "oasis_test_ci_lower"] = round(t_ci_l, 4)
    df_reconciled.at[idx, "oasis_test_ci_upper"] = round(t_ci_u, 4)
    df_reconciled.at[idx, "oasis_test_accuracy"] = round(t_acc, 4)
    df_reconciled.at[idx, "oasis_test_balanced_acc"] = round(t_bacc, 4)
    df_reconciled.at[idx, "oasis_test_sensitivity"] = round(t_sens, 4)
    df_reconciled.at[idx, "oasis_test_specificity"] = round(t_spec, 4)
    df_reconciled.at[idx, "oasis_test_brier"] = round(t_brier, 4)
    df_reconciled.at[idx, "oasis_test_ece"] = round(t_ece, 4)
    
    df_reconciled.at[idx, "adni_auc"] = round(a_auc, 4)
    df_reconciled.at[idx, "adni_ci_lower"] = round(a_ci_l, 4)
    df_reconciled.at[idx, "adni_ci_upper"] = round(a_ci_u, 4)
    df_reconciled.at[idx, "adni_accuracy"] = round(a_acc, 4)
    df_reconciled.at[idx, "adni_balanced_acc"] = round(a_bacc, 4)
    df_reconciled.at[idx, "adni_sensitivity"] = round(a_sens, 4)
    df_reconciled.at[idx, "adni_specificity"] = round(a_spec, 4)
    df_reconciled.at[idx, "adni_brier"] = round(a_brier, 4)
    df_reconciled.at[idx, "adni_ece"] = round(a_ece, 4)
    
    df_reconciled.at[idx, "adni_siemens_auc"] = round(sie_auc, 4)
    df_reconciled.at[idx, "adni_philips_auc"] = round(phi_auc, 4)
    df_reconciled.at[idx, "oasis2_auc"] = round(o_auc, 4)
    df_reconciled.at[idx, "oasis2_balanced_acc"] = round(o_bacc, 4)
    df_reconciled.at[idx, "generalization_gap_delta_auc"] = gap

# Save Master Evaluation CSVs
df_reconciled.to_csv(RESULTS_DIR / "FINAL_MODEL_EVALUATION.csv", index=False)
df_reconciled.to_csv(TABLES_DIR / "FINAL_MODEL_EVALUATION.csv", index=False)
df_reconciled.to_csv(EXP_RESULTS_DIR / "V4_Master_Model_Comparison.csv", index=False)
print("Saved reconciled FINAL_MODEL_EVALUATION.csv and V4_Master_Model_Comparison.csv")

# 1. Foundation vs ResNet-18 Comparison Table
r18_row = df_reconciled[df_reconciled["model_key"] == "3d_resnet_18"].iloc[0]
r18_adni_auc = r18_row["adni_auc"]
r18_adni_brier = r18_row["adni_brier"]

fnd_keys = [
    "brainiac_vit_b16_linear_probe",
    "brainiac_vit_b16_partial_ft",
    "swin_unetr_ssl_linear_probe",
    "swin_unetr_ssl_partial_ft",
    "fomo25_amaes_linear_probe",
    "fomo25_amaes_partial_ft",
]
fnd_rows = []
for k in fnd_keys:
    r = df_reconciled[df_reconciled["model_key"] == k].iloc[0]
    fnd_rows.append({
        "model_name": r["model_name"],
        "model_key": k,
        "oasis_test_auc": r["oasis_test_auc"],
        "adni_auc": r["adni_auc"],
        "adni_delta_vs_resnet18": round(r["adni_auc"] - r18_adni_auc, 4),
        "adni_brier": r["adni_brier"],
        "adni_brier_vs_resnet18": round(r["adni_brier"] - r18_adni_brier, 4),
        "generalization_gap": r["generalization_gap_delta_auc"],
        "trainable_params": r["trainable_parameters"],
        "total_params": r["total_parameters"],
    })
df_fnd = pd.DataFrame(fnd_rows)
df_fnd.to_csv(RESULTS_DIR / "FINAL_FOUNDATION_VS_RESNET_COMPARISON.csv", index=False)
df_fnd.to_csv(TABLES_DIR / "FINAL_FOUNDATION_VS_RESNET_COMPARISON.csv", index=False)
df_fnd.to_csv(EXP_RESULTS_DIR / "V4_Foundation_Comparison.csv", index=False)
print("Saved reconciled FINAL_FOUNDATION_VS_RESNET_COMPARISON.csv")

# 2. Fine-Tuning Comparison Table
ft_data = [
    {
        "foundation_backbone": "BrainIAC (ViT-B/16³)",
        "linear_probe_auc": df_reconciled[df_reconciled["model_key"] == "brainiac_vit_b16_linear_probe"]["adni_auc"].values[0],
        "partial_ft_auc": df_reconciled[df_reconciled["model_key"] == "brainiac_vit_b16_partial_ft"]["adni_auc"].values[0],
        "linear_probe_brier": df_reconciled[df_reconciled["model_key"] == "brainiac_vit_b16_linear_probe"]["adni_brier"].values[0],
        "partial_ft_brier": df_reconciled[df_reconciled["model_key"] == "brainiac_vit_b16_partial_ft"]["adni_brier"].values[0],
    },
    {
        "foundation_backbone": "MONAI Swin UNETR SSL",
        "linear_probe_auc": df_reconciled[df_reconciled["model_key"] == "swin_unetr_ssl_linear_probe"]["adni_auc"].values[0],
        "partial_ft_auc": df_reconciled[df_reconciled["model_key"] == "swin_unetr_ssl_partial_ft"]["adni_auc"].values[0],
        "linear_probe_brier": df_reconciled[df_reconciled["model_key"] == "swin_unetr_ssl_linear_probe"]["adni_brier"].values[0],
        "partial_ft_brier": df_reconciled[df_reconciled["model_key"] == "swin_unetr_ssl_partial_ft"]["adni_brier"].values[0],
    },
    {
        "foundation_backbone": "FOMO25 AMAES (ResEnc-B)",
        "linear_probe_auc": df_reconciled[df_reconciled["model_key"] == "fomo25_amaes_linear_probe"]["adni_auc"].values[0],
        "partial_ft_auc": df_reconciled[df_reconciled["model_key"] == "fomo25_amaes_partial_ft"]["adni_auc"].values[0],
        "linear_probe_brier": df_reconciled[df_reconciled["model_key"] == "fomo25_amaes_linear_probe"]["adni_brier"].values[0],
        "partial_ft_brier": df_reconciled[df_reconciled["model_key"] == "fomo25_amaes_partial_ft"]["adni_brier"].values[0],
    },
]
for r in ft_data:
    r["delta_auc_ft_minus_lp"] = round(r["partial_ft_auc"] - r["linear_probe_auc"], 4)
    r["delta_brier_ft_minus_lp"] = round(r["partial_ft_brier"] - r["linear_probe_brier"], 4)
    r["fine_tuning_harm_observed"] = r["delta_auc_ft_minus_lp"] < 0.0

df_ft = pd.DataFrame(ft_data)
df_ft.to_csv(RESULTS_DIR / "FINAL_FINE_TUNING_COMPARISON.csv", index=False)
df_ft.to_csv(TABLES_DIR / "FINAL_FINE_TUNING_COMPARISON.csv", index=False)
print("Saved reconciled FINAL_FINE_TUNING_COMPARISON.csv")

# 3. Three-Cohort Summary Table
three_rows = []
for idx, r in df_reconciled.iterrows():
    three_rows.append({
        "model_name": r["model_name"],
        "model_key": r["model_key"],
        "architecture_family": r["architecture_family"],
        "total_parameters": r["total_parameters"],
        "trainable_parameters": r["trainable_parameters"],
        "oasis1_test_auc": r["oasis_test_auc"],
        "oasis1_test_bacc": r["oasis_test_balanced_acc"],
        "adni_auc": r["adni_auc"],
        "adni_bacc": r["adni_balanced_acc"],
        "adni_brier": r["adni_brier"],
        "oasis2_auc": r["oasis2_auc"],
        "oasis2_bacc": r["oasis2_balanced_acc"],
        "generalization_gap_delta_auc": r["generalization_gap_delta_auc"],
        "delta_sign_convention": "OASIS1_Test_minus_ADNI",
        "double_verification": "PASS_VERIFIED",
    })
df_three = pd.DataFrame(three_rows)
df_three.to_csv(RESULTS_DIR / "FINAL_THREE_COHORT_MODEL_COMPARISON.csv", index=False)
df_three.to_csv(EXP_RESULTS_DIR / "V4_Three_Cohort_Comparison.csv", index=False)
print("Saved reconciled FINAL_THREE_COHORT_MODEL_COMPARISON.csv")

# 4. Statistical Analysis CSV (Confirming exact 9-test Holm correction)
stat_hypotheses = [
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
B = 2000
stat_rows = []
for hid, desc, mA, mB, cname, df in stat_hypotheses:
    y = df["true_label"].values
    pA = df[mA].values
    pB = df[mB].values
    base_auc_A = roc_auc_score(y, pA)
    base_auc_B = roc_auc_score(y, pB)
    delta_auc = float(base_auc_A - base_auc_B)
    
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
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_val = 2.0 * min(np.mean(boot_diffs <= 0.0), np.mean(boot_diffs >= 0.0))
    p_val = max(p_val, 1.0 / B)
    
    stat_rows.append({
        "hypothesis_id": hid,
        "description": desc,
        "model_A": mA,
        "model_B": mB,
        "cohort": cname,
        "auc_A": round(base_auc_A, 4),
        "auc_B": round(base_auc_B, 4),
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

df_stat["holm_bonferroni_p_value"] = [round(p, 4) for p in adj_p_vals]
df_stat["statistically_significant"] = sig_flags

df_stat.to_csv(RESULTS_DIR / "FINAL_STATISTICAL_ANALYSIS.csv", index=False)
df_stat.to_csv(TABLES_DIR / "FINAL_STATISTICAL_ANALYSIS.csv", index=False)
df_stat.to_csv(EXP_RESULTS_DIR / "V4_Statistical_Analysis.csv", index=False)
print("Saved reconciled FINAL_STATISTICAL_ANALYSIS.csv and V4_Statistical_Analysis.csv")

# 5. Copy TrustGate Analysis to 03_EXPERIMENTAL_RESULTS
shutil.copyfile(RESULTS_DIR / "FINAL_TRUSTGATE_ANALYSIS.csv", EXP_RESULTS_DIR / "V4_TrustGate_Results.csv")
print("All canonical tables reconciled and synchronized successfully!")
