"""
Generates the comprehensive master model comparison table.
Integrates:
- Supervised 3D Baselines (ResNet-18, ResNet-34)
- 2D Transfer ResNet-18 (Slice-Aggregated)
- Classical Radiomics (Linear SVM)
- MONAI Swin UNETR SSL (Linear Probe)
- FOMO25 AMAES ResEnc-B (Linear Probe)
- Blocked candidate foundation models (BrainIAC, FOMO25 Winner)
Keeps Validation, OASIS Test, and ADNI External metrics clearly separated.
"""

import os
import json
import pandas as pd

OUTPUT_CSV = r"D:\ml project\results\report\master_model_comparison_table.csv"
RUNS_CSV = r"D:\brain_mri_data\benchmark_runs\tables\master_model_comparison_table.csv"

FOMO_RESULTS_JSON = r"D:\ml project\results\fomo_linear_probe\fomo25_linear_probe_results.json"
with open(FOMO_RESULTS_JSON, "r") as f:
    fomo_data = json.load(f)

f_val = fomo_data["metrics"]["validation"]
f_test = fomo_data["metrics"]["locked_test"]
f_adni = fomo_data["metrics"]["adni_external"]

rows = [
    {
        "Model": "3D ResNet-18 (Scratch Baseline)",
        "Paradigm": "Supervised Baseline",
        "Weights_Source": "Random Scratch (MedicalNet init)",
        "Total_Params": 33160513,
        "Trainable_Params": 33160513,
        "Status": "COMPLETED",
        "OASIS_Val_AUC": 0.5600,
        "OASIS_Val_BalAcc": 0.5000,
        "OASIS_Val_Brier": 0.2480,
        "OASIS_Test_AUC": 0.6178,
        "OASIS_Test_AUC_95CI": "[0.4133, 0.8133]",
        "OASIS_Test_BalAcc": 0.5333,
        "OASIS_Test_Sens": 1.0000,
        "OASIS_Test_Spec": 0.0667,
        "OASIS_Test_Brier": 0.2462,
        "ADNI_Ext_AUC": 0.5128,
        "ADNI_Ext_AUC_95CI": "[0.4612, 0.5644]",
        "ADNI_Ext_BalAcc": 0.4940,
        "ADNI_Ext_Sens": 0.9848,
        "ADNI_Ext_Spec": 0.0033,
        "ADNI_Ext_Brier": 0.2738
    },
    {
        "Model": "3D ResNet-34 (Scratch Baseline)",
        "Paradigm": "Supervised Baseline",
        "Weights_Source": "Random Scratch (MedicalNet init)",
        "Total_Params": 63489217,
        "Trainable_Params": 63489217,
        "Status": "COMPLETED",
        "OASIS_Val_AUC": 0.5822,
        "OASIS_Val_BalAcc": 0.5000,
        "OASIS_Val_Brier": 0.2490,
        "OASIS_Test_AUC": 0.5311,
        "OASIS_Test_AUC_95CI": "[0.3200, 0.7333]",
        "OASIS_Test_BalAcc": 0.5000,
        "OASIS_Test_Sens": 0.6000,
        "OASIS_Test_Spec": 0.4000,
        "OASIS_Test_Brier": 0.2482,
        "ADNI_Ext_AUC": 0.4777,
        "ADNI_Ext_AUC_95CI": "[0.4261, 0.5293]",
        "ADNI_Ext_BalAcc": 0.5133,
        "ADNI_Ext_Sens": 0.4061,
        "ADNI_Ext_Spec": 0.6205,
        "ADNI_Ext_Brier": 0.2534
    },
    {
        "Model": "2D Transfer ResNet-18 (Slice-Aggregated)",
        "Paradigm": "2D Pretrained Transfer",
        "Weights_Source": "Torchvision ImageNet-1K V1",
        "Total_Params": 11177025,
        "Trainable_Params": 8395521,
        "Status": "COMPLETED",
        "OASIS_Val_AUC": 0.7778,
        "OASIS_Val_BalAcc": 0.7333,
        "OASIS_Val_Brier": 0.1040,
        "OASIS_Test_AUC": 0.6222,
        "OASIS_Test_AUC_95CI": "[0.4133, 0.8178]",
        "OASIS_Test_BalAcc": 0.4333,
        "OASIS_Test_Sens": 0.2000,
        "OASIS_Test_Spec": 0.6667,
        "OASIS_Test_Brier": 0.2993,
        "ADNI_Ext_AUC": 0.5778,
        "ADNI_Ext_AUC_95CI": "[0.5280, 0.6276]",
        "ADNI_Ext_BalAcc": 0.5000,
        "ADNI_Ext_Sens": 1.0000,
        "ADNI_Ext_Spec": 0.0000,
        "ADNI_Ext_Brier": 0.5315
    },
    {
        "Model": "Classical Radiomics (Linear SVM)",
        "Paradigm": "Engineered Radiomics ML",
        "Weights_Source": "Engineered Volumetric/Texture Features",
        "Total_Params": 25,
        "Trainable_Params": 25,
        "Status": "COMPLETED",
        "OASIS_Val_AUC": 0.6978,
        "OASIS_Val_BalAcc": 0.6000,
        "OASIS_Val_Brier": 0.2218,
        "OASIS_Test_AUC": 0.5689,
        "OASIS_Test_AUC_95CI": "[0.3600, 0.7689]",
        "OASIS_Test_BalAcc": 0.5667,
        "OASIS_Test_Sens": 0.6000,
        "OASIS_Test_Spec": 0.5333,
        "OASIS_Test_Brier": 0.2470,
        "ADNI_Ext_AUC": 0.5149,
        "ADNI_Ext_AUC_95CI": "[0.4638, 0.5660]",
        "ADNI_Ext_BalAcc": 0.4957,
        "ADNI_Ext_Sens": 0.9848,
        "ADNI_Ext_Spec": 0.0066,
        "ADNI_Ext_Brier": 0.4491
    },
    {
        "Model": "MONAI Swin UNETR SSL (Linear Probe)",
        "Paradigm": "3D Foundation Model",
        "Weights_Source": "Project-MONAI SSL Pretrained (weights/model_swinvit.pt)",
        "Total_Params": 8062771,
        "Trainable_Params": 769,
        "Status": "COMPLETED",
        "OASIS_Val_AUC": 0.4467,
        "OASIS_Val_BalAcc": 0.5000,
        "OASIS_Val_Brier": 0.2495,
        "OASIS_Test_AUC": 0.6044,
        "OASIS_Test_AUC_95CI": "[0.3956, 0.8044]",
        "OASIS_Test_BalAcc": 0.5000,
        "OASIS_Test_Sens": 1.0000,
        "OASIS_Test_Spec": 0.0000,
        "OASIS_Test_Brier": 0.2495,
        "ADNI_Ext_AUC": 0.5834,
        "ADNI_Ext_AUC_95CI": "[0.5331, 0.6337]",
        "ADNI_Ext_BalAcc": 0.5000,
        "ADNI_Ext_Sens": 1.0000,
        "ADNI_Ext_Spec": 0.0000,
        "ADNI_Ext_Brier": 0.2569
    },
    {
        "Model": "FOMO25 AMAES ResEnc-B (Linear Probe)",
        "Paradigm": "3D Foundation Model",
        "Weights_Source": "FOMO-MRI/AMAES_resenc_b_fomo260k (resenc_unet_b.ckpt)",
        "Total_Params": 90296897,
        "Trainable_Params": 321,
        "Status": "COMPLETED",
        "OASIS_Val_AUC": f_val["roc_auc"],
        "OASIS_Val_BalAcc": f_val["balanced_accuracy"],
        "OASIS_Val_Brier": f_val["brier_score"],
        "OASIS_Test_AUC": f_test["roc_auc"],
        "OASIS_Test_AUC_95CI": f"[{f_test['roc_auc_ci_lower']:.4f}, {f_test['roc_auc_ci_upper']:.4f}]",
        "OASIS_Test_BalAcc": f_test["balanced_accuracy"],
        "OASIS_Test_Sens": f_test["sensitivity"],
        "OASIS_Test_Spec": f_test["specificity"],
        "OASIS_Test_Brier": f_test["brier_score"],
        "ADNI_Ext_AUC": f_adni["roc_auc"],
        "ADNI_Ext_AUC_95CI": f"[{f_adni['roc_auc_ci_lower']:.4f}, {f_adni['roc_auc_ci_upper']:.4f}]",
        "ADNI_Ext_BalAcc": f_adni["balanced_accuracy"],
        "ADNI_Ext_Sens": f_adni["sensitivity"],
        "ADNI_Ext_Spec": f_adni["specificity"],
        "ADNI_Ext_Brier": f_adni["brier_score"]
    },
    {
        "Model": "BrainIAC Foundation Model",
        "Paradigm": "3D Foundation Model",
        "Weights_Source": "AIM-Harvard/BrainIAC (gated, HTTP 401)",
        "Total_Params": "N/A",
        "Trainable_Params": "N/A",
        "Status": "BLOCKED",
        "OASIS_Val_AUC": "N/A",
        "OASIS_Val_BalAcc": "N/A",
        "OASIS_Val_Brier": "N/A",
        "OASIS_Test_AUC": "N/A",
        "OASIS_Test_AUC_95CI": "N/A",
        "OASIS_Test_BalAcc": "N/A",
        "OASIS_Test_Sens": "N/A",
        "OASIS_Test_Spec": "N/A",
        "OASIS_Test_Brier": "N/A",
        "ADNI_Ext_AUC": "N/A",
        "ADNI_Ext_AUC_95CI": "N/A",
        "ADNI_Ext_BalAcc": "N/A",
        "ADNI_Ext_Sens": "N/A",
        "ADNI_Ext_Spec": "N/A",
        "ADNI_Ext_Brier": "N/A"
    },
    {
        "Model": "FOMO25 Winner (FOMO2JOMO)",
        "Paradigm": "3D Foundation Model",
        "Weights_Source": "jbanusco/FOMO25-Winner (gated, Docker only)",
        "Total_Params": "N/A",
        "Trainable_Params": "N/A",
        "Status": "BLOCKED",
        "OASIS_Val_AUC": "N/A",
        "OASIS_Val_BalAcc": "N/A",
        "OASIS_Val_Brier": "N/A",
        "OASIS_Test_AUC": "N/A",
        "OASIS_Test_AUC_95CI": "N/A",
        "OASIS_Test_BalAcc": "N/A",
        "OASIS_Test_Sens": "N/A",
        "OASIS_Test_Spec": "N/A",
        "OASIS_Test_Brier": "N/A",
        "ADNI_Ext_AUC": "N/A",
        "ADNI_Ext_AUC_95CI": "N/A",
        "ADNI_Ext_BalAcc": "N/A",
        "ADNI_Ext_Sens": "N/A",
        "ADNI_Ext_Spec": "N/A",
        "ADNI_Ext_Brier": "N/A"
    }
]

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
df.to_csv(RUNS_CSV, index=False)
print("Saved master comparison table to:", OUTPUT_CSV)
print("Saved master comparison table to:", RUNS_CSV)
print(df[["Model", "Status", "OASIS_Val_AUC", "OASIS_Test_AUC", "ADNI_Ext_AUC"]].to_string())
