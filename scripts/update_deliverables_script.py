import re
from pathlib import Path

path = Path(r"D:\ml project\scripts\build_all_final_submission_deliverables.py")
content = path.read_text(encoding="utf-8")

# 1. Update AUC numbers in text
replacements = [
    # Swin UNETR & ResNet-18
    ("ROC-AUC = 0.5756 [0.5245, 0.6253]), significantly outperforming standard 3D ResNet-18 (ROC-AUC = 0.4601 [0.4095, 0.5121], Holm-Bonferroni adjusted p = 0.036)",
     "ROC-AUC = 0.5752 [0.5241, 0.6249]), significantly outperforming standard 3D ResNet-18 (ROC-AUC = 0.4606 [0.4101, 0.5125], Holm-Bonferroni adjusted p = 0.036)"),
    ("top ADNI ROC-AUC of 0.5756 [0.5245, 0.6253] (ΔAUC = +0.0255). Paired non-parametric bootstrap testing confirmed this advantage was statistically significant (ΔAUC = +0.1146, Holm-Bonferroni adjusted p = 0.036)",
     "top ADNI ROC-AUC of 0.5752 [0.5241, 0.6249] (ΔAUC = +0.0293). Paired non-parametric bootstrap testing confirmed this advantage was statistically significant (ΔAUC = +0.1146, Holm-Bonferroni adjusted p = 0.036)"),
    ("ADNI (ROC-AUC = 0.5756),", "ADNI (ROC-AUC = 0.5752),"),
    ("ADNI AUC (0.5756), significantly outperforming 3D ResNet-18 (0.4601, p=0.036)",
     "ADNI AUC (0.5752), significantly outperforming 3D ResNet-18 (0.4606, p=0.036)"),
    ("ADNI AUC = 0.5756), \n\"        \"statistically outperforming 3D ResNet-18 (AUC = 0.4601, p = 0.036)",
     "ADNI AUC = 0.5752), \n\"        \"statistically outperforming 3D ResNet-18 (AUC = 0.4606, p = 0.036)"),
    ("ADNI (**ROC-AUC = 0.5756** [0.5245, 0.6253]), followed by 2.5D Multi-View ResNet-18 (0.5373) and FOMO25 AMAES Linear Probe (0.5324)",
     "ADNI (**ROC-AUC = 0.5752** [0.5241, 0.6249]), followed by 2.5D Multi-View ResNet-18 (0.5366) and FOMO25 AMAES Linear Probe (0.5331)"),
    ("3D ResNet-18 (AUC = 0.4601, $\\Delta = -0.0032$, $p = 0.976$)",
     "3D ResNet-18 (AUC = 0.4606, $\\Delta = -0.0032$, $p = 0.976$)"),
    ("Swin UNETR SSL linear probe achieved ADNI AUC = 0.5756 versus 3D ResNet-18 AUC = 0.4601 ($\Delta\\text{AUC} = +0.1146$, 95% CI [$+0.0327, +0.1968$])",
     "Swin UNETR SSL linear probe achieved ADNI AUC = 0.5752 versus 3D ResNet-18 AUC = 0.4606 ($\Delta\\text{AUC} = +0.1146$, 95% CI [$+0.0327, +0.1968$])"),
    ("FOMO25 AMAES linear probe achieved ADNI AUC = 0.5324 versus ResNet-18 AUC = 0.4601 ($\Delta\\text{AUC} = +0.0725$, raw $p = 0.045$)",
     "FOMO25 AMAES linear probe achieved ADNI AUC = 0.5331 versus ResNet-18 AUC = 0.4606 ($\Delta\\text{AUC} = +0.0725$, raw $p = 0.045$)"),
    ("BrainIAC ViT-B/16 architectural control (ROC-AUC = 0.4574 vs 0.5044, p = 0.138)",
     "BrainIAC ViT-B/16 architectural control (ROC-AUC = 0.4574 vs 0.5038, p = 0.138)"),
    ("randomly initialized ViT-B/16 control achieved AUC = 0.5044 (Δ = -0.0464, 95% CI [-0.1066, 0.0132], p = 0.138)",
     "randomly initialized ViT-B/16 control achieved AUC = 0.5038 (Δ = -0.0464, 95% CI [-0.1066, 0.0132], p = 0.138)"),
    ("Random ViT-B/16 Control AUC = 0.5044 ($\Delta = -0.0464$, 95% CI [$-0.1066, +0.0132$], $p = 0.138$)",
     "Random ViT-B/16 Control AUC = 0.5038 ($\Delta = -0.0464$, 95% CI [$-0.1066, +0.0132$], $p = 0.138$)"),
    ("control \n(AUC = 0.5044, p = 0.138)", "control \n(AUC = 0.5038, p = 0.138)"),
    # Fine-Tuning
    ("Swin UNETR's ADNI performance dropped from 0.5756 to 0.5084 (Δ = -0.0672). Similarly, FOMO25 dropped from 0.5324 to 0.4878 (Δ = -0.0446), and BrainIAC dropped from 0.4574 to 0.4492 (Δ = -0.0082)",
     "Swin UNETR's ADNI performance dropped from 0.5752 to 0.5087 (Δ = -0.0665). Similarly, FOMO25 dropped from 0.5331 to 0.4882 (Δ = -0.0449), and BrainIAC dropped from 0.4574 to 0.4495 (Δ = -0.0079)"),
    ("- BrainIAC: Linear Probe 0.4574 -> Partial FT 0.4492 ($\Delta = -0.0082)\n- MONAI Swin UNETR SSL: Linear Probe 0.5756 -> Partial FT 0.5084 ($\Delta = -0.0672)\n- FOMO25 AMAES: Linear Probe 0.5324 -> Partial FT 0.4878 ($\Delta = -0.0446)",
     "- BrainIAC: Linear Probe 0.4574 -> Partial FT 0.4495 ($\Delta = -0.0079)\n- MONAI Swin UNETR SSL: Linear Probe 0.5752 -> Partial FT 0.5087 ($\Delta = -0.0665)\n- FOMO25 AMAES: Linear Probe 0.5331 -> Partial FT 0.4882 ($\Delta = -0.0449)"),
    ("OASIS-1 locked test set ($N=30$), **3D ResNet-18** achieved **ROC-AUC = 0.6477** [0.4399, 0.8400], closely followed by Classical Radiomics (0.6467) and 2D Transfer ResNet-18 (0.6104)",
     "OASIS-1 locked test set ($N=30$), **3D ResNet-18** achieved **ROC-AUC = 0.6489** [0.4402, 0.8410], closely followed by Classical Radiomics (0.6489) and 2D Transfer ResNet-18 (0.6133)"),
    ("OASIS-2 cohort ($N=150$), **2.5D Multi-View ResNet-18** achieved **ROC-AUC = 0.6939**, followed by MONAI Swin UNETR SSL (0.6798)",
     "OASIS-2 cohort ($N=150$), **2.5D Multi-View ResNet-18** achieved **ROC-AUC = 0.6923**, followed by MONAI Swin UNETR SSL (0.6789)"),
    ("3D ResNet-18 (AUC = 0.6477) and Classical Radiomics (AUC = 0.6467)",
     "3D ResNet-18 (AUC = 0.6489) and Classical Radiomics (AUC = 0.6489)"),
    ("3D ResNet-18 collapsed to AUC = 0.4601 (Generalization Gap ΔAUC = +0.1876)",
     "3D ResNet-18 collapsed to AUC = 0.4606 (Generalization Gap ΔAUC = +0.1883)"),
    # Summary README text
    ("AUC = 0.5756", "AUC = 0.5752"),
    ("AUC = 0.4601", "AUC = 0.4606"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"Replaced: {old[:40]}... -> {new[:40]}...")
    else:
        print(f"NOT FOUND: {old[:50]}...")

# 2. Exclude Participant_Split_Audit.csv from zip creation to adhere to DUA
zip_exclusion_old = 'if f_path == zip_path or f.endswith((".pt", ".ckpt", ".safetensors", ".npy", ".nii", ".nii.gz")):'
zip_exclusion_new = 'if f_path == zip_path or f.endswith((".pt", ".ckpt", ".safetensors", ".npy", ".nii", ".nii.gz")) or "Participant_Split_Audit.csv" in f or "predictions" in f:'
if zip_exclusion_old in content:
    content = content.replace(zip_exclusion_old, zip_exclusion_new)
    print("Updated ZIP exclusion to strictly filter restricted participant data.")

# 3. Add explanation of 848 cohort enrollment in Question 16
q16_old = "848 scans audited across OASIS-1 ($N=198$), ADNI ($N=500$), and OASIS-2 ($N=150$). Confirmed 0.0% cross-split subject leakage, 0 non-finite values, and strictly 0 GE scanners in ADNI (370 Siemens, 130 Philips)."
q16_new = "848 total scans audited across OASIS-1 ($N=198$: 138 train, 30 val, 30 test), ADNI ($N=500$), and OASIS-2 ($N=150$). Confirmed 848 denotes total cohort enrollment; saved evaluation prediction rows count 710 across all 4 files (30 val, 30 locked test, 500 ADNI, 150 OASIS-2, totaling 680 out-of-sample test evaluation predictions). OASIS-1 train scans (138) are reserved exclusively for gradient parameter fitting (138 + 710 = 848). Confirmed 0.0% cross-split subject leakage and strictly 0 GE scanners in ADNI (370 Siemens, 130 Philips)."
if q16_old in content:
    content = content.replace(q16_old, q16_new)
    print("Updated Question 16 with exact 848 enrollment vs 710 prediction rows distinction.")

# 4. Clarify TrustGate V4 vs V3 distinction in Question 15
q15_old = "Rejection of the 20% highest-risk scans reduced accuracy by -0.85%, confirming the V3 negative finding."
q15_new = "Rejection of the 20% highest-risk scans reduced accuracy by -0.85%. While confirming the broad V3 negative finding of domain-shift failure gating collapse, V4 exhibits distinct numerical dynamics: single-model confidence margin collapsed below chance to AUROC 0.4722 (vs V3 margin AUROC 0.5829), while composite risk AUROC reached 0.5082 (vs V3 composite 0.4697)."
if q15_old in content:
    content = content.replace(q15_old, q15_new)
    print("Updated Question 15 with explicit TrustGate V4 vs V3 comparison.")

path.write_text(content, encoding="utf-8")
print("Saved updated build_all_final_submission_deliverables.py")
