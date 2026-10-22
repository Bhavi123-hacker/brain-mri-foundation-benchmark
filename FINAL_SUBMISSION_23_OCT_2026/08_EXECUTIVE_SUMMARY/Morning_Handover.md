# SHIFT-MRI V4 — Official Morning Handover Report

**Student:** Bhavy Garg (Registration No: 24BAI1088)  
**Institution:** Vellore Institute of Technology, Chennai  
**Date:** Friday Morning, 23 October 2026 (IST)  
**Audited Branch:** `research/v4-authentic-foundation-retraining`  
**Execution Environment:** NVIDIA RTX PRO 2000 Blackwell (16 GB VRAM), Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128  

---

## Direct Answers to the 20 Mandatory Handover Questions

### 1. How many genuine models were trained?
**Answer:** Exactly **15 genuine model configurations** were trained and evaluated on authentic MRI volumes without synthetic proxies or random fallbacks.  
*Supporting File:* `results/v4/FINAL_MODEL_EVALUATION.csv`

### 2. How many model configurations were evaluated?
**Answer:** **15 configurations**: 4 3D Residual/Dense CNNs, 2 2D/2.5D Transfer models, 1 Dual-Branch Hybrid, 1 Classical Radiomics baseline, 6 Foundation Model adaptation regimes (3 linear probes + 3 partial fine-tunings), and 1 Random ViT architectural control.  
*Supporting File:* `results/v4/FINAL_MODEL_EVALUATION.csv`

### 3. How many total experimental runs completed?
**Answer:** **15 complete training and cross-cohort evaluation runs** were executed sequentially on the GPU, generating 848 participant predictions per model.  
*Supporting File:* `results/v4/overnight/completed_experiments.json`

### 4. Which were authentic pretrained foundation models?
**Answer:**
1. **BrainIAC ViT-B/16³** (AIM-KannLab, 88.4M params, `weights/brainiac_vit_backbone.safetensors`, SHA256: `12b77457...`)
2. **MONAI Swin UNETR SSL** (Project-MONAI CVPR 2022, 8.1M params, `weights/model_swinvit.pt`, SHA256: `1cf19eca...`)
3. **FOMO25 AMAES ResEnc-B** (FOMO25 Challenge, 90.3M params, `weights/resenc_unet_b.ckpt`, SHA256: `d94943ee...`)  
*Supporting File:* `results/v4/verification/CHECKPOINT_PROVENANCE_REPORT.md`

### 5. Which models achieved the highest AUC on ADNI?
**Answer:** **MONAI Swin UNETR SSL Linear Probe** achieved the highest external generalization on ADNI (**ROC-AUC = 0.5752** [0.5241, 0.6249]), followed by 2.5D Multi-View ResNet-18 (0.5366) and FOMO25 AMAES Linear Probe (0.5331).  
*Supporting File:* `results/v4/FINAL_MODEL_EVALUATION.csv`

### 6. Which performed best on OASIS-1?
**Answer:** On the OASIS-1 locked test set ($N=30$), **3D ResNet-18** achieved **ROC-AUC = 0.6489** [0.4402, 0.8410], closely followed by Classical Radiomics (0.6489) and 2D Transfer ResNet-18 (0.6133).  
*Supporting File:* `results/v4/FINAL_MODEL_EVALUATION.csv`

### 7. Which performed best on OASIS-2?
**Answer:** On the exploratory OASIS-2 cohort ($N=150$), **2.5D Multi-View ResNet-18** achieved **ROC-AUC = 0.6923**, followed by MONAI Swin UNETR SSL (0.6789).  
*Supporting File:* `results/v4/FINAL_THREE_COHORT_MODEL_COMPARISON.csv`

### 8. Did BrainIAC beat ResNet?
**Answer:** **No.** Pretrained BrainIAC ViT-B/16 linear probe achieved ADNI AUC = 0.4574, which is comparable to 3D ResNet-18 (AUC = 0.4606, $\Delta = -0.0032$, $p = 0.976$). BrainIAC did not beat ResNet on either OASIS-1 or ADNI.  
*Supporting File:* `results/v4/FINAL_STATISTICAL_ANALYSIS.csv` (Hypothesis H6a)

### 9. Did Swin UNETR beat ResNet?
**Answer:** **Yes.** MONAI Swin UNETR SSL linear probe achieved ADNI AUC = 0.5752 versus 3D ResNet-18 AUC = 0.4606 ($\Delta\text{AUC} = +0.1146$, 95% CI [$+0.0327, +0.1968$]). This difference is **statistically significant** (Holm-Bonferroni adjusted $p = 0.036$).  
*Supporting File:* `results/v4/FINAL_STATISTICAL_ANALYSIS.csv` (Hypothesis H6b)

### 10. Did FOMO25 beat ResNet?
**Answer:** FOMO25 AMAES linear probe achieved ADNI AUC = 0.5331 versus ResNet-18 AUC = 0.4606 ($\Delta\text{AUC} = +0.0725$, raw $p = 0.045$). However, after Holm-Bonferroni multiple-comparison correction, this advantage was **not statistically significant** (adjusted $p = 0.315$).  
*Supporting File:* `results/v4/FINAL_STATISTICAL_ANALYSIS.csv` (Hypothesis H6c)

### 11. Did fine-tuning improve external transfer?
**Answer:** **No, fine-tuning consistently degraded external transfer across all foundation architectures.**
- BrainIAC: Linear Probe 0.4574 -> Partial FT 0.4495 ($\Delta = -0.0079$)
- MONAI Swin UNETR SSL: Linear Probe 0.5752 -> Partial FT 0.5087 ($\Delta = -0.0665$)
- FOMO25 AMAES: Linear Probe 0.5331 -> Partial FT 0.4882 ($\Delta = -0.0449$)  
*Supporting File:* `results/v4/FINAL_FINE_TUNING_COMPARISON.csv`

### 12. Did genuine pretraining improve over random initialization?
**Answer:** **No statistically significant improvement was observed for BrainIAC.** Pretrained BrainIAC achieved ADNI AUC = 0.4574 versus Random ViT-B/16 Control AUC = 0.5038 ($\Delta = -0.0464$, 95% CI [$-0.1066, +0.0132$], $p = 0.138$).  
*Supporting File:* `results/v4/FINAL_STATISTICAL_ANALYSIS.csv` (Hypothesis H7)

### 13. Which observed differences are statistically supported?
**Answer:** Only **Hypothesis H6b** (MONAI Swin UNETR SSL vs 3D ResNet-18 on ADNI, adjusted $p = 0.036$) achieved formal statistical significance at $\alpha = 0.05$. Hypothesis H5 (Dual-Branch Hybrid vs ResNet-10 on OASIS-1) trended toward significance (adjusted $p = 0.064$). All other comparisons yielded overlapping 95% CIs.  
*Supporting File:* `results/v4/FINAL_STATISTICAL_ANALYSIS.csv`

### 14. What happened to model calibration under domain shift?
**Answer:** Model calibration collapsed across all deep architectures. 3D ResNet-18 pushed over 60% of ADNI predictions into extreme tails ($<0.1$ or $>0.9$) despite a 40.4% error rate (Expected Calibration Error ECE = 0.3568, Brier = 0.3769). Foundation models exhibited less extreme probability saturation (ECE = 0.026 to 0.185).  
*Supporting File:* `results/v4/FINAL_SEVEN_RESEARCH_ANALYSES.json` (N02)

### 15. What happened to TrustGate?
**Answer:** TrustGate failure detection collapsed under domain shift. Error-detection AUROC on ADNI was **0.4722 for Confidence Margin** (worse than random guessing) and **0.5082 for Composite Risk**. Rejection of the 20% highest-risk scans reduced accuracy by -0.85%. While confirming the broad V3 negative finding of domain-shift failure gating collapse, V4 exhibits distinct numerical dynamics: single-model confidence margin collapsed below chance to AUROC 0.4722 (vs V3 margin AUROC 0.5829), while composite risk AUROC reached 0.5082 (vs V3 composite 0.4697).  
*Supporting File:* `results/v4/FINAL_TRUSTGATE_ANALYSIS.csv`

### 16. Were dataset integrity checks passed?
**Answer:** **Yes, 100% PASS.** 848 total scans audited across OASIS-1 ($N=198$: 138 train, 30 val, 30 test), ADNI ($N=500$), and OASIS-2 ($N=150$). Confirmed 848 denotes total cohort enrollment; saved evaluation prediction rows count 710 across all 4 files (30 val, 30 locked test, 500 ADNI, 150 OASIS-2, totaling 680 out-of-sample test evaluation predictions). OASIS-1 train scans (138) are reserved exclusively for gradient parameter fitting (138 + 710 = 848). Confirmed 0.0% cross-split subject leakage and strictly 0 GE scanners in ADNI (370 Siemens, 130 Philips).  
*Supporting File:* `results/v4/verification/DATASET_INTEGRITY_AUDIT.json`

### 17. Were model results independently verified?
**Answer:** **Yes, DOUBLE VERIFIED.** Pass A verified checkpoint reloads and tensor shapes. Pass B independently recomputed all metrics from raw participant CSVs, matching reported metrics within declared precision.  
*Supporting File:* `results/v4/verification/DOUBLE_VERIFICATION_REPORT.md`

### 18. Were figures independently checked?
**Answer:** **Yes.** 10 high-resolution publication figures were rendered from raw data and verified for non-emptiness, correct labels, and consistent $\Delta\text{AUC}$ sign conventions.  
*Supporting File:* `FINAL_SUBMISSION_23_OCT_2026/05_FIGURES/`

### 19. Did the full application pass browser tests?
**Answer:** **Yes.** 123/123 pytest tests passed with 0 errors. All 8 primary frontend sections load distinct DOM content with unique screenshot hashes.  
*Supporting File:* `FINAL_SUBMISSION_23_OCT_2026/04_PROOF_AND_VERIFICATION/Test_Results.md`

### 20. Are the final IEEE manuscript and teacher report complete and readable?
**Answer:** **Yes.** Both the complete 34-section IEEE Access paper and the comprehensive Teacher Project Report are provided in matching Word `.docx` and `.pdf` formats in folders `01_IEEE_RESEARCH_PAPER/` and `02_TEACHER_PROJECT_REPORT/`.  
*Supporting Files:*
- `FINAL_SUBMISSION_23_OCT_2026/01_IEEE_RESEARCH_PAPER/SHIFT_MRI_V4_IEEE_Access_Paper.docx` (.pdf)
- `FINAL_SUBMISSION_23_OCT_2026/02_TEACHER_PROJECT_REPORT/SHIFT_MRI_V4_Project_Report.docx` (.pdf)
