# SHIFT-MRI V4: Initial Scientific and Software Audit

**Date:** 2026-10-22 13:51:07  
**Campaign Branch:** `research/v4-authentic-foundation-retraining`  
**Investigator:** Bhavy Garg, Vellore Institute of Technology, Chennai  

---

## 1. Executive Summary & Hardware Verification

The SHIFT-MRI V4 experimental campaign initiates a fresh, scientifically rigorous, and fully reproducible retraining and re-evaluation cycle.

### Computational Environment
- **Operating System:** Windows-11-10.0.26300-SP0
- **Python Version:** 3.12.10
- **PyTorch Version:** 2.11.0+cu128 (CUDA: 12.8)
- **GPU Accelerator:** NVIDIA RTX PRO 2000 Blackwell (15.93 GB VRAM)
- **Host RAM:** 50.4 GB Available / 63.34 GB Total
- **Storage Availability:**
  - System Drive (C:): 409.29 GB Free / 610.21 GB Total
  - Project Drive (D:): 310.12 GB Free / 341.8 GB Total
- **Core Scientific Libraries:** MONAI `1.6.1`, TorchVision `0.26.0+cu128`, Timm `1.0.30`, NiBabel `5.4.2`

---

## 2. Dataset Governance & Clinical Label Audits

### 2.1 OASIS-1 In-Domain Cohort
- **Manifest:** `D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv`
- **Total Registered Scans:** 198 ($1.5$T Siemens Vision)
- **Clinical Stratification:** 98 Cognitively Normal (CDR 0) / 100 Cognitively Impaired (CDR $\ge 0.5$)
- **Source-Locked Partitioning:**
  - Training Partition: 138 subjects
  - Validation Partition: 30 subjects (used strictly for convergence monitoring, early stopping, and threshold calibration)
  - Locked Holdout Test: 30 subjects (strictly isolated from model training)
- **Participant Isolation Check:** `PASS: ZERO PATIENT OVERLAP (138 train / 30 val / 30 test)`

### 2.2 ADNI Zero-Shot External Evaluation Cohort
- **Manifest:** `D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv`
- **Total Registered Scans:** 500 multi-site scans
- **Diagnostic Stratification:** 303 Cognitively Normal (CN) / 197 Alzheimer's Disease (AD)
- **Scanner Provenance:** Siemens ($N=370, 74\%$) and Philips ($N=130, 26\%$). Zero ($0$) General Electric systems in this preprocessed cohort.
- **Evaluation Status:** External evaluation benchmark; previously audited in V3, disclosed as an established external transfer benchmark.

### 2.3 OASIS-2 Longitudinal Cohort
- **Manifest:** `D:\brain_mri_data\OASIS2\manifests\oasis2_baseline_manifest.csv`
- **Total Scans:** 150 scans (85 CN, 65 Impaired)
- **Integrity Status:** UNRESOLVED / EXPLORATORY ONLY

---

## 3. Foundation Model Authenticity & Provenance Audit

### 3.1 BrainIAC
- **Historical Defect:** Previous pilot utilized a proxy adapter with convolutional blocks.
- **V4 Action Gate:** Reject historical proxy. Verify official BrainIAC repository (`https://github.com/AIM-KannLab/BrainIAC`), load authentic Vision Transformer weights, confirm parameter count and weight hash, and prevent random initialization fallback.

### 3.2 MONAI Swin UNETR SSL
- **Architecture:** 3D Swin Vision Transformer with shifted windows.
- **Pretraining Modality Disclosure:** Trained on CVPR 2022 self-supervised cohort of $5,050$ CT volumes (not brain MRI). Must be explicitly described as cross-modality medical volumetric pretraining.
- **Weights Integrity:** Verify official MONAI self-supervised checkpoint, verify state dictionary keys, ensure exact layer alignment.

### 3.3 FOMO25 AMAES ResEnc-B
- **Architecture:** 3D Residual Encoder with Masked Autoencoding (`AMAESResEncB`).
- **Pretraining Dataset:** FOMO-60K / multi-organ collection.
- **Integrity Audit:** Audit potential participant overlap with OASIS-1; distinguish self-supervised pretraining representation from downstream supervised leakage.

---

## 4. Retraining Campaign Execution Plan (Stages D–Q)

1. **Pipeline Engineering:** Config-driven PyTorch + MONAI training system with AMP, gradient accumulation, and deterministic seed control.
2. **Supervised Baseline Retraining:** Retrain 3D ResNet-10, 18, 34, DenseNet-121, 2D Transfer, 2.5D, and Hybrid architectures on $N=138$ OASIS-1.
3. **Foundation Model Adaptation:** Linear probing and validation-driven fine-tuning of BrainIAC, Swin UNETR, and FOMO25.
4. **Controlled Evaluation:** Locked evaluation on $N=30$ OASIS-1 and $N=500$ ADNI, computing ROC-AUC, 95% bootstrap CIs, calibration (Brier score, ECE), and TrustGate risk scores.
5. **Final Scientific Deliverables:** Produce IEEE Access research manuscript, reproducible figures, tables, and test reports.
