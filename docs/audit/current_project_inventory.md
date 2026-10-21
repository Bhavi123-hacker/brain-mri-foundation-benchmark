# Current Project Inventory & Scientific Reconciliation Ledger
**Date**: October 21, 2026  
**Auditors**: Principal Investigator, Senior Deep Learning Lead & Statistical Methodologist  
**Scope**: End-to-end inventory of code modules, datasets, manifests, checkpoints, prediction files, metrics, and reconciliation of historical discrepancies.

---

## 1. Codebase Architecture & Modular Inventory

```text
D:\ml project/
├── configs/                  # Experiment configuration specifications (benchmark.yaml, models.yaml, datasets.yaml)
├── models/                   # Neural network implementations and foundation adapters
│   ├── baseline/             # 3D ResNet-18/34 implementations
│   ├── foundation_fomo25/    # Official FOMO25 AMAES ResEnc-B adapter (90.3M params)
│   └── foundation_swin/      # MONAI Swin UNETR SSL transformer adapter (8.1M params)
├── preprocessing/            # Medical image transformation, Z-score normalization, caching
├── training/                 # PyTorch training loop, budget sampling (10-100%), optimizer/scheduler
├── evaluation/               # Metrics calculation (ROC-AUC, BalAcc, Brier, ECE, CIs)
├── scripts/                  # Standalone execution pipelines (preprocessing, training, audits)
├── tests/                    # Pytest validation suites (test_master_benchmark_integrity.py, test_statistical_audit.py)
├── visualization/            # Plotting routines (ROC curves, calibration diagrams, label efficiency)
├── weights/                  # Heavy foundation model weights (resenc_unet_b.ckpt, model_swinvit.pt)
├── results/
│   ├── report/               # Authoritative V1 benchmark reports, master tables, and figures
│   ├── statistical_audit/    # Paired bootstrap tables, Holm corrections, calibration profiles
│   ├── fomo_linear_probe/    # FOMO25 feature manifests, linear probe model, and predictions
│   └── v2/                   # Reserved workspace for Version 2 experimental campaign
└── docs/                     # Comprehensive audit, research, and data documentation
```

---

## 2. Dataset & Manifest Inventory

| Dataset Identifier | Physical Disk Path | Primary Manifest | Unique Subjects | Classes / Ground Truth | Verification Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **OASIS-1 Primary** | `D:\brain_mri_data\OASIS1` | `manifests/oasis1_primary_cohort_splits.csv` | 198 | CDR (98 CN, 100 AD/Impaired) | **VERIFIED** (0.0% patient leakage) |
| - *Train Split* | `D:\brain_mri_data\OASIS1\processed\tensors` | Splits CSV (`split == "train"`) | 138 | 69 CN, 69 Impaired/AD | **VERIFIED** |
| - *Val Split* | `D:\brain_mri_data\OASIS1\processed\tensors` | Splits CSV (`split == "val"`) | 30 | 15 CN, 15 Impaired/AD | **VERIFIED** |
| - *Locked Test* | `D:\brain_mri_data\OASIS1\processed\tensors` | Splits CSV (`split == "test"`) | 30 | 15 CN, 15 Impaired/AD | **VERIFIED** |
| **ADNI External** | `D:\brain_mri_data\ADNI` | `manifests/adni_full_preprocessed_manifest.csv` | 500 | Clinical AD (197) vs CN (303) | **VERIFIED** (0.0% overlap with OASIS) |
| - *Siemens Subgroup*| `D:\brain_mri_data\ADNI\processed\tensors` | Manifest (`manufacturer == "SIEMENS"`) | 370 | 222 CN, 148 AD | **VERIFIED** |
| - *Philips Subgroup*| `D:\brain_mri_data\ADNI\processed\tensors` | Manifest (`manufacturer == "Philips"`) | 130 | 81 CN, 49 AD | **VERIFIED** |
| - *1.5T Subgroup* | `D:\brain_mri_data\ADNI\processed\tensors` | Manifest (`field_strength < 2.0`) | 165 | 94 CN, 71 AD | **VERIFIED** |
| - *3.0T Subgroup* | `D:\brain_mri_data\ADNI\processed\tensors` | Manifest (`field_strength >= 2.0`)| 335 | 209 CN, 126 AD | **VERIFIED** |
| **MIRIAD Template** | `D:\brain_mri_data\MIRIAD` | `results/miriad_external_validation/miriad_manifest_template.csv` | 69 | Clinical AD (46) vs CN (23) | **ACCESS_GATED** (UCL DUA required) |

---

## 3. Verified Checkpoints and Weights Ledger

| Checkpoint Name | File Location | Size | SHA-256 Checksum | Parameters | Provenance & Legitimacy |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **FOMO25 AMAES ResEnc-B** | `weights/resenc_unet_b.ckpt` | 1,167,183,049 B | `115d509241ad3a421a851dda3a1e5d98ae8377973ea9b29bbb2c2a181c277749` | 90,296,897 | **VERIFIED**: Authentic official baseline from FOMO-MRI (arXiv:2408.00640). |
| **MONAI Swin UNETR SSL** | `weights/model_swinvit.pt` | 411,162,269 B | `1cf19eca3ed0b4278c7c98692c8e3ca91438902506b3a3c915f7b4ee7be88f4e` | 8,062,771 | **VERIFIED**: Authentic Project-MONAI 3D SwinViT SSL weights (CVPR 2022). |
| **3D ResNet-18 (Seed 42)**| `D:\brain_mri_data\benchmark_runs\checkpoints\resnet18_seed42_best.pt` | 132,723,083 B | Verified | 33,160,513 | **VERIFIED**: Supervised baseline checkpoint. |
| **3D ResNet-34 (Seed 42)**| `D:\brain_mri_data\benchmark_runs\checkpoints\resnet34_seed42_best.pt` | 254,024,171 B | Verified | 63,489,217 | **VERIFIED**: Supervised baseline checkpoint. |
| **2D Transfer ResNet-18**| `D:\brain_mri_data\benchmark_runs\checkpoints\resnet2d_partial_finetune_best.pt` | 44,787,467 B | Verified | 11,177,025 | **VERIFIED**: Torchvision ImageNet-1K pretrained transfer checkpoint. |
| **Classical Radiomics** | `D:\brain_mri_data\benchmark_runs\checkpoints\classical_radiomics_best.pkl` | 52,154 B | Verified | 25 | **VERIFIED**: Scikit-learn Linear SVM fitted on 24 morphometric features. |
| **BrainIAC** | N/A (HuggingFace Hub) | N/A | Gated (`HTTP 401`) | ~65M | **BLOCKED**: Requires private organizational token; zero mock weights used. |
| **FOMO25 Winner** | N/A (Docker container) | N/A | Gated (`HTTP 401`) | ~86M | **BLOCKED**: Requires private token; monolithic Docker distribution. |

---

## 4. Reconciled Baseline Performance Ledger (Version 1)

Every metric reproduced directly from `oasis_locked_test_predictions.csv` and `adni_external_predictions.csv`:

| Model Architecture | OASIS Val AUC | OASIS Test ROC-AUC | OASIS Test 95% CI | OASIS Test BalAcc | ADNI Ext ROC-AUC | ADNI Ext 95% CI | ADNI Ext BalAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D ResNet-18 (Scratch)** | 0.5600 | **0.6178** | [0.4133, 0.8133] | 0.5333 | **0.5128** | [0.4612, 0.5644] | 0.4940 |
| **3D ResNet-34 (Scratch)** | 0.5822 | **0.5311** | [0.3200, 0.7333] | 0.5000 | **0.4777** | [0.4261, 0.5293] | 0.5133 |
| **2D Transfer ResNet-18** | **0.7778** | **0.6222** | [0.4133, 0.8178] | 0.4333 | **0.5778** | [0.5280, 0.6276] | 0.5000 |
| **Classical Radiomics (SVM)** | **0.6978** | **0.5689** | [0.3600, 0.7689] | 0.5667 | **0.5149** | [0.4638, 0.5660] | 0.4957 |
| **MONAI Swin UNETR SSL** | 0.4467 | **0.6044** | [0.3956, 0.8044] | 0.5000 | **0.5834** | [0.5331, 0.6337] | 0.5000 |
| **FOMO25 AMAES ResEnc-B** | **0.6667** | **0.5733** | [0.3664, 0.7989] | 0.5333 | **0.5054** | [0.4516, 0.5552] | 0.5000 |

---

## 5. Audit of Key Discrepancies & Resolutions

1. **Discrepancy A — Foundation Architecture Identity**:
   - *Investigation*: Early repository prototypes (October 9) used a 3D CNN surrogate class named `SwinUNETRAdapter`. On October 12, the authentic `monai.networks.nets.SwinUNETR` ViT backbone was downloaded, verified, and loaded with zero missing keys from `weights/model_swinvit.pt`. Today's FOMO25 AMAES adapter (`models/foundation_fomo25/amaes_adapter.py`) loads authentic PyTorch Lightning ResEnc-B weights (90.3M parameters).
   - *Resolution*: Historical surrogate results are preserved and labeled as `HISTORICAL_PROTOTYPE`; all authoritative benchmark tables reflect authentic transformer and residual encoder weights.
2. **Discrepancy B — Historical Holdout Exposure**:
   - *Investigation*: The OASIS test set ($N=30$) was evaluated on October 9 (ResNet-18 baseline AUC 0.6044) and early October 12 (AUC 0.6178).
   - *Resolution*: It is inaccurate to describe the test set as "untouched". We explicitly document that baseline test results were previously observed. Sprint models (2D Transfer, Radiomics, Swin UNETR, FOMO AMAES) were selected strictly using validation loss/AUC prior to test evaluation.
3. **Discrepancy C — Label Budget Multi-Seed Inconsistencies**:
   - *Investigation*: Early exploratory files recorded single-seed validation runs (e.g. ResNet-18 Val AUC 0.5422 at 10%), whereas later multi-paradigm summaries averaged over seeds.
   - *Resolution*: We maintain separate single-seed deterministic records and multi-seed aggregated metrics with explicit standard deviations.
4. **Discrepancy D — Public Data Governance**:
   - *Audit*: Verified that `.gitignore` prevents committing raw DICOM/NIfTI scans, cached `.npy` volumes, or private PHI. Participant IDs in manifests are anonymous research identifiers (`OAS1_XXXX`, `PTID_XXXX`) compliant with OASIS and ADNI open data guidelines.

---
*Inventory verified and sealed on October 21, 2026.*
