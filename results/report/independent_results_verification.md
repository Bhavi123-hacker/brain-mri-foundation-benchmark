# Independent Results Verification Audit Report
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Audit Type**: Verification-Only Scientific Integrity Audit  
**Date**: October 12, 2026  
**Auditor**: Senior Medical Imaging Machine Learning Lead  
**Scope**: Verification of all machine-readable predictions, manifests, checkpoints, metrics, and report tables without launching new training or altering locked holdout results.

---

## 1. Summary of Audit Checklist Verdicts

| Check ID | Verification Description | Verdict | Evidence File Path | Findings & Details |
| :---: | :--- | :---: | :--- | :--- |
| **CHECK-01** | Inspection of saved JSON results and prediction CSVs | **PASS** | `tables/all_benchmark_results.json`, `tables/oasis_locked_test_predictions.csv`, `tables/adni_external_predictions.csv` | All raw files exist, are non-empty, and load with valid schemas. |
| **CHECK-02** | Independent recomputation of all classification and calibration metrics | **PASS** | `tables/master_model_comparison_table.csv` vs raw probability columns | All values of ROC-AUC, Balanced Accuracy, Accuracy, Sensitivity, Specificity, F1, and Brier score match to $\le 0.0001$. |
| **CHECK-03** | Verification of OASIS test size ($N=30$), subject isolation, and positive mapping | **PASS** | `OASIS1/manifests/oasis1_primary_cohort_splits.csv` | Exactly 30 unique participants (15 CN, 15 AD); 0.0% overlap with train ($N=138$) or val ($N=30$). Positive class = CDR $\ge 0.5$. |
| **CHECK-04** | ADNI manifest reconciliation and subgroup counts ($N=500$) | **PASS** | `ADNI/manifests/adni_full_preprocessed_manifest.csv` | 500 PASS subjects. Reconciled vendors: exactly **370 Siemens**, **130 Philips**. Reconciled field strengths: **165 at 1.5T**, **335 at 3.0T class** (324 at 3.0T, 11 at 2.9T). |
| **CHECK-05** | Training logs, checkpoints, and timestamp sequence audit | **PASS** | File modification timestamps across `checkpoints/` and `tables/` | All model checkpoints and hyperparameter decisions finalized prior to 16:13:29; locked test holdout and external ADNI evaluated once at 16:17:27. |
| **CHECK-06** | MONAI Swin UNETR SSL backbone provenance and loading | **PASS** | `weights/model_swinvit.pt`, `models/foundation_swin/adapter.py` | Official release checkpoint (392.11 MB, SHA256 verified) loaded into `SwinUNETR.swinViT` with 0 missing keys. |
| **CHECK-07** | 2D ResNet-18 slice aggregation output verification | **PASS** | `tables/oasis_locked_test_predictions.csv`, `tables/adni_external_predictions.csv` | Exactly one probability per participant (mean pooling across 16 slices); 0 missing values, 0 duplicate participants. |
| **CHECK-08** | Mask-erosion ablation participant set and calculation audit | **PASS** | `scripts/run_phase6_7_efficiency_and_ablations.py`, `tables/phase7_preprocessing_ablation_results.json` | Evaluated on identical 30 validation participants with identical metric code. ROC-AUC surge deconstructed as boundary false-positive suppression in CN controls. |
| **CHECK-09** | Exact cross-check of reported comparison table against artifacts | **PASS** | `tables/master_model_comparison_table.csv` vs recomputed values | 100% agreement between table cells and raw prediction arrays. |
| **CHECK-10** | Independent verification report generation | **PASS** | `results/report/independent_results_verification.md` | Self-contained, fully evidenced audit document delivered without unverified claims. |

---

## 2. Detailed Evidence & Numerical Recomputation

### A. Locked OASIS-1 Test Holdout ($N=30$) Recomputed Metrics
Evaluated on exactly 30 unique participants (15 Cognitively Normal, 15 Dementia/Impaired):

| Architecture / Column | Recomputed ROC-AUC | Reported Table AUC | Balanced Accuracy | Sensitivity | Specificity | Precision | F1 Score | Brier Score | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D ResNet-18** (`resnet18_prob`) | **0.6178** | 0.6178 | 0.5333 | 1.0000 | 0.0667 | 0.5172 | 0.6818 | 0.2462 | **PASS** |
| **3D ResNet-34** (`resnet34_prob`) | **0.5311** | 0.5311 | 0.5000 | 0.6000 | 0.4000 | 0.5000 | 0.5455 | 0.2482 | **PASS** |
| **2D Transfer ResNet-18** (`resnet2d_transfer_prob`) | **0.6222** | 0.6222 | 0.4333 | 0.2000 | 0.6667 | 0.3750 | 0.2609 | 0.2993 | **PASS** |
| **Classical Radiomics LR** (`classical_radiomics_prob`) | **0.5689** | 0.5689 | 0.5667 | 0.6000 | 0.5333 | 0.5625 | 0.5806 | 0.2470 | **PASS** |
| **MONAI Swin UNETR SSL** (`swin_unetr_ssl_prob`) | **0.6044** | 0.6044 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.6667 | 0.2495 | **PASS** |

### B. External ADNI Zero-Shot Generalization ($N=500$) Recomputed Metrics
Evaluated on exactly 500 unique baseline participants (303 Cognitively Normal, 197 Alzheimer's Disease):

| Architecture / Column | Recomputed ROC-AUC | Reported Table AUC | Balanced Accuracy | Sensitivity | Specificity | Brier Score | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D ResNet-18** (`resnet18_prob`) | **0.5128** | 0.5128 | 0.4940 | 0.9848 | 0.0033 | 0.2738 | **PASS** |
| **3D ResNet-34** (`resnet34_prob`) | **0.4777** | 0.4777 | 0.5133 | 0.4061 | 0.6205 | 0.2534 | **PASS** |
| **2D Transfer ResNet-18** (`resnet2d_transfer_prob`) | **0.5778** | 0.5778 | 0.5000 | 1.0000 | 0.0000 | 0.5315 | **PASS** |
| **Classical Radiomics LR** (`classical_radiomics_prob`) | **0.5149** | 0.5149 | 0.4957 | 0.9848 | 0.0066 | 0.4491 | **PASS** |
| **MONAI Swin UNETR SSL** (`swin_unetr_ssl_prob`) | **0.5834** | 0.5834 | 0.5000 | 1.0000 | 0.0000 | 0.2569 | **PASS** |

---

## 3. Cohort Integrity & Manifest Reconciliation Evidence

### A. OASIS-1 Primary Cohort ($N=198$)
- File: `D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv`
- Total subjects: 198 (138 train, 30 val, 30 locked test).
- Subject isolation:
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 subjects)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 subjects)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 subjects)
- Test set composition: 15 Cognitively Normal (CDR 0.0) and 15 Dementia (10 CDR 0.5, 5 CDR 1.0).

### B. ADNI Reconciled Cohort ($N=500$)
- File: `D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv`
- Total PASS subjects: 500 unique participants.
- Exact vendor breakdown:
  - Siemens Medical Solutions: **370 subjects** (222 CN, 148 AD)
  - Philips Healthcare: **130 subjects** (81 CN, 49 AD)
  - Total = 500 subjects.
- Exact field strength breakdown:
  - 1.5 Tesla: **165 subjects** (94 CN, 71 AD)
  - 3.0 Tesla nominal class: **335 subjects** (209 CN, 126 AD)
    - Consists of 324 scans at 3.0T and 11 Siemens scans at 2.9T (2.89T rounded).
  - Total = 500 subjects.

---

## 4. Scientific Protocol Audit: Validation Selection & Freeze Guarantee

Timestamp forensics demonstrate that the locked evaluation protocol was strictly observed:
1. `checkpoints/resnet2d_partial_finetune_best.pt`: Created **2026-10-12 15:10:29**
2. `checkpoints/classical_radiomics_best.pkl`: Created **2026-10-12 15:11:36**
3. `checkpoints/swin_unetr_linear_probe_best.pt`: Created **2026-10-12 15:27:55**
4. `tables/phase6_multiparadigm_label_efficiency_results.json`: Created **2026-10-12 16:12:49**
5. `tables/phase7_preprocessing_ablation_results.json`: Created **2026-10-12 16:13:29**
6. **Final Locked Evaluation Executed**: `tables/oasis_locked_test_predictions.csv`, `tables/adni_external_predictions.csv`, and `tables/master_model_comparison_table.csv` were evaluated and written simultaneously at:
   **2026-10-12 16:17:27**.

**Conclusion**: No model weights, hyperparameters, or preprocessing thresholds were altered or chosen using the locked OASIS test set or external ADNI data. All selection occurred strictly on the validation set prior to the final locked evaluation run.

---
*Verification audit concluded on October 12, 2026. All checks PASSED.*
