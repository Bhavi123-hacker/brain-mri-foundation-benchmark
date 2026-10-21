# Pre-Publication Statistical, Uncertainty & Calibration Audit Summary
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Author / Auditor**: Senior Medical Imaging Machine Learning Lead  
**Audit Scope**: Statistical reproducibility, paired bootstrap confidence bounds, multiple-comparison corrections, calibration drift deconstruction, and historical test exposure audit.  
**Execution Environment**: Local NVIDIA RTX PRO 2000 Blackwell (16 GB VRAM, sm_120), Python 3.12.10, PyTorch 2.11.0+cu128.  
**Artifact Dependencies**: 
- Predictions: [`results/report/oasis_locked_test_predictions.csv`](file:///D:/ml%20project/results/report/oasis_locked_test_predictions.csv), [`results/report/adni_external_predictions.csv`](file:///D:/ml%20project/results/report/adni_external_predictions.csv)
- Master Table: [`results/report/master_model_comparison_table.csv`](file:///D:/ml%20project/results/report/master_model_comparison_table.csv)
- Paired Diff Tables: [`results/statistical_audit/oasis_test_paired_auc_differences_holm_corrected.csv`](file:///D:/ml%20project/results/statistical_audit/oasis_test_paired_auc_differences_holm_corrected.csv), [`results/statistical_audit/adni_external_paired_auc_differences_holm_corrected.csv`](file:///D:/ml%20project/results/statistical_audit/adni_external_paired_auc_differences_holm_corrected.csv)
- Calibration Profile: [`results/statistical_audit/threshold_and_calibration_profiles.csv`](file:///D:/ml%20project/results/statistical_audit/threshold_and_calibration_profiles.csv)

---

## 1. Metric Reproduction & Integrity Verification

All reported point metrics across all 6 evaluated model pipelines were independently recomputed from the raw, subject-level prediction CSVs. Zero rounding errors, data truncation, or discrepancies were detected:

| Evaluation Split | Model Architecture | Stored ROC-AUC | Recomputed ROC-AUC | Stored BalAcc | Recomputed BalAcc | Stored Brier | Recomputed Brier | Match Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OASIS Locked Test ($N=30$)** | 3D ResNet-18 (Scratch) | 0.6178 | 0.6178 | 0.5333 | 0.5333 | 0.2462 | 0.2462 | **EXACT MATCH** |
| **OASIS Locked Test ($N=30$)** | 3D ResNet-34 (Scratch) | 0.5311 | 0.5311 | 0.5000 | 0.5000 | 0.2482 | 0.2482 | **EXACT MATCH** |
| **OASIS Locked Test ($N=30$)** | 2D Transfer ResNet-18 | 0.6222 | 0.6222 | 0.4333 | 0.4333 | 0.2993 | 0.2993 | **EXACT MATCH** |
| **OASIS Locked Test ($N=30$)** | Classical Radiomics (SVM) | 0.5689 | 0.5689 | 0.5667 | 0.5667 | 0.2470 | 0.2470 | **EXACT MATCH** |
| **OASIS Locked Test ($N=30$)** | MONAI Swin UNETR SSL | 0.6044 | 0.6044 | 0.5000 | 0.5000 | 0.2495 | 0.2495 | **EXACT MATCH** |
| **OASIS Locked Test ($N=30$)** | FOMO25 AMAES ResEnc-B | 0.5733 | 0.5733 | 0.5333 | 0.5333 | 0.2512 | 0.2512 | **EXACT MATCH** |
| **ADNI External ($N=500$)** | 3D ResNet-18 (Scratch) | 0.5128 | 0.5128 | 0.4940 | 0.4940 | 0.2738 | 0.2738 | **EXACT MATCH** |
| **ADNI External ($N=500$)** | 3D ResNet-34 (Scratch) | 0.4777 | 0.4777 | 0.5133 | 0.5133 | 0.2534 | 0.2534 | **EXACT MATCH** |
| **ADNI External ($N=500$)** | 2D Transfer ResNet-18 | 0.5778 | 0.5778 | 0.5000 | 0.5000 | 0.5315 | 0.5315 | **EXACT MATCH** |
| **ADNI External ($N=500$)** | Classical Radiomics (SVM) | 0.5149 | 0.5149 | 0.4957 | 0.4957 | 0.4491 | 0.4491 | **EXACT MATCH** |
| **ADNI External ($N=500$)** | MONAI Swin UNETR SSL | 0.5834 | 0.5834 | 0.5000 | 0.5000 | 0.2569 | 0.2569 | **EXACT MATCH** |
| **ADNI External ($N=500$)** | FOMO25 AMAES ResEnc-B | 0.5054 | 0.5054 | 0.5000 | 0.5000 | 0.3442 | 0.3442 | **EXACT MATCH** |

---

## 2. Paired Bootstrap Statistical Methodology

To control for subject sampling variability and provide unbiased uncertainty intervals, we implemented paired non-parametric bootstrap resampling ($B = 2,000$ iterations) using fixed seed `42`:
1. In each bootstrap replicate $b \in \{1, \dots, B\}$, a random sample of $N$ subject indices is drawn **with replacement** from the evaluation cohort.
2. For every candidate model pair $(A, B)$, predictions are evaluated on the **exact same bootstrap sample**, preserving intra-subject error correlations:
   $$\Delta \text{AUC}_{A-B}^{(b)} = \text{AUC}_A^{(b)} - \text{AUC}_B^{(b)}$$
3. Two-sided empirical $p$-values are calculated as:
   $$p = 2 \times \min\left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta \text{AUC}^{(b)} \le 0), \, \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta \text{AUC}^{(b)} \ge 0) \right)$$
4. Two-sided 95% confidence intervals are computed from the 2.5th and 97.5th percentiles of the empirical bootstrap distribution.

---

## 3. Multiple-Comparison Corrections (Holm-Bonferroni FWER Control)

In benchmarks evaluating $K=6$ models, there are $m = \binom{6}{2} = 15$ pairwise hypothesis tests. Reporting unadjusted $p$-values creates substantial inflation of the Family-Wise Error Rate ($\text{FWER} \le 1 - (1 - 0.05)^{15} \approx 53.7\%$).

We applied the sequentially rejective **Holm-Bonferroni method** to control the FWER at $\alpha = 0.05$:
- Sort raw $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
- For index $i \in \{1, \dots, m\}$, adjusted $p$-value is:
  $$p_{\text{Holm},(i)} = \min\left(1, \, \max_{k \le i} \left[ (m - k + 1) p_{(k)} \right] \right)$$

### A. OASIS Locked Test Cohort ($N=30$)

| Pairwise Comparison (Model A vs Model B) | Median $\Delta \text{AUC}$ | 95% Bootstrap CI | Raw $p$-value | Holm-Adjusted $p$-value | Statistically Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 3D ResNet-34 vs 2D Transfer ResNet-18 | -0.0889 | [-0.3416, +0.1701] | 0.4930 | 1.0000 | **No** |
| 2D Transfer ResNet-18 vs Classical Radiomics | +0.0578 | [-0.1111, +0.2188] | 0.5060 | 1.0000 | **No** |
| 3D ResNet-18 vs 3D ResNet-34 | +0.0848 | [-0.2536, +0.4097] | 0.6420 | 1.0000 | **No** |
| 3D ResNet-34 vs MONAI Swin UNETR SSL | -0.0741 | [-0.3820, +0.2625] | 0.6700 | 1.0000 | **No** |
| 2D Transfer ResNet-18 vs FOMO25 AMAES | +0.0490 | [-0.2081, +0.2946] | 0.6860 | 1.0000 | **No** |
| 3D ResNet-18 vs Classical Radiomics | +0.0448 | [-0.2176, +0.3068] | 0.7430 | 1.0000 | **No** |
| 3D ResNet-18 vs FOMO25 AMAES | +0.0407 | [-0.1897, +0.2625] | 0.7470 | 1.0000 | **No** |
| MONAI Swin UNETR vs FOMO25 AMAES | +0.0282 | [-0.1806, +0.2356] | 0.7890 | 1.0000 | **No** |
| 3D ResNet-34 vs FOMO25 AMAES | -0.0412 | [-0.3527, +0.2632] | 0.7940 | 1.0000 | **No** |
| Classical Radiomics vs MONAI Swin UNETR | -0.0335 | [-0.2804, +0.2127] | 0.7970 | 1.0000 | **No** |
| 3D ResNet-18 vs MONAI Swin UNETR SSL | +0.0111 | [-0.0711, +0.1025] | 0.8090 | 1.0000 | **No** |
| 3D ResNet-34 vs Classical Radiomics | -0.0356 | [-0.3254, +0.2501] | 0.8420 | 1.0000 | **No** |
| 2D Transfer ResNet-18 vs MONAI Swin UNETR | +0.0201 | [-0.2311, +0.2800] | 0.8650 | 1.0000 | **No** |
| Classical Radiomics vs FOMO25 AMAES | -0.0089 | [-0.2443, +0.2356] | 0.9520 | 1.0000 | **No** |
| 3D ResNet-18 vs 2D Transfer ResNet-18 | -0.0089 | [-0.2823, +0.2711] | 0.9570 | 1.0000 | **No** |

> [!IMPORTANT]
> **Audit Conclusion on OASIS Test**: **All 15 paired confidence intervals cross zero, and all Holm-adjusted $p$-values equal 1.0000.** The locked OASIS test set ($N=30$) lacks the statistical power to declare any architecture superior.

---

### B. ADNI External Cohort ($N=500$)

| Pairwise Comparison (Model A vs Model B) | Median $\Delta \text{AUC}$ | 95% Bootstrap CI | Raw $p$-value | Holm-Adjusted $p$-value | Statistically Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3D ResNet-34 vs MONAI Swin UNETR SSL** | **-0.1060** | **[-0.1762, -0.0364]** | **0.0010** | **0.0150** | **YES (Swin Superior)** |
| 3D ResNet-34 vs 2D Transfer ResNet-18 | -0.1011 | [-0.1789, -0.0231] | 0.0110 | 0.1540 | No (Nominal only) |
| 2D Transfer ResNet-18 vs FOMO25 AMAES | +0.0737 | [+0.0125, +0.1318] | 0.0140 | 0.1820 | No (Nominal only) |
| MONAI Swin UNETR vs FOMO25 AMAES | +0.0778 | [+0.0102, +0.1494] | 0.0280 | 0.3360 | No (Nominal only) |
| 3D ResNet-18 vs 2D Transfer ResNet-18 | -0.0660 | [-0.1247, -0.0020] | 0.0410 | 0.4510 | No (Nominal only) |
| Classical Radiomics vs MONAI Swin UNETR | -0.0684 | [-0.1386, +0.0037] | 0.0600 | 0.6000 | No |
| 2D Transfer ResNet-18 vs Classical Radiomics | +0.0623 | [-0.0060, +0.1345] | 0.0760 | 0.6840 | No |
| 3D ResNet-18 vs MONAI Swin UNETR SSL | -0.0709 | [-0.1478, +0.0081] | 0.0790 | 0.6840 | No |
| 3D ResNet-34 vs Classical Radiomics | -0.0384 | [-0.1045, +0.0270] | 0.2600 | 1.0000 | No |
| 3D ResNet-18 vs 3D ResNet-34 | +0.0370 | [-0.0420, +0.1124] | 0.3530 | 1.0000 | No |
| 3D ResNet-34 vs FOMO25 AMAES ResEnc-B | -0.0286 | [-0.0997, +0.0469] | 0.4580 | 1.0000 | No |
| Classical Radiomics vs FOMO25 AMAES | +0.0106 | [-0.0605, +0.0778] | 0.7630 | 1.0000 | No |
| 3D ResNet-18 vs FOMO25 AMAES ResEnc-B | +0.0083 | [-0.0591, +0.0730] | 0.8070 | 1.0000 | No |
| 2D Transfer ResNet-18 vs MONAI Swin UNETR | -0.0054 | [-0.0702, +0.0603] | 0.8610 | 1.0000 | No |
| 3D ResNet-18 vs Classical Radiomics | -0.0024 | [-0.0631, +0.0602] | 0.9480 | 1.0000 | No |

> [!TIP]
> **Audit Conclusion on ADNI External ($N=500$)**:
> - Only **MONAI Swin UNETR SSL vs 3D ResNet-34** retains statistical significance after strict Holm-Bonferroni correction ($p_{\text{Holm}} = 0.0150$).
> - Four comparisons (2D Transfer vs 3D ResNet-34, 2D Transfer vs FOMO AMAES, MONAI Swin UNETR vs FOMO AMAES, and 2D Transfer vs 3D ResNet-18) show **nominal significance** ($p_{\text{raw}} < 0.05$) with strictly positive/negative 95% CIs, but do not meet family-wise significance after penalizing for 15 simultaneous tests.
> - **FOMO25 AMAES ResEnc-B is indistinguishable from 3D ResNet-18 from scratch** ($p_{\text{raw}} = 0.8070$).

---

## 4. Discrimination vs Calibration Deconstruction

### The ADNI Extreme Operating Point Artifact
On the external ADNI dataset ($N=500$), evaluating models at the standard default decision threshold $t = 0.50$ produces degenerate operating characteristics for three models:
- **2D Transfer ResNet-18**: Sensitivity = 1.0000, Specificity = 0.0000, Balanced Accuracy = 0.5000, Brier Score = 0.5315.
- **MONAI Swin UNETR SSL**: Sensitivity = 1.0000, Specificity = 0.0000, Balanced Accuracy = 0.5000, Brier Score = 0.2569.
- **FOMO25 AMAES ResEnc-B**: Sensitivity = 1.0000, Specificity = 0.0000, Balanced Accuracy = 0.5000, Brier Score = 0.3442.

### Probability Distribution Drift Across Domains:
```text
Cohort / Model                       Mean Prob    Min Prob    Max Prob    IQR
---------------------------------------------------------------------------------
OASIS Test (2D Transfer ResNet-18)    0.4185      0.0676      0.9716      [0.21, 0.65]
ADNI Ext   (2D Transfer ResNet-18)    0.9398      0.6837      0.9997      [0.91, 0.98]
OASIS Test (FOMO25 AMAES ResEnc-B)    0.4414      0.2958      0.6353      [0.37, 0.51]
ADNI Ext   (FOMO25 AMAES ResEnc-B)    0.7157      0.5706      0.8316      [0.67, 0.76]
OASIS Test (MONAI Swin UNETR SSL)     0.5148      0.5063      0.5215      [0.51, 0.52]
ADNI Ext   (MONAI Swin UNETR SSL)     0.5292      0.5273      0.5312      [0.528, 0.530]
```

### Forensic Diagnosis:
1. **Severe Multi-Center Calibration Drift**: 
   When models trained on 1.5T single-center OASIS scans are deployed zero-shot on 500 multi-center ADNI scans, physical differences in scanner hardware, gradient coils, RF receive arrays, and background soft tissue cause the logit outputs to drift strongly positive.
2. **Threshold Mismatch**:
   Because the minimum predicted probability for 2D ResNet-18 on ADNI is **0.6837** and for FOMO AMAES is **0.5706**, **every single patient exceeds $t=0.50$**. Consequently, the confusion matrix has $\text{TN} = 0$ and $\text{FP} = 303$, yielding sensitivity of $1.0000$ and specificity of $0.0000$.
3. **Exploratory Post-Hoc Thresholding (Youden's Index)**:
   When thresholds are adjusted post-hoc to optimize Youden's $J = \text{Sens} + \text{Spec} - 1$:
   - 2D Transfer ResNet-18 at $t^* = 0.96$: Balanced Accuracy recovers to **0.5741** (Sensitivity 0.5838, Specificity 0.5644).
   - FOMO25 AMAES at $t^* = 0.69$: Balanced Accuracy recovers to **0.5146** (Sensitivity 0.7157, Specificity 0.3135).
   - Swin UNETR at $t^* = 0.53$: Balanced Accuracy recovers to **0.5246** (Sensitivity 0.2538, Specificity 0.7954).
4. **Mandatory Reporting Boundary**:
   **Youden thresholds chosen on ADNI are post-hoc exploratory analyses and must NEVER be reported as unbiased zero-shot generalization.** ROC-AUC is invariant to monotonic probability shifts and must remain the authoritative metric for ranking representation discrimination.

---

## 5. Historical Test Exposure Reconciliation

```
+----------------------------------------------------------------------------------------------------+
|                                    HISTORICAL EVALUATION TIMELINE                                  |
+------------------------------+--------------------------------------+------------------------------+
| 2026-10-09 15:00:03          | 2026-10-12 13:01:00                  | 2026-10-12 16:17:27          |
| Full ResNet-18 Baseline Run  | Master Benchmark Campaign            | Two-Hour Sprint Final Eval   |
| Test evaluated in log        | Test evaluated in campaign           | Final locked test evaluation |
| Test AUC = 0.6044            | Test AUC = 0.6178                    | Evaluated all models once    |
+------------------------------+--------------------------------------+------------------------------+
```

### Forensic Reconciliation:
- Early reports asserted the test set was "completely untouched prior to 16:17:27 on October 12".
- Git commit logs and json records reveal that supervised 3D ResNet-18 test performance was observed on October 9 (AUC 0.6044) and early October 12 (AUC 0.6178).
- **Integrity Clarification**: While new architectures (2D Transfer, Radiomics, Swin UNETR, FOMO linear probe) were fitted strictly on train data and tuned strictly on validation data without test snooping, the *supervised baseline* had prior test visibility. Fully disclosing this timeline maintains scientific transparency.

---
*Summary compiled and verified by Senior Medical Imaging ML Lead on October 21, 2026.*
