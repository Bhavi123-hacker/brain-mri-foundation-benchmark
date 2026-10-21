# SHIFT-MRI: Definitive Research Source of Truth (Canonical Release)

**Document Version:** 3.0.0-final  
**Release Date:** October 2026  
**Audited Commit:** `435daf0`  
**Research Branch:** `research/v3-final-ui-paper-submission`  
**License:** Apache 2.0  

---

## 1. Primary Mission and Scientific Principles

The **SHIFT-MRI Benchmark** establishes a scientifically disciplined, open-source framework for cross-cohort reliability and failure analysis of 3D Brain-MRI Foundation Models versus well-tuned supervised deep learning, transfer learning, classical radiomics, and dual-branch hybrid architectures.

Every metric, figure, and claim in this repository is bound to the following single source of truth rules:
1. **Zero Test-Label Leakage:** Decision thresholds ($t^*$) are derived strictly on authentic source validation predictions (`results/v3/v3_oasis_validation_predictions.csv`, $N=30$) using Youden's J statistic ($y_{\text{val}}$). Test and external cohort labels are never accessed during threshold selection or calibration.
2. **Reconstructed Validation TrustGate:** Rejection threshold $\tau^* = 0.6340$ is calibrated strictly on the 80th percentile of authentic OASIS-1 validation risk scores ($S_A^{\text{val}}, S_B^{\text{val}}, S_C^{\text{val}}, S_D^{\text{val}}$). Zero test data is mixed into threshold determination.
3. **Matched Demographic Cohorts:** Demographic baselines are evaluated across the complete matched 500-subject ADNI cohort.
4. **Honest Reporting of Negative Results:** The multi-signal TrustGate composite (AUC = 0.4697) underperforms simple Classification Margin alone (AUC = 0.5829) on external ADNI due to consensus inversion under prediction collapse. This negative/parity result is reported as an empirical finding.
5. **No Speculative Causal Claims:** Observable distribution shifts are described as associative and empirical findings.

---

## 2. Canonical Model Rankings on External Multi-Site ADNI ($N=500$)

Evaluated under source-locked validation-derived decision thresholds ($t^*$):

| Rank | Model Name | Architectural Paradigm | Parameter Count | Checkpoint / Weights Source | ADNI Ext ROC-AUC [95% CI] | ADNI Ext Brier Score |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: |
| **1** | **MONAI Swin UNETR SSL** | 3D Self-Supervised ViT (Linear Probe) | 8,062,771 | MONAI SSL (`model_swinvit.pt`) | **0.5834** [0.5331, 0.6337] | 0.2569 |
| **2** | **2D Transfer ResNet-18** | 2D Slice-Aggregated Transfer | 11,177,025 | Torchvision ImageNet-1K V1 | **0.5778** [0.5280, 0.6276] | 0.5315 |
| **3** | **Dual-Branch Hybrid** | ResNet-10 + Morphometric Proxies | 14,525,153 | Supervised Scratch + Proxies | **0.5554** [0.5044, 0.6066] | 0.3163 |
| **4** | **3D ResNet-10** | Lightweight Supervised 3D CNN | 14,619,585 | Supervised Scratch (Pipeline B) | **0.5444** [0.4909, 0.5953] | 0.5724 |
| **—** | *Demographic Baseline: Age Alone* | Tabular Logistic Regression | 1 | Non-Imaging Tabular Prior | *0.5242 [0.4727, 0.5753]* | *0.2516* |
| **5** | **Classical Radiomics** | Linear SVM on 25 Morphometric Features | 25 | Engineered Volume/Texture | **0.5149** [0.4638, 0.5660] | 0.4491 |
| **6** | **3D ResNet-18 (Scratch)** | Supervised 3D CNN Baseline | 33,160,513 | Random (MedicalNet init) | **0.5128** [0.4612, 0.5644] | 0.2738 |
| **—** | *Demographic Baseline: Age + Sex* | Tabular Logistic Regression | 2 | Non-Imaging Tabular Prior | *0.5123 [0.4590, 0.5650]* | *0.2629* |
| **7** | **FOMO25 AMAES ResEnc-B** | 3D Foundation Model (Linear Probe) | 90,296,897 | FOMO-50K (`resenc_unet_b.ckpt`) | **0.5054** [0.4516, 0.5552] | 0.3442 |
| **8** | **3D DenseNet-121** | Dense Feature Reuse 3D CNN | 11,767,937 | Supervised Scratch (Pipeline B) | **0.5048** [0.4537, 0.5592] | 0.2603 |
| **9** | **2.5D Multi-View ResNet-18** | Triplanar Orthogonal Transfer | 11,570,241 | Torchvision ImageNet-1K V1 | **0.4943** [0.4434, 0.5464] | 0.2654 |
| **10** | **3D ResNet-34 (Scratch)** | Heavy Supervised 3D CNN Baseline | 63,489,217 | Random (MedicalNet init) | **0.4777** [0.4261, 0.5293] | 0.2534 |

---

## 3. Canonical Dataset Manifest

| Cohort Name | Sample Count ($N$) | Cohort Role | Scanners & Field Strengths | Preprocessing Pipeline | Governance & Overlap Status |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **OASIS-1** | 198 (138 tr, 30 val, 30 test) | Primary In-Domain Benchmark | Siemens Vision 1.5T | Trilinear $128^3$ Resampling, Nonzero Z-Score | Washington University ADRC; fully disjoint participant splits |
| **ADNI** | 500 (229 CN, 271 AD) | Zero-Shot Multi-Site External | GE, Siemens, Philips (1.5T & 3.0T) | Matched Resampling $128^3$, Nonzero Z-Score | Multi-site DUA; fully independent external validation cohort |
| **OASIS-2** | 150 (85 CN, 65 AD) | Secondary Exploratory Cohort | Siemens Vision 1.5T | Matched Resampling $128^3$, Nonzero Z-Score | Washington University ADRC; **unverified participant overlap** with OASIS-1 |

---

## 4. Canonical TrustGate Failure-Detection Results

Evaluated on primary model MONAI Swin UNETR SSL at validation threshold $t^* = 0.5088$ (error rate = 60.6% on ADNI):

| Method / Risk Estimator | Error-Detection ROC-AUC [95% CI] | $\text{pAURC}_{[0.5, 1.0]}$ | Full AURC $[0.0, 1.0]$ | Theoretical / Empirical Status |
| :--- | :---: | :---: | :---: | :--- |
| **Classification Margin Risk** | **0.5829** [0.5337, 0.6306] | **0.2871** | **0.5488** | Strongest failure detector; reduces error from 60.6% to 54.0% at 50% coverage |
| **Maximum Probability (MSP)** | **0.5829** [0.5337, 0.6306] | **0.2871** | **0.5488** | Mathematically rank-equivalent to Margin Risk under binary classification |
| **Predictive Entropy** | **0.5829** [0.5337, 0.6306] | **0.2871** | **0.5488** | Mathematically rank-equivalent to Margin Risk under binary classification |
| **Feature Distance (Raw Mahalanobis)** | 0.4808 [0.4288, 0.5332] | N/A | N/A | Sub-chance within-cohort error discrimination |
| **Feature Distance (Clipped / Saturated)** | 0.5000 [0.5000, 0.5000] | 0.3088 | 0.6470 | 100% of ADNI samples saturate to 1.0000 (zero variance $\implies$ AUC 0.5000) |
| **Model Disagreement (6 Archs)** | 0.4674 [0.4155, 0.5190] | 0.3086 | 0.6481 | Inverted consensus under positive collapse (agreement on false positives) |
| **Proposed TrustGate Composite** | **0.4697** [0.4172, 0.5221] | **0.3088** | **0.6470** | **Negative / Parity Result:** Composite degraded by disagreement and saturation |

---

## 5. Canonical File and Artifact Registry

- **Master Reliability Matrix:** `results/v3/reliability/cross_cohort_reliability_matrix.csv`
- **Prediction Collapse Table:** `results/v3/failure_analysis/prediction_collapse_table.csv`
- **Demographic Baselines:** `results/v3/demographic_baselines/demographic_baselines_results.csv`
- **Selective Abstention:** `results/v3/abstention/abstention_metrics.csv`
- **TrustGate Benchmark:** `results/v3/trustgate/trustgate_benchmark_results.csv`
- **TrustGate Ablations:** `results/v3/trustgate_ablation/trustgate_ablation_results.csv`
- **TrustGate Scanner Subgroups:** `results/v3/trustgate/trustgate_scanner_subgroup_analysis.csv`
- **OOD Evaluation:** `results/v3/ood_detection/ood_detection_evaluation.csv`
- **Complete Research Paper (PDF):** `docs/submission/final/SHIFT_MRI_Complete_Research_Paper.pdf`
- **Complete Research Paper (DOCX):** `docs/submission/final/SHIFT_MRI_Complete_Research_Paper.docx`
- **Final Presentation (PPTX):** `docs/submission/final/SHIFT_MRI_Final_Presentation.pptx`
- **Automated Test Suite:** `tests/test_v3_shift_mri_integrity.py` (90 passed, 4 skipped)
