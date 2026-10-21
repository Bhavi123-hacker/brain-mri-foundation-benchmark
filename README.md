# SHIFT-MRI: Source-Locked, Label-Aware Cross-Cohort Reliability and Failure Analysis of Brain MRI Foundation Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11+cu128-red.svg)](https://pytorch.org/)
[![MONAI 1.6](https://img.shields.io/badge/MONAI-1.6.1-brightgreen.svg)](https://monai.io/)
[![Hardware: NVIDIA RTX PRO 2000 Blackwell](https://img.shields.io/badge/GPU-RTX_PRO_2000_Blackwell_16GB-green.svg)](https://www.nvidia.com/)
[![Tests: 90 Passed](https://img.shields.io/badge/Tests-90%20Passed-brightgreen.svg)](tests/)
[![Branch: research/v3-final-ui-paper-submission](https://img.shields.io/badge/Branch-research%2Fv3--final--ui--paper--submission-orange.svg)](https://github.com/Bhavi123-hacker/brain-mri-foundation-benchmark)

**SHIFT-MRI** is a rigorous, multi-cohort neuroimaging AI benchmark and reliability framework investigating whether large-scale 3D brain MRI foundation models (e.g., FOMO-25, MONAI Swin UNETR SSL) outperform well-tuned supervised 3D CNNs, slice-aggregated 2D transfer networks, classical radiomics, and dual-branch hybrid architectures under matched labeled-data budgets and realistic cross-scanner domain shifts. 

Going beyond discrimination alone, SHIFT-MRI audits **decision threshold collapse**, **calibration drift**, **demographic shortcuts**, **clinical label sensitivity (CDR 0.5 vs 1.0+)**, **scanner perturbation robustness**, **prospective out-of-distribution (OOD) detection**, and introduces **SHIFT-MRI TrustGate**, a source-locked failure-risk estimation prototype with full disclosure of an empirical negative/parity result.

---

## 📊 Definitive Evaluated Model Rankings (External Multi-Site ADNI $N=500$)

All models trained strictly on **OASIS-1** ($N=198$; 138 train, 30 validation, 30 locked test) and evaluated zero-shot across **ADNI** ($N=500$; 229 CN, 271 AD; GE, Siemens, Philips; 1.5T and 3.0T) using source-locked thresholds ($t^*$) derived strictly from validation data:

| Rank | Model Name | Architectural Family | Parameters | Weights Source | ADNI Ext AUC [95% CI] | ADNI Ext Brier | OASIS-1 Test AUC [95% CI] | OASIS-1 Test BalAcc |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **MONAI Swin UNETR SSL** | 3D Self-Supervised ViT (Linear Probe) | 8,062,771 | MONAI SSL (`model_swinvit.pt`) | **0.5834** [0.5331, 0.6337] | 0.2569 | **0.6044** [0.3956, 0.8044] | 0.5333 |
| **2** | **2D Transfer ResNet-18** | 2D Slice-Aggregated Transfer | 11,177,025 | ImageNet-1K V1 | **0.5778** [0.5280, 0.6276] | 0.5315 | **0.6222** [0.4133, 0.8178] | 0.4333 |
| **3** | **Dual-Branch Hybrid** | ResNet-10 + Handcrafted Proxies | 14,525,153 | Scratch + Radiomics Proxies | **0.5554** [0.5044, 0.6066] | 0.3163 | **0.6311** [0.4152, 0.8170] | 0.5000 |
| **4** | **3D ResNet-10** | Lightweight Supervised 3D CNN | 14,619,585 | Random Scratch | **0.5444** [0.4909, 0.5953] | 0.5724 | **0.6133** [0.3973, 0.8145] | 0.5667 |
| **—** | *Demographic Baseline: Age Alone* | Tabular Logistic Regression | 1 | Non-Imaging Tabular Prior | *0.5242 [0.4727, 0.5753]* | *0.2516* | *0.5303 [0.3178, 0.7444]* | *0.5000* |
| **5** | **Classical Radiomics** | Linear SVM on 25 Morphometric Features | 25 | Engineered Volume/Texture | **0.5149** [0.4638, 0.5660] | 0.4491 | **0.5689** [0.3600, 0.7689] | 0.5667 |
| **6** | **3D ResNet-18 (Scratch)** | Supervised 3D CNN Baseline | 33,160,513 | Random (MedicalNet init) | **0.5128** [0.4612, 0.5644] | 0.2738 | **0.6178** [0.4133, 0.8133] | 0.5000 |
| **—** | *Demographic Baseline: Age + Sex* | Tabular Logistic Regression | 2 | Non-Imaging Tabular Prior | *0.5123 [0.4590, 0.5650]* | *0.2629* | *0.5280 [0.3111, 0.7467]* | *0.5333* |
| **7** | **FOMO25 AMAES ResEnc-B** | 3D Foundation Model (Linear Probe) | 90,296,897 | FOMO-50K (`resenc_unet_b.ckpt`) | **0.5054** [0.4516, 0.5552] | 0.3442 | **0.5733** [0.3664, 0.7989] | 0.5000 |
| **8** | **3D DenseNet-121** | Dense Feature Reuse 3D CNN | 11,767,937 | Random Scratch | **0.5048** [0.4537, 0.5592] | 0.2603 | **0.4133** [0.2187, 0.6269] | 0.4000 |
| **9** | **2.5D Multi-View ResNet-18** | Triplanar Orthogonal Transfer | 11,570,241 | ImageNet-1K V1 | **0.4943** [0.4434, 0.5464] | 0.2654 | **0.6444** [0.4261, 0.8300] | 0.5333 |
| **10** | **3D ResNet-34 (Scratch)** | Heavy Supervised 3D CNN Baseline | 63,489,217 | Random (MedicalNet init) | **0.4777** [0.4261, 0.5293] | 0.2534 | **0.5311** [0.3200, 0.7333] | 0.5333 |

---

## 🔬 Key Scientific Discoveries & Methodological Novelties (N01–N07)

1. **N01: Cross-Cohort Reliability Matrix:** Cross-site validation drops discrimination from in-domain validation ($\text{AUC} \sim 0.60–0.83$) to external ADNI ($\text{AUC} \sim 0.48–0.58$), accompanied by substantial calibration drift and positive Brier degradation across all models (`results/v3/reliability/cross_cohort_reliability_matrix.csv`).
2. **N02: Prediction Collapse Under Domain Shift:** 7 out of 10 evaluated architectures exhibit catastrophic decision collapse under default thresholds ($t=0.50$), predicting 99.0%–100% of external scans as positive (AD), resulting in balanced accuracy collapsing to 0.5000 (`results/v3/failure_analysis/prediction_collapse_table.csv`).
3. **N03: Demographic Shortcut Verification:** On the complete matched 500-subject ADNI cohort, non-imaging Age+Sex logistic regression yields near-chance discrimination ($\text{AUC} = \mathbf{0.5123}$). Deep vision models (Swin UNETR at **0.5834** and 2D Transfer at **0.5778**) demonstrate genuine image-derived structural signal beyond demographic shortcuts, definitively resolving the earlier 50-patient pilot artifact (which had yielded an inflated 0.5671).
4. **N04: Confidence-Aware Selective Abstention:** Abstaining on uncertain predictions based on decision-boundary distance $|p(x) - t^*|$ monotonically drives down diagnostic error rate on ADNI from 60.6% down to **54.0%** at 50% coverage ($\text{pAURC}_{[0.5, 1.0]} = \mathbf{0.2871}$, Full $\text{AURC} = \mathbf{0.5488}$; `results/v3/abstention/abstention_metrics.csv`).
5. **N05: Clinical Stage Sensitivity Bottleneck:** Detecting very mild cognitive impairment (CDR 0.5) is severely impaired in scratch 3D CNNs (sensitivity 20%–70%), whereas the Dual-Branch Hybrid achieves **86.7% sensitivity on CDR 0.5** by integrating explicit morphometric proxies.
6. **N06: Scanner & Image Perturbation Robustness:** Controlled evaluations under Gaussian sensor noise ($\sigma \in [0.0, 0.10]$) and intensity scaling ($0.85\times - 1.15\times$) demonstrate architectural resilience variations across 3D and 2.5D models (`results/v3/robustness/perturbation_robustness_results.csv`).
7. **N07: Prospective Out-of-Distribution Detection:** Latent regularized Mahalanobis distance in FOMO25 feature space prospectively separates in-domain OASIS-1 scans from external ADNI scans (**OOD ROC-AUC = 1.0000**), while centroid Euclidean distance achieves **0.9930**, prospectively locked without post-hoc label knowledge (`results/v3/ood_detection/ood_detection_evaluation.csv`).

---

## 🛡️ SHIFT-MRI TrustGate: Prototype & Negative Result Disclosure

**SHIFT-MRI TrustGate** investigated a source-locked failure-risk estimator combining:
- **Component A ($S_A$):** Classification Margin / Entropy Risk
- **Component B ($S_B$):** Regularized Latent Feature Distance (FOMO25 Mahalanobis)
- **Component C ($S_C$):** Multi-Model Architectural Disagreement (6 diverse architectures)
- **Component D ($S_D$):** Acquisition / Image QC Anomaly

### Empirical Verdict on External ADNI ($N=500$):
- **Classification Margin Alone:** Error-detection ROC-AUC = **0.5829** [95% CI: 0.5337, 0.6306], $\text{pAURC}_{[0.5, 1.0]} = \mathbf{0.2871}$.
- **Proposed TrustGate Composite:** Error-detection ROC-AUC = **0.4697** [95% CI: 0.4172, 0.5221], $\text{pAURC}_{[0.5, 1.0]} = \mathbf{0.3088}$.
- **Mathematical Equivalence Theorem:** Under binary classification, Margin Risk ($1 - 2|p - 0.5|$), MSP Risk ($0.5 - |p - 0.5|$), and Predictive Entropy are strictly monotonic transformations of $|p - 0.5|$, yielding identical error-detection AUC (**0.5829**).
- **The Consensus Inversion Paradox:** Due to positive logit collapse on ADNI, models exhibit their highest agreement on false positive cases. High multi-model consensus actively correlates with error, causing Component C to degrade composite performance.
- **Component B Saturation:** 100% of ADNI samples exceed the source validation 95th percentile ($q_{95} = 29.38$), saturating clipped feature distance to 1.0000 (zero variance $\implies$ AUC 0.5000; raw Mahalanobis distance yields AUC = 0.4808).
- **Status:** Officially declared an **implemented research prototype**. TrustGate demonstrates that simple Classification Margin remains the most reliable selective prediction signal under cross-scanner domain shift. Full negative result details are documented in [`docs/research/v3/trustgate_research_findings.md`](docs/research/v3/trustgate_research_findings.md).

---

## 📥 Research Paper & Submission Artifacts

All deliverables for academic submission and examination are located in [`docs/submission/final/`](docs/submission/final/):

| Artifact | File Path | Format | Description |
| :--- | :--- | :---: | :--- |
| **Complete Research Paper** | `docs/submission/final/SHIFT_MRI_Complete_Research_Paper.pdf` | PDF | Comprehensive 30-section academic research paper with full methodology, figures, tables, code snippets, and citations |
| **Editable Manuscript** | `docs/submission/final/SHIFT_MRI_Complete_Research_Paper.docx` | DOCX | Editable Microsoft Word manuscript formatted to academic submission standards |
| **Slide Deck** | `docs/submission/final/SHIFT_MRI_Final_Presentation.pptx` | PPTX | 16-slide academic presentation with speaker notes |
| **Submission Package** | `docs/submission/final/SHIFT_MRI_Submission_Package.zip` | ZIP | Complete deliverable archive including paper, slides, result matrices, and documentation |

---

## 🖥️ Launching the SHIFT-MRI Research Studio Application

The application provides a neuroscience-themed research workstation featuring an **Interactive Orthogonal MRI Slice Viewer**, **Live Per-Scan Inference Explorer**, **Model Library & Architecture Lab**, **TrustGate Studio**, **Research Analytics Center**, and **Guided Viva Defense Tour**.

### One-Command Windows Launcher:
```powershell
.\launch_shift_mri.bat
```
or via PowerShell:
```powershell
.\launch_shift_mri.ps1
```
The application will launch locally at: **`http://localhost:8000`**

---

## 🧪 Automated Testing & Verification

Execute the complete automated test suite (90 passed, 4 skipped across the repository):
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/ -v
```
All tests verify schema validity, absence of test-label leakage in threshold derivation, matched 500-subject demographic joins, mathematical equivalence of confidence measures, saturation detection, and artifact integrity.

---

## ⚖️ Data Governance & Ethical Statements

1. **Research Use Only:** This software and benchmark findings are strictly for computational and neuroimaging AI research. They are **not certified medical devices** and must not be used for clinical diagnosis.
2. **Data Privacy & Governance:** Raw MRI datasets are governed by institutional Data Use Agreements:
   - **OASIS-1 & OASIS-2:** Washington University ADRC; OASIS-2 is strictly classified as an *exploratory secondary cohort with unverified participant overlap*.
   - **ADNI:** Alzheimer's Disease Neuroimaging Initiative Data Use Terms. No protected health information (PHI) or identifiable facial voxel geometries are stored or distributed in public commits.
3. **Open Source License:** All benchmark code is released under the **Apache 2.0 License**.
