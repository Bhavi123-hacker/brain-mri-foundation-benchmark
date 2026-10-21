# SHIFT-MRI: Final Research and Submission Verification Checklist

**Project Title:** Fair Benchmarking and Reliability Analysis of Brain-MRI Foundation Models (SHIFT-MRI)  
**Submission Deadline:** October 22, 2026  
**Repository:** `https://github.com/Bhavi123-hacker/brain-mri-foundation-benchmark`  
**Research Branch:** `research/v3-shift-mri-reliability`  
**Operating Environment:** Windows / Python 3.12 / PyTorch 2.11 / NVIDIA RTX PRO 2000 Blackwell 16GB  

---

## 1. Governance and Scientific Integrity Checklist

- [x] **Priority 1: Dataset Independence and Governance Verified**
  - Confirmed Washington University ADRC shared origin between OASIS-1 and OASIS-2.
  - Withdrawn false claims of OASIS-2 being a "confirmed disjoint external hospital cohort".
  - Formally disclosed OASIS-2 as an *Exploratory Secondary Cohort with Unverified Participant Overlap*.
  - Disclosed Hugging Face Radiata ODC-By collection terms vs underlying scan DUAs in `dataset_license_and_access_matrix.csv`.
  - Documented in `docs/research/v3/new_dataset_discovery_report.md`.

- [x] **Priority 2: Preprocessing Authenticity Verified**
  - Verified actual physical transformations in `processed/tensors`: trilinear isotropic resampling to $128^3$ and nonzero foreground Z-score normalization.
  - Explicitly withdrawn historical false claims of N4 bias field correction, SynthStrip, and MNI non-linear atlas registration.
  - Executed true controlled intensity clipping experiment (unclipped Z-score vs 0.5%–99.5% percentile clipped Z-score).
  - Documented in `docs/research/v3/v2_scientific_correction_audit.md`.

- [x] **Priority 3: Model and Biomarker Authenticity Verified**
  - Renamed "biomarkers" to "heuristic bounding-box anatomical proxies" (total brain voxels, ventricular CSF voxels, medial temporal tissue voxels, ventricle-to-brain ratio, hippocampal ratio, intensity skewness).
  - Verified no clinical FreeSurfer or FastSurfer segmentations were executed.

- [x] **Priority 4: Benchmark Results Accuracy Verified**
  - Re-derived exact model parameter counts directly from instantiated PyTorch objects:
    - 3D ResNet-18: 33,160,513
    - 3D ResNet-34: 63,489,217
    - 3D ResNet-10: 14,619,585
    - 3D DenseNet-121: 11,767,937
    - 2.5D Multi-View ResNet-18: 11,570,241
    - Dual-Branch Hybrid: 14,525,153
    - MONAI Swin UNETR SSL: 8,062,771
    - FOMO25 AMAES ResEnc-B: 90,296,897
    - Classical Radiomics: 25
  - Confirmed definitive external ADNI rankings: MONAI Swin UNETR SSL (**0.5834**) > 2D Transfer ResNet-18 (**0.5778**) > Dual-Branch Hybrid (**0.5554**).
  - Corrected historical statistical misattribution: $p = 0.641$ confirmed to belong strictly to Dual-Branch Hybrid vs Classical Radiomics on OASIS-1 locked test, not FOMO25 vs Radiomics.
  - Separated actual training duration from checkpoint loading and inference time.

- [x] **Priority 5: Working Project, Presentation, and Viva Voce Guide**
  - Complete scientific manuscript: `results/v3/SHIFT_MRI_FINAL_RESEARCH_REPORT.md`.
  - Executive handoff memo: `results/v3/SHIFT_MRI_EXECUTIVE_HANDOFF.md`.
  - Academic 10–12 slide deck with speaker notes: `docs/submission/SHIFT_MRI_PRESENTATION.md`.
  - 20-question viva voce technical defense guide: `docs/submission/SHIFT_MRI_NOVELTY_DEFENSE.md`.

---

## 2. SHIFT-MRI Seven Methodological Novelties (N01–N07) Checklist

- [x] **N01: Cross-Cohort Reliability Matrix**
  - Implemented in `scripts/run_v3_shift_mri_campaign.py`.
  - Output generated: `results/v3/reliability/cross_cohort_reliability_matrix.csv`.
  - Metrics include: ROC-AUC [95% CI], PR-AUC, BalAcc at $t=0.50$ and $t^*$, Brier score, ECE across 10 bins, calibration slope/intercept, and generalization degradation gaps.

- [x] **N02: Prediction Collapse and Failure Mode Analysis**
  - Output generated: `results/v3/failure_analysis/prediction_collapse_table.csv`.
  - Output generated: `results/v3/failure_analysis/failure_case_audit.csv`.
  - Quantified 99%–100% positive rate collapse on external ADNI at $t=0.50$.

- [x] **N03: Demographic Shortcut Baselines**
  - Implemented in `evaluation/v3/shift_mri.py`.
  - Evaluated on OASIS-1 Test, ADNI External, and OASIS-2 External.
  - Output generated: `results/v3/demographic_baselines/demographic_baselines_results.csv`.
  - Proved Age + Sex Logistic Regression (AUC = 0.5671) outperforms 7/10 deep learning models on external ADNI.

- [x] **N04: Confidence-Aware Selective Prediction / Abstention**
  - Implemented risk-coverage curve integration (AURC) based on boundary distance $|p - 0.50|$.
  - Output generated: `results/v3/abstention/abstention_metrics.csv`.
  - Plot generated: `results/v3/figures/risk_coverage_curves.png`.

- [x] **N05: Clinical Label Sensitivity Stratification**
  - Stratified detection by CDR 0.0 vs CDR 0.5 vs CDR 1.0+.
  - Documented in `docs/research/v3/clinical_label_sensitivity_audit.md`.
  - Demonstrated Dual-Branch Hybrid superiority on CDR 0.5 (86.7% sensitivity).

- [x] **N06: Scanner & Image Perturbation Robustness**
  - Perturbations tested: Additive Gaussian Noise ($\sigma \in [0.0, 0.10]$) and Global Intensity Scaling ($0.85\times - 1.15\times$).
  - Output generated: `results/v3/robustness/perturbation_robustness_results.csv`.
  - Plot generated: `results/v3/figures/perturbation_robustness_curves.png`.

- [x] **N07: Out-of-Distribution (OOD) Detection**
  - Evaluated using FOMO25 320-dim feature L2 norms and Swin UNETR predictive entropy.
  - Output generated: `results/v3/ood_detection/ood_detection_evaluation.csv`.
  - Plot generated: `results/v3/figures/ood_detection_roc.png` (OOD AUC = 0.9933 and 1.0000).

---

## 3. SHIFT-MRI TrustGate Methodological Prototype Checklist

- [x] **TrustGate Literature Review & Conceptual Mapping:**
  - `docs/research/v3/trustgate_prior_art_comparison.md` documenting failure detection, label-free FM selection, selective prediction, and conformal prediction limitations.
  - Transparent boundary: composite formulation of established signals rather than unproven fundamental novelty.
- [x] **TrustGate Method Definition & Mathematical Formulation:**
  - `docs/research/v3/trustgate_method_definition.md` defining components A (Margin), B (Reg. Feature Distance), C (Disagreement), D (QC Anomaly).
  - Source-locked parameter calibration on OASIS-1 with zero external label leakage.
- [x] **TrustGate Empirical Execution and Comparative Baselines:**
  - `scripts/run_v3_trustgate_campaign.py` executed successfully.
  - `results/v3/trustgate/trustgate_benchmark_results.csv` comparing MSP, Entropy, Margin, Feature Distance, Disagreement, Aggregation, and Composite.
  - `results/v3/trustgate/trustgate_risk_coverage_table.csv` and `results/v3/figures/trustgate_risk_coverage.png`.
- [x] **TrustGate Component Ablations & Scanner Subgroups:**
  - `results/v3/trustgate_ablation/trustgate_ablation_results.csv` isolating component impact.
  - `results/v3/trustgate/trustgate_scanner_subgroup_analysis.csv` across Siemens vs Philips and 1.5T vs 3.0T.
- [x] **Empirical Findings & Negative Result Disclosure:**
  - `docs/research/v3/trustgate_research_findings.md` documenting that Classification Margin alone (AUC = 0.5829, AURC = 0.2871) outperforms Full TrustGate Composite (AUC = 0.4697), caused by consensus inversion of model disagreement under positive logit collapse.
  - Official maturity level declared: *Implemented research prototype*.

---

## 4. Code Quality, Test Suite, and Security Verification

- [x] **Automated PyTest Suite Status:**
  - `tests/test_adni_pipeline.py`: Patched and passing.
  - `tests/test_frontend_application.py`: Patched and passing.
  - Total tests passing: **83 passed, 4 skipped, 0 failed across all test files**.
- [x] **Git and Data Security:**
  - `.gitignore` verified to exclude large raw MRI NIfTI scans, private participant identifiers, credentials, and bulky weights.
  - All source code, small CSV tables, markdown reports, and lightweight figures tracked in Git.
  - Research branch `research/v3-shift-mri-reliability` ready for final commit and push.
