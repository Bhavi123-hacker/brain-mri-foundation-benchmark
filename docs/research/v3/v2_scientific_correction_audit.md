# Scientific Correction and Integrity Audit: Version 2 Benchmark Findings

**Working Group:** Antigravity Neuroimaging AI Research & Verification Team  
**Audit Date:** October 21, 2026  
**Document Target:** `docs/research/v3/v2_scientific_correction_audit.md`  
**Prior Commit Audited:** `ce31ef65add5f33e12e5a40669dd72968271ff1a`  
**Repository Branch:** `research/v3-shift-mri-reliability` (transferred from `research/v2-foundation-generalization-upgrade`)  

---

## 1. Executive Summary of Audit

An independent peer review of the Version 2 (V2) codebase and publication artifacts identified critical scientific inconsistencies, nomenclature imprecisions, and reporting errors. Under the strict governance protocols of the SHIFT-MRI initiative, this document provides an item-by-item accounting of each finding, root cause investigation, empirical impact, verified truth, and corrective remediation.

---

## 2. Item-by-Item Scientific Corrections (Items A through J)

### Item A — OASIS-1 vs. OASIS-2 Cohort Overlap & Independence
* **Problem Identified:** A direct contradiction existed between `docs/data/participant_overlap_audit.md` (which cautioned that OASIS-1 and OASIS-2 shared 20–40% potential participant overlap due to common Washington University ADRC recruitment) and `docs/data/oasis2_expansion_audit.md` (which claimed 0.0% overlap based solely on string prefix matching `sub-OASIS2xxxx` vs. `OAS1_xxxx`).
* **Source Files Affected:** `docs/data/oasis2_expansion_audit.md`, `docs/data/participant_overlap_audit.md`, `results/v2/v2_final_research_report.md`, `README.md`.
* **Impact on Results:** Incorrectly labeling OASIS-2 as a "confirmed independent external validation cohort" created false confidence in cross-cohort generalization, obscuring potential training-to-test participant contamination.
* **Investigation:** Marcus et al. (2010) explicitly documents that the OASIS-2 longitudinal study recruited non-demented and demented older adults from the ongoing Washington University Alzheimer's Disease Research Center (WashU ADRC) patient registry—the exact same clinical center and time window that populated the cross-sectional OASIS-1 repository. The OASIS project assigned new session and subject IDs (`OAS2_xxxx`) without publishing an official one-to-one cross-mapping table in open releases. Different identifier namespaces do NOT establish biological participant independence.
* **Verified Conclusion:** Biological participant independence between OASIS-1 and OASIS-2 **cannot be certified as 0.0%** without access to restricted identifiable clinical linkage files.
* **Correction Implemented:**
  1. Withdrew the claim of "confirmed independent external validation."
  2. Reclassified OASIS-2 strictly as a **"Secondary Exploratory / Unverified-Overlap Multi-Session Cohort"**.
  3. Re-evaluated all cross-cohort findings to ensure no confirmatory conclusions rely on OASIS-2 as an independent external test.
* **Tests Added:** `test_oasis2_unverified_overlap_disclosure()` in `tests/test_v3_shift_mri_integrity.py`.
* **Remaining Limitations:** Without institutional WashU ADRC master patient registry access, exact duplicate identification is impossible; OASIS-2 must remain an exploratory reference only.

---

### Item B — Preprocessing Pipeline Ablation Validity
* **Problem Identified:** Experiment E02 in `scripts/run_v2_experiment_campaign.py` claimed to evaluate a preprocessing ablation between "Pipeline A (Baseline Zero-Mean)" and "Pipeline B (Harmonized Percentile Z-Score)," but inspection revealed that both branches were trained on the identical pre-cached tensor arrays in `OASIS_TENSORS_DIR` without applying distinct transformations. Furthermore, reports claimed N4 bias correction, MNI atlas registration, and SynthStrip brain extraction.
* **Source Files Affected:** `scripts/run_v2_experiment_campaign.py`, `results/v2/v2_final_research_report.md`, `docs/data/dataset_expansion_execution_report.md`.
* **Impact on Results:** Fabricated claims of preprocessing superiority (+0.0017 on ADNI, +0.0143 on OASIS-2) were reported based on stochastic model weight initialization variance rather than genuine preprocessing differences.
* **Investigation:** Review of `OASIS_TENSORS_DIR` showed all `.npy` files were generated during V1 by a uniform pipeline: trilinear spatial resampling to $(1, 128, 128, 128)$ isotropic matrices followed by nonzero Z-score intensity normalization. No N4 bias field correction, non-linear MNI atlas registration, or SynthStrip brain extraction was ever executed in code or logged on disk.
* **Verified Conclusion:** Stored tensors reflect basic spatial resizing and intensity Z-scoring only. The reported Pipeline A vs. Pipeline B ablation was scientifically invalid.
* **Correction Implemented:**
  1. Withdrew all claims of N4 bias field correction, MNI non-linear registration, and spatial harmonization.
  2. Withdrew the unverified claims that Pipeline B enhanced external generalization.
  3. Replaced E02 with a **strictly controlled intensity outlier percentile clipping experiment**: comparing raw unclipped Z-score arrays versus dynamically applied 0.5%–99.5% percentile clipped arrays on 3D ResNet-10.
* **Tests Added:** `test_preprocessing_factual_disclosure()` in `tests/test_v3_shift_mri_integrity.py`.
* **Remaining Limitations:** True N4 bias field correction and MNI non-linear registration require heavy CPU/GPU processing pipelines (SimpleITK/ANTs/FreeSurfer) on raw DICOM/NIfTI volumes, which cannot be retroactively applied to pre-cached tensors before deadline.

---

### Item C — Model & Checkpoint Provenance Verification
* **Problem Identified:** Need to certify that all ten reported model architectures, checkpoints, and weights correspond to authentic implementations, with specific verification of pretrained key matching for MONAI Swin UNETR SSL.
* **Source Files Affected:** `models/v2_models.py`, `models/foundation_swin/adapter.py`, `results/v2/checkpoints/`.
* **Impact on Results:** Risk of evaluating randomly initialized or partially loaded weights under the guise of foundation models.
* **Investigation:** 
  - `weights/model_swinvit.pt` (392.11 MB, SHA-256 `1cf19eca3ed0472daf33dedd2f095b118db16edac0fc53d9e878f328ca68c988`) was verified against the official Project-MONAI extra-test-data repository.
  - Inspection of `results/swin_unetr_checkpoint_verification_report.json` confirmed 0 critical missing keys during state dict loading into `monai.networks.nets.SwinUNETR`.
  - Checkpoints for V2 models in `results/v2/checkpoints/` (`3d_resnet_10_seed42_best.pt`, `3d_densenet_121_seed42_best.pt`, `2.5d_multi_view_resnet_18_seed42_best.pt`, `dual_branch_hybrid_seed42_best.pt`) were verified to load into their respective PyTorch definitions with exact key matching.
* **Verified Conclusion:** All model checkpoints correspond to authentic PyTorch modules and valid pretrained weights.
* **Correction Implemented:** Added automated cryptographic hash and parameter loading verification tests.
* **Tests Added:** `test_all_model_checkpoints_loadable()` in `tests/test_v3_shift_mri_integrity.py`.

---

### Item D — Model Parameter Count Discrepancies
* **Problem Identified:** The V2 master comparison table hard-coded parameter counts that directly contradicted the actual instantiated model objects (e.g., 3D ResNet-10 was listed as 5,438,881 params, while the actual model object had 14,619,585 params; Dual-Branch Hybrid was listed as 5,440,417 params vs. 14,525,153 params).
* **Source Files Affected:** `scripts/run_v2_experiment_campaign.py`, `results/v2/tables/v2_master_model_comparison_table.csv`, `results/v2/v2_final_research_report.md`.
* **Impact on Results:** Severely distorted model capacity comparisons, falsely presenting 3D ResNet-10 as a "lightweight 5.4M model" when it actually possessed 14.6M parameters.
* **Investigation:** In `scripts/run_v2_experiment_campaign.py` lines 600–605, parameter numbers were hard-coded strings from an outdated MONAI 2D architecture configuration rather than queried from `sum(p.numel() for p in model.parameters())`.
* **Verified Conclusion:** True model parameter counts:
  - 3D ResNet-10: **14,619,585**
  - 3D DenseNet-121: **11,767,937**
  - 2.5D Multi-View ResNet-18: **11,570,241**
  - Dual-Branch Hybrid: **14,525,153**
  - 3D ResNet-18: **33,160,513**
  - 3D ResNet-34: **63,470,145**
  - 2D Transfer ResNet-18: **11,177,025**
  - Classical Radiomics: **25**
  - MONAI Swin UNETR SSL: **8,062,771**
  - FOMO25 AMAES ResEnc-B: **90,296,897**
* **Correction Implemented:** Replaced hard-coded parameter values with dynamic runtime instantiation calls `model.get_parameter_count()["total_parameters"]`.
* **Tests Added:** `test_parameter_counts_exact_match()` in `tests/test_v3_shift_mri_integrity.py`.

---

### Item E — Training Time vs. Checkpoint Evaluation Runtime
* **Problem Identified:** Reported training durations in V2 campaign outputs listed 0.3s, 0.6s, 1.2s, 1.6s for full deep learning models, confusing checkpoint evaluation time with actual model training duration.
* **Source Files Affected:** `scripts/run_v2_experiment_campaign.py`, `results/v2/tables/v2_benchmark_campaign_results.json`.
* **Impact on Results:** Gave the false impression that deep 3D CNNs trained in milliseconds.
* **Investigation:** When `scripts/run_v2_experiment_campaign.py` detected existing checkpoints in `results/v2/checkpoints/`, it loaded the weights and ran validation passes in ~1s, recording `elapsed = time.time() - t_start` as `train_time_sec`.
* **Verified Conclusion:** Actual scratch training durations (from training logs):
  - 3D ResNet-10: **48.2 seconds** (15 epochs)
  - 3D DenseNet-121: **64.7 seconds** (15 epochs)
  - 2.5D Multi-View ResNet-18: **24.8 seconds** (15 epochs)
  - Dual-Branch Hybrid: **51.5 seconds** (15 epochs)
  - Checkpoint loading and forward evaluation time: **0.3 – 1.6 seconds**.
* **Correction Implemented:** Separated `actual_training_time_sec` from `checkpoint_eval_time_sec` and flagged checkpoint loading explicitly in all JSON ledgers and tables.

---

### Item F — Statistical Attribution of p-Values
* **Problem Identified:** The text in `v2_final_research_report.md` attributed a raw p-value of $p = 0.641$ to "FOMO25 AMAES vs. Classical Radiomics," when $p = 0.641$ actually belonged to "Dual-Branch Hybrid vs. Classical Radiomics."
* **Source Files Affected:** `results/v2/v2_final_research_report.md`, `results/v2/v2_executive_handoff_memo.md`.
* **Impact on Results:** Mislabeled statistical attribution in publication text.
* **Investigation:** Inspection of `results/v2/tables/v2_paired_bootstrap_statistical_audit.csv` confirmed:
  - Row 6: `model_a = Dual-Branch Hybrid`, `model_b = Classical Radiomics`, `oasis_raw_p = 0.641`, `adni_raw_p = 0.345`.
  - Historical V1 comparison for FOMO25 vs. Classical Radiomics: `oasis_raw_p = 0.963`, `adni_raw_p = 0.807`.
* **Verified Conclusion:** The $p = 0.641$ statistic belongs exclusively to Dual-Branch Hybrid vs. Radiomics.
* **Correction Implemented:** Corrected all textual attributions in the report, memo, and presentation slides.

---

### Item G — External ADNI Model Ranking Correction
* **Problem Identified:** The V2 report text claimed that "Dual-Branch Hybrid achieved the highest out-of-domain transfer on ADNI ($\text{AUC} = 0.5554$)." In reality, the master comparison table reported **MONAI Swin UNETR SSL at 0.5834** and **2D Transfer ResNet-18 at 0.5778**.
* **Source Files Affected:** `results/v2/v2_final_research_report.md`, `results/v2/v2_executive_handoff_memo.md`.
* **Impact on Results:** Directly contradicted the numerical evidence in the master leaderboard table.
* **Investigation:** Examination of `v2_master_model_comparison_table.csv` confirms:
  1. MONAI Swin UNETR SSL: **0.5834** [0.5331, 0.6337]
  2. 2D Transfer ResNet-18: **0.5778** [0.5280, 0.6276]
  3. Dual-Branch Hybrid: **0.5554** [0.5044, 0.6066]
  4. 3D ResNet-10: **0.5444** [0.4909, 0.5953]
  5. Classical Radiomics: **0.5149** [0.4638, 0.5660]
  Dual-Branch Hybrid was the highest performing model *among the newly trained V2 scratch/hybrid models*, but did NOT surpass the foundation transformer or 2D transfer model on ADNI.
* **Verified Conclusion:** MONAI Swin UNETR SSL is the top-ranked model on ADNI.
* **Correction Implemented:** Corrected all ranking descriptions in research reports, executive memos, and README.

---

### Item H — Anatomical Biomarker Nomenclature Correction
* **Problem Identified:** The features extracted by `AnatomicalROIExtractor` were referred to as "clinically validated hippocampal and ventricular volumetric measurements."
* **Source Files Affected:** `preprocessing/enhanced_pipeline.py`, `results/v2/v2_final_research_report.md`.
* **Impact on Results:** Claimed clinical segmentation validity that was neither executed nor scientifically justified.
* **Investigation:** Code in `preprocessing/enhanced_pipeline.py` lines 100–140 uses static bounding box coordinates:
  - Ventricle proxy: `v[50:85, 45:85, 48:80]`
  - Left hippocampal proxy: `v[40:70, 48:78, 36:58]`
  - Right hippocampal proxy: `v[40:70, 48:78, 70:92]`
  These are fixed sub-volumes in $128^3$ space with intensity thresholding, NOT segmented anatomical structures from FreeSurfer, FastSurfer, or manual radiologist tracings.
* **Verified Conclusion:** These features are heuristic regional occupancy proxies.
* **Correction Implemented:** Renamed all references to **"heuristic bounding-box region-of-interest (ROI) intensity and volume proxies"**. Clarified that no segmented anatomical volumetric validation was performed.

---

### Item I — Clinical Label Semantics & Diagnostic Harmonization
* **Problem Identified:** Treating binary positive labels across OASIS-1, ADNI, and OASIS-2 as interchangeable diagnostic targets.
* **Source Files Affected:** `docs/data/clinical_label_harmonization.md`, `results/v2/v2_final_research_report.md`.
* **Impact on Results:** Obscured clinical label heterogeneity as a primary driver of observed domain transfer drops.
* **Investigation:** 
  - OASIS-1 positive class: Clinical Dementia Rating $\text{CDR} \ge 0.5$ (contains very mild impairment $\text{CDR}=0.5$, mild $\text{CDR}=1.0$, moderate $\text{CDR}=2.0$).
  - ADNI positive class: Clinical consensus diagnosis of Alzheimer's disease ($\text{AD}$ vs. $\text{CN}$), excluding MCI from the binary benchmark.
  - OASIS-2 positive class: Mapped from third-party binary labels derived from longitudinal visits.
* **Verified Conclusion:** Label definitions differ fundamentally across cohorts; what appears as "scanner domain shift" is heavily confounded by clinical diagnostic criteria drift.
* **Correction Implemented:** Explicitly documented label semantics and added Novelty Experiment N05 (Clinical Label Sensitivity) to isolate CDR severity strata.

---

### Item J — Historical Holdout Exposure Disclosure
* **Problem Identified:** The OASIS-1 locked test split ($N=30$) was evaluated during V1 iterations.
* **Source Files Affected:** `results/report/`, `results/v2/`.
* **Impact on Results:** Risk of presenting the OASIS-1 test results as an untouched prospective confirmatory trial.
* **Investigation:** V1 logs demonstrate that test split predictions were previously evaluated.
* **Verified Conclusion:** OASIS-1 test split is a historical holdout, not an unexamined prospective trial.
* **Correction Implemented:** Maintained prominent disclosure in all V2 and V3 documentation that the OASIS-1 locked test split is a historically examined benchmark split, and that zero hyperparameters in V2 or V3 were tuned against it.

---

## 3. Certification of Scientific Audit

This audit was conducted with zero tolerance for scientific fabrication, confirmation bias, or artifact inflation. All corrections detailed above have been applied to the codebase, test suites, and documentation.
