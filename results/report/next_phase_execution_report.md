# Next-Phase Execution & Benchmark Validation Report
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Phase**: Phase A (MIRIAD Integration Audit), Phase B (FOMO Checkpoint Validation), Phase C (External Protocol Specification)  
**Date**: October 12, 2026  
**Auditor**: Senior Medical Imaging Machine Learning Lead  
**Scope**: In-depth verification of MIRIAD access terms, download protocols, official FOMO25 AMAES ResEnc-B checkpoint loading, GPU smoke testing, and experimental protocol freezing without altering existing checkpoints or the locked OASIS test set.

---

## 1. Executive Summary of Outcomes

| Phase / Objective | Target Asset / Cohort | Verified Status | Key Findings & Evidence | Next Required Action |
| :--- | :--- | :---: | :--- | :--- |
| **Phase A — MIRIAD Access & Integration** | MIRIAD Dataset ($N=69$, UCL Queen Square) | **STOPPED PENDING USER CREDENTIALING** | Access is governed by the **MIRIAD Data Use Agreement** (UCL DRC XNAT). Automated anonymous scraping is prohibited; user registration and administrative approval required. Resolved license discrepancy in `dataset_expansion_audit.md`. | User must register at `http://miriad.drc.ion.ucl.ac.uk/` and place baseline scans in `D:\brain_mri_data\MIRIAD\raw\`. |
| **Phase B — FOMO Foundation Checkpoint** | `FOMO-MRI/AMAES_resenc_b_fomo260k` (`resenc_unet_b.ckpt`) | **VERIFIED & OPERATIONAL** | Downloaded 1.11 GB checkpoint (`115d50...`). Strict match across all 448 encoder keys. Forward pass executed on RTX PRO 2000 Blackwell GPU using real OASIS-1 tensor: extracts 320-dim bottleneck embedding, peak VRAM 1.63 GB. | Ready for linear probing and zero-shot feature evaluation. |
| **Phase C — External Cohort Protocol** | MIRIAD External Evaluation Protocol ($N=69$) | **PREDEFINED & FROZEN** | Complete protocol specified in `results/miriad_external_validation/protocol_specification.json`. Multi-paradigm comparison (2D transfer, 3D supervised, radiomics, Swin UNETR SSL, AMAES ResEnc-B) with bootstrap 95% CIs. | Automated evaluation script `scripts/run_miriad_external_evaluation.py` ready to execute upon scan download. |

---

## 2. Phase A: MIRIAD Access Instructions & Terms Resolution

### A. Resolution of License Discrepancy in Prior Audit
In the initial dataset expansion review, MIRIAD was tentatively described as open access under CC-BY 4.0 / Open Government License with direct wget availability. **Our verification audit revealed this was inaccurate:**
1. **Official Repository**: Hosted exclusively on the UCL Dementia Research Centre (DRC) XNAT database (`http://miriad.drc.ion.ucl.ac.uk/` redirecting to `https://www.ucl.ac.uk/drc/research/methods/miriad-scan-database`).
2. **Governing Legal Terms**: **MIRIAD Data Use Agreement**.
3. **Core Agreement Restrictions**:
   - **No Redistribution**: Third-party redistribution or hosting of raw MRI volumes is strictly forbidden without prior written authorization from UCL.
   - **Privacy Protection**: Users explicitly covenant not to attempt re-identification or contact with study participants.
   - **Mandatory Citation**: All publications must cite the primary data release paper:  
     *Malone IB, Cash D, Ridgway GR, Macmanus DG, Ourselin S, Fox NC, Schott JM. MIRIAD—Public release of a multiple time point Alzheimer's MR imaging dataset. Neuroimage. 2012 Dec 28;70C:33-36. doi:10.1016/j.neuroimage.2012.12.044.*
   - **Mandatory Acknowledgment**: Publications must include the formal statement:  
     *"Data used in the preparation of this article were obtained from the MIRIAD database (http://miriad.drc.ion.ucl.ac.uk). The MIRIAD investigators did not participate in analysis or writing of this report."*
   - **Manuscript Notification**: Authors must share accepted manuscripts with `drc-miriad@ucl.ac.uk`.
4. **Access Mechanism**: Access requires creating a named personal user account on the MIRIAD XNAT portal, which is manually reviewed and approved by UCL administrators. There is no anonymous public mirror or authorized open wget endpoint.
5. **Report Correction**: `results/report/dataset_expansion_audit.md` has been updated to reflect these verified terms and mark MIRIAD's status as `ACCESS_GATED`.

### B. Scientific Integrity Compliance: Stop Condition
Because the MIRIAD Data Use Agreement forbids automated public redistribution and mandates credentialed account registration, **we immediately stopped the automated download process**. We did not attempt unauthorized scraping from unofficial third-party mirrors, adhering strictly to research compliance.

A complete user guide has been saved to [`results/miriad_external_validation/download_and_access_guide.md`](file:///D:/ml%20project/results/miriad_external_validation/download_and_access_guide.md) detailing exact registration steps for the user.

### C. Participant Isolation & Provenance Audit
- **MIRIAD Demographics**: 69 participants (46 confirmed Alzheimer's Disease, 23 Cognitively Normal elderly controls; 708 total longitudinal scans).
- **Scanner Provenance**: Single 1.5 Tesla General Electric (GE) Signa Horizon Echospeed scanner located at the Dementia Research Centre, Queen Square, London, UK (acquired 2000–2002).
- **Overlap Analysis**:
  - OASIS-1: Recruited exclusively at Washington University Alzheimer's Disease Research Center (ADRC), St. Louis, Missouri, USA on a Siemens Vision scanner.
  - ADNI: Recruited across North American academic clinical sites.
  - MIRIAD: Recruited at Queen Square, London, UK on a GE Signa scanner.
  - **Documented Overlap**: **0.0% participant overlap**.
  - **Limitations of Check**: Because participant data is pseudonymized across all three cohorts (subject IDs: `miriad_xxx` vs `OAS1_xxxx` vs `xxx_S_xxxx`) without universal biometric identifiers, mathematical verification relies on independent clinical recruitment records, geographic centers, and distinct study protocols.

---

## 3. Phase B: Official FOMO25 AMAES ResEnc-B Foundation Checkpoint Validation

### A. Checkpoint Provenance, Hash & Licensing
- **Repository**: `FOMO-MRI/AMAES_resenc_b_fomo260k` on HuggingFace Hub.
- **Downloaded Checkpoint**: `weights/resenc_unet_b.ckpt` (1,167,183,049 bytes / 1.11 GB).
- **File Integrity**:
  - SHA256: `115d509241ad3a421a851dda3a1e5d98ae8377973ea9b29bbb2c2a181c277749`
  - ETag: `c6d5a93c8cbb1dc9436fbb1b981e87dba68e9d409594926f2d9e22f4839d3b29`
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (**CC-BY-NC-SA 4.0**).
- **Pretraining Objective**: Augmented Masked Autoencoder Pretraining on Public Brain MRI Data for 3D-Native Segmentation (**AMAES**, Munk et al., arXiv:2408.00640).
- **Pretraining Pool**: FOMO260K (260,000 multi-sequence 3D brain MRI volumes).

### B. Network Architecture & Dependency Verification
- **Framework Packages**: Integrated via `asparagus` (v0.4.6) and `gardening_tools` (v0.3.5).
- **Environment Dependencies**: `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `monai 1.6.1`, `nibabel 5.4.2`. Full CUDA hardware acceleration on the NVIDIA RTX PRO 2000 Blackwell GPU verified.
- **Model Parameters**:
  - Total UNet Parameters: 102,348,193 (~102.35M).
  - Encoder Backbone Parameters: **90,296,576 (~90.30M)**.
  - Encoder Stages: 6 hierarchical residual stages:
    - Stage 0: 32 channels, 1 block
    - Stage 1: 64 channels, 3 blocks
    - Stage 2: 128 channels, 4 blocks
    - Stage 3: 256 channels, 6 blocks
    - Stage 4: 320 channels, 6 blocks
    - Stage 5: 320 channels, 6 blocks
- **Weight Loading Verification**:
  - All **448 encoder keys** from the official PyTorch Lightning state dictionary loaded into the `ResidualUNetEncoder` backbone with **strict=True** (0 missing keys, 0 unexpected keys).

### C. Feature Extraction & GPU Inference Smoke Test
We tested feature extraction using the official backbone adapter [`models/foundation_fomo25/amaes_adapter.py`](file:///D:/ml%20project/models/foundation_fomo25/amaes_adapter.py) on a real preprocessed OASIS-1 brain tensor (`OAS1_0001_MR1.pt`):

```text
Input Tensor: [1, 1, 128, 128, 128] float32 on CUDA (RTX PRO 2000 Blackwell)
  Stage 0 Feature Map: [1, 32, 128, 128, 128]  (Finite: True)
  Stage 1 Feature Map: [1, 64, 64, 64, 64]      (Finite: True)
  Stage 2 Feature Map: [1, 128, 32, 32, 32]    (Finite: True)
  Stage 3 Feature Map: [1, 256, 16, 16, 16]    (Finite: True)
  Stage 4 Feature Map: [1, 320, 8, 8, 8]       (Finite: True)
  Stage 5 Feature Map: [1, 320, 4, 4, 4]       (Finite: True)
Bottleneck Representation: Stage 5 pooled via AdaptiveAvgPool3d to [1, 320]
  Feature Vector Norm: 43.1902
  Peak GPU VRAM Allocated: 1632.59 MB (1.63 GB)
```

### D. Critical Scientific Note: Pretext Alignment Caution
The AMAES pretext objective is masked volume reconstruction optimized for downstream **dense segmentation** (e.g. brain extraction, stroke lesion segmentation, WMH). Dense reconstruction representations preserve localized voxel intensities but do not automatically guarantee linear separability for whole-brain clinical dementia classification. Linear probing must be tested empirically against supervised baselines rather than assuming foundation pretraining superiority.

---

## 4. Phase C: Predefined Participant-Level Protocol for MIRIAD

To ensure zero risk of p-hacking or test-set snooping on the small cohort ($N=69$), the evaluation protocol was codified in [`results/miriad_external_validation/protocol_specification.json`](file:///D:/ml%20project/results/miriad_external_validation/protocol_specification.json) prior to any evaluation:

1. **Cohort Specification**: Exactly 69 participants (46 AD, 23 Controls). Primary analysis uses one baseline volumetric scan per subject.
2. **Partition Quarantine**: Longitudinal follow-up scans from any participant are strictly quarantined to the same partition.
3. **Multi-Paradigm Comparative Models**:
   - Supervised 3D ResNet-18 (`checkpoints/oasis1_full_resnet18_best.pt`)
   - Supervised 3D ResNet-34 (`checkpoints/oasis1_resnet34_best.pt`)
   - 2D Transfer ResNet-18 Slice Aggregated (`checkpoints/resnet2d_partial_finetune_best.pt`)
   - Classical Volume/Intensity Radiomics Linear Model (`checkpoints/classical_radiomics_best.pkl`)
   - MONAI Swin UNETR SSL Linear Probe (`checkpoints/swin_unetr_linear_probe_best.pt`)
   - FOMO25 AMAES ResEnc-B Linear Probe (`models/foundation_fomo25/amaes_adapter.py`)
4. **Statistical Metrics & Uncertainty Quantification**:
   - Due to $N=69$, point estimates alone are insufficient. All reporting requires **2,000-iteration stratified bootstrap 95% confidence intervals** for ROC-AUC.
   - Balanced Accuracy, Sensitivity, Specificity, Precision, F1 Score, Brier score, and Expected Calibration Error (ECE).
5. **Threshold & Model Selection Freeze**:
   - Decision thresholds remain locked at the standard 0.5 prior and fixed OASIS validation optimal threshold.
   - Zero parameter tuning or threshold adjustments on MIRIAD.
6. **OASIS Test Set Protection**:
   - The locked OASIS-1 test set ($N=30$) remains completely frozen and untouched. All historical checkpoints, predictions, and logs are preserved.

---

## 5. Summary of Verified vs Failed vs Pending Work

| Category | Component / Task | Verdict / Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Verified Outcomes** | MIRIAD DUA & Access Audit | **COMPLETED** | Verified manual approval requirement; updated `dataset_expansion_audit.md`. |
| **Verified Outcomes** | FOMO Checkpoint Download | **COMPLETED** | `weights/resenc_unet_b.ckpt` (1.11 GB, SHA256 verified). |
| **Verified Outcomes** | FOMO Encoder Weight Loading | **COMPLETED** | 448/448 encoder keys matched strictly. |
| **Verified Outcomes** | GPU Smoke Test & VRAM | **COMPLETED** | Forward pass on Blackwell GPU: shape `[1, 320]`, 1.63 GB peak VRAM. |
| **Verified Outcomes** | Protocol Specification | **COMPLETED** | Frozen protocol saved in `results/miriad_external_validation/protocol_specification.json`. |
| **Verified Outcomes** | Pipeline Execution Script | **COMPLETED** | `scripts/run_miriad_external_evaluation.py` ready and tested. |
| **Failed / Blocked** | Automated Anonymous MIRIAD Scraping | **BLOCKED** | Denied by MIRIAD DUA; automated downloading halted to prevent non-compliance. |
| **Pending Work** | MIRIAD Scan Placement | **PENDING USER ACTION** | User registers on UCL XNAT and places baseline NIfTIs in `D:\brain_mri_data\MIRIAD\raw\`. |
| **Pending Work** | MIRIAD Multi-Model Inference | **READY FOR EXECUTION** | Runs automatically via `run_miriad_external_evaluation.py` once scans are present. |

---
*Report completed on October 12, 2026. Deliverable recorded in `results/report/next_phase_execution_report.md`.*
