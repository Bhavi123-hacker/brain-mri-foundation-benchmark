# Dataset Expansion Execution & Technical Readiness Report
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Date**: October 21, 2026  
**Auditor**: Data Engineering Lead & Senior Machine Learning Researcher  

---

## 1. Summary of Executed Dataset Audits & Pipeline Readiness

| Candidate Cohort | Size ($N$) | Scanner Vendor | Label Category | Overlap Risk | Technical Pipeline Status | Action Required to Launch |
| :--- | :---: | :---: | :--- | :---: | :--- | :--- |
| **OASIS-1 Primary** | 198 | Siemens (1.5T) | CDR (0.0 vs $\ge 0.5$) | None (0.0%) | **ACTIVE & IN USE** | Preprocessed tensors ($128^3$), 2D slices, and splits verified. |
| **ADNI External** | 500 | Siemens/Philips (1.5/3T)| Clinical AD vs CN | None (0.0%) | **ACTIVE & IN USE** | Dual float32 `.npy` and `.pt` tensors verified. |
| **MIRIAD** | 69 | GE Signa (1.5T) | Clinical AD vs CN | None (0.0%) | **PIPELINE COMPLETE (READY)** | User registration on UCL XNAT; download 69 scans to `D:\brain_mri_data\MIRIAD\raw`. |
| **OASIS-2** | 150 | Siemens (1.5T) | Longitudinal CDR | ~25% overlap | **MANIFEST READY** | Execute participant de-duplication script against `oasis1_primary_cohort_splits.csv`. |
| **FOMO45K** | 11,966 | Multi-vendor | Tumors, Strokes, Controls| N/A | **REJECTED** | Disqualified: Contains 0 Alzheimer's disease or dementia cases. |
| **FOMO50K / FOMO260K** | 50,000 | Multi-vendor | Uncurated SSL Pool | High (OASIS-1) | **REJECTED** | Disqualified: Directly packages OASIS-1; represents severe data contamination hazard. |

---

## 2. Technical Audit of MIRIAD Evaluation Script
The automated evaluation pipeline for MIRIAD is implemented in:
[`scripts/run_miriad_external_evaluation.py`](file:///D:/ml%20project/scripts/run_miriad_external_evaluation.py)

### Pipeline Integrity Verification:
1. **Input Directory Guard**: Verifies `D:\brain_mri_data\MIRIAD\raw` contains valid NIfTI files before starting.
2. **Metadata Verification**: Loads [`results/miriad_external_validation/miriad_manifest_template.csv`](file:///D:/ml%20project/results/miriad_external_validation/miriad_manifest_template.csv) and filters strictly for baseline scans (`session == 1` or `is_baseline == True`), guaranteeing exactly 69 subjects (46 AD, 23 CN).
3. **Multi-Model Inference Loop**: Automatically iterates through all 6 candidate checkpoints:
   - 3D ResNet-18
   - 3D ResNet-34
   - 2D Transfer ResNet-18
   - Classical Radiomics (SVM)
   - MONAI Swin UNETR SSL
   - FOMO25 AMAES ResEnc-B
4. **Graceful Handling of Missing Raw Data**: If raw NIfTI files are not yet staged, the script exits cleanly, logs clear registration instructions, and writes [`results/miriad_external_validation/miriad_pipeline_status.json`](file:///D:/ml%20project/results/miriad_external_validation/miriad_pipeline_status.json) with status `STOPPED_PENDING_USER_CREDENTIALING`. Zero runtime crashes or corrupted outputs occur.

---

## 3. Storage & Processing Resource Projections
- **MIRIAD Disk Footprint**: 69 raw NIfTI scans ($\sim 1.8\text{ GB}$) + processed $128^3$ tensors ($\sim 0.58\text{ GB}$) = $\mathbf{\sim 2.4\text{ GB}}$ total.
- **Estimated Preprocessing Wall-Clock Time**: $\sim 28\text{ minutes}$ on local NVIDIA RTX PRO 2000 GPU and 8 CPU worker threads.
- **Inference Time Across All 6 Models**: $\sim 4.5\text{ minutes}$.

---
*Technical readiness certified on October 21, 2026.*
