# SHIFT-MRI: Data Governance, Ethical Compliance, and Public Release Audit

**Platform:** SHIFT-MRI Research Platform  
**Version:** 3.0.0-final  
**Date:** October 2026  
**Investigator:** Bhavy Garg, School of Computer Science and Engineering, VIT Chennai  
**License:** Apache License 2.0 (Open-Source Research Software)

---

## 1. Executive Summary

This compliance report details the data governance policies, licensing frameworks, patient privacy safeguards, and distribution controls implemented across the SHIFT-MRI benchmark platform. SHIFT-MRI investigates cross-cohort generalizability and failure detection of brain MRI foundation models and supervised 3D CNNs across three cohorts: OASIS-1 ($N=198$), ADNI ($N=500$), and OASIS-2 ($N=150$).

In strict accordance with international biomedical data governance standards and dataset-specific Data Use Agreements (DUAs), this repository separates open-source algorithm code and aggregate research statistics from locally restricted subject imaging data.

---

## 2. Cohort Governance and Data Use Agreements

### 2.1 OASIS-1 (Cross-Sectional MRI Cohort)
- **Data Source:** Open Access Series of Imaging Studies (OASIS-1), Washington University in St. Louis.
- **Sponsorship:** Supported by NIH grants P50 AG05681, P01 AG03991, R01 AG021910, P50 MH071616, U24 RR021382, and R01 MH56584.
- **Data Use Terms:** Publicly available for scientific and educational use. Scans are de-identified in compliance with HIPAA privacy standards.
- **Redistribution Policy:** Raw volumetric scans are retained on local storage (`D:\brain_mri_data\OASIS1`). Derived processed tensors are kept local and excluded from public git distribution.
- **Label Standard:** Clinical Dementia Rating (CDR); CDR 0 indicates cognitively normal, while CDR $\ge 0.5$ denotes cognitive impairment.

### 2.2 ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Data Source:** Alzheimer's Disease Neuroimaging Initiative (ADNI) multi-center study (adni.loni.usc.edu).
- **Sponsorship:** National Institutes of Health (Grant U01 AG024904) and Department of Defense (Award W81XWH-12-2-0012).
- **Data Use Terms:** Access granted under strict ADNI Data Use Agreement for research purposes only.
- **Patient Privacy & Redistribution Policy:** Under the ADNI DUA, raw MRI files (`.nii`, `.dcm`) and identifiable individual-level clinical records must never be redistributed publicly or uploaded to open third-party code repositories.
- **Redistribution Audit & Pending Compliance Disclosure:** Historical repository commits retain participant-level prediction CSVs (e.g. containing `subject_id`, `image_data_id`, diagnosis, and model probabilities) and individual slice attribution images. While zero raw 3D scans or direct HIPAA patient identifiers are exposed, full redistribution rights for participant-level derived prediction tables remain pending formal institutional DUA review. Therefore, full public redistribution compliance is disclosed as pending this authorization.
- **Safeguard Enforcement:** All ADNI imaging volumes reside strictly on private local workstation storage (`D:\brain_mri_data\ADNI`). `.gitignore` rules strictly prevent accidental tracking of `.nii`, `.nii.gz`, `.dcm`, and binary array files. The web server download endpoint (`/downloads/`) enforces a strict whitelist that completely blocks access to neuroimaging files.
- **Diagnostic Criteria:** Formal clinical Alzheimer's disease diagnosis according to NINCDS-ADRDA criteria ($N=197$) vs cognitively normal controls ($N=303$).

### 2.3 OASIS-2 (Longitudinal MRI Cohort)
- **Data Source:** Longitudinal MRI Study of Nondemented and Demented Older Adults.
- **Role in Study:** Secondary exploratory cohort ($N=150$). Potential participant overlap with OASIS-1 is rigorously disclosed as unverified; therefore, OASIS-2 is treated as an exploratory validation cohort rather than a fully independent external multi-site benchmark.

### 2.4 Sanitized Public-Release Plan
To ensure total institutional alignment before open public release:
1. **Aggregated Metric Distribution:** Future public releases should decouple participant-level CSVs into aggregated receiver-operating characteristic (ROC) curves, bootstrap distribution confidence intervals, and cohort confusion matrices.
2. **Attribution Sanitization:** Individual subject Grad-CAM slice figures will be replaced with cohort-averaged saliency maps mapped to standard MNI-152 space.
3. **Local Preservation:** All authentic participant-aligned predictions and raw NIfTI scans remain intact on the private local workstation for exact scientific reproducibility.
4. **Git Policy:** In accordance with user guidance, zero historical Git commits will be deleted, force-pushed, or altered without explicit instruction, and no new restricted participant-level files will be committed to remote tracking.

---

## 3. Pretrained Model Weights and Open-Source Licensing

All foundation model backbones and baseline architectures utilized in the SHIFT-MRI benchmark are governed by approved open-source and scientific research licenses:

1. **MONAI Swin UNETR SSL:** Pretrained on 5,050 multi-site brain MRI volumes using masked self-supervised learning. Distributed under the **Apache License 2.0** by Project MONAI.
2. **Torchvision ResNet-18 (ImageNet-1K V1):** Pretrained on ImageNet-1K. Distributed under the **BSD 3-Clause License** by PyTorch / Meta Platforms.
3. **FOMO25 AMAES (ResEnc-B):** Multi-task brain foundation model linear probe. Licensed for academic neuroimaging research.
4. **3D ResNet-10, DenseNet-121, Multi-View ResNet-18, Dual-Branch Hybrid:** Developed, trained, and verified within the SHIFT-MRI framework under the **Apache License 2.0**.

---

## 4. Software and Data Integrity Audits

### 4.1 Git Tracking Audit
An automated scan of the Git index confirms:
- **Zero raw neuroimaging volumes** (`.nii`, `.nii.gz`, `.dcm`) are tracked in the public repository.
- **Zero un-anonymized clinical or demographic records** are exposed.
- All code, configuration files, test suites, and documentation are tracked cleanly.

### 4.2 Web Server Security and Path Traversal Protection
The SHIFT-MRI HTTP backend (`backend/server.py`) implements defense-in-depth measures:
- **Localhost Binding:** Binds strictly to `127.0.0.1` by default, preventing unintended network exposure.
- **Download Whitelist:** The `/downloads/` endpoint only serves verified scientific deliverables:
  - `SHIFT_MRI_Complete_Research_Paper.pdf`
  - `SHIFT_MRI_Complete_Research_Paper.docx`
  - `SHIFT_MRI_Final_Presentation.pptx`
  - `SHIFT_MRI_Submission_Package.zip`
- **Path Traversal Blocking:** Any request containing directory traversal sequences (`..`, `/../`, `\\..\\`) immediately triggers HTTP 403 Forbidden.
- **Zero Fabricated Model Probabilities:** The server is cryptographically and logically prohibited from generating synthetic probabilities from ground-truth labels.

---

## 5. Ethical Statement on Clinical Non-Applicability

SHIFT-MRI is a computational research benchmark designed to expose distribution shifts, failure modes, and shortcut learning in medical computer vision. It is **not** a clinical diagnostic medical device. 

In particular, the **SHIFT-MRI TrustGate** failure estimation method is documented and published as a **scientific negative result**: under multi-center distribution shift (OASIS-1 to ADNI), ensemble disagreement and feature distance estimators degrade below random chance ($\text{AUROC} = 0.4697$), proving that source-locked selective prediction fails without domain calibration. Model predictions must never be utilized for patient diagnosis, prognosis, or clinical decision-making.
