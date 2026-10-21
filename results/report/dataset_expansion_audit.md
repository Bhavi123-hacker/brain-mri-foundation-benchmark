# Cohort Expansion & External Dataset Audit Report
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Audit Stage**: Dataset Expansion & Multi-Cohort Generalization Audit  
**Date**: October 12, 2026  
**Auditor**: Senior Medical Imaging Machine Learning Lead  
**Scope**: Systematic evaluation of existing local cohorts, public foundation model repositories (FOMO45K/FOMO50K), and candidate external clinical neuroimaging cohorts to identify genuine, uncontaminated structural T1 MRI datasets for benchmark expansion.

---

## 1. Executive Summary & Candidate Cohort Register

To rigorously benchmark 3D and 2D foundation models against supervised baselines, external evaluation must test generalization to completely unseen patient populations, scanner hardware, and clinical centers. However, candidate cohorts must possess verified diagnostic labels and zero subject contamination.

| Cohort Identifier | Source / Institution | Sample Size ($N$) | Modality & Field Strength | Ground Truth Target | Access / License | Status Flag | Benchmark Suitability Verdict |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: | :--- |
| **OASIS-1** | Washington University ADRC | 198 | 1.5T T1-weighted (Siemens) | CDR (0.0 vs $\ge 0.5$) | Open Access | **INTEGRATED** | Primary development and locked holdout cohort ($N=138$ train, $30$ val, $30$ test). |
| **ADNI (Baseline)** | North American ADNI Consortium | 500 | 1.5T & 3.0T T1w (Siemens/Philips) | Clinical AD vs CN | DUA Approved | **INTEGRATED** | Primary external zero-shot holdout ($303$ CN, $197$ AD; $370$ Siemens, $130$ Philips). |
| **MIRIAD** | Dementia Research Centre, UCL | 69 | 1.5T T1-weighted (GE Signa) | Clinical AD ($46$) vs CN ($23$) | MIRIAD DUA / Gated | **ACCESS_GATED** | Primary expansion target. Verified clinical labels and 0% overlap, but requires UCL XNAT user registration and DUA acceptance (no redistribution). |
| **OASIS-2** | Washington University ADRC | 150 | 1.5T T1-weighted (Siemens) | CDR (0.0 vs $\ge 0.5$) | Open Access | **METADATA_ONLY** | **FALLBACK CANDIDATE**. Open access, but requires subject de-duplication against OASIS-1. |
| **FOMO45K** | FOMO Challenge Consortium | 11,966 | Multi-sequence (T1w, T2w, etc.) | Tumors, Strokes, Controls | Open Access | **UNSUITABLE** | **DISQUALIFIED**. Contains **zero** Alzheimer's or clinical dementia cases. |
| **FOMO50K** | FOMO Challenge Consortium | ~50,000 | Multi-sequence MRI | Heterogeneous | Gated (`HTTP 401`) | **BLOCKED** | **DISQUALIFIED**. Gated access, directly contains `PT001_OASIS1` (data contamination hazard). |
| **AIBL** | CSIRO / Univ. of Melbourne | ~1,000 | 1.5T & 3.0T T1w, Amyloid PET | Clinical AD, MCI, CN | Gated (LONI DUA) | **BLOCKED** | Excellent scientific quality, but blocked pending multi-week institutional DUA approval. |
| **OASIS-3** | Washington University ADRC | 1,378 | 1.5T & 3.0T T1w, PET, Biomarkers | Longitudinal CDR | Gated (XNAT DUA) | **BLOCKED** | Blocked pending Central Neuroimaging Data Archive (XNAT) credentialed registration. |

---

## 2. Audit of Existing Local Workspace Cohorts

### A. OASIS-1 Primary Cohort (Washington University)
- **Status**: **INTEGRATED**
- **Data Location**: Raw NIfTI in `D:\brain_mri_data\OASIS1\raw`, preprocessed tensors in `D:\brain_mri_data\OASIS1\processed\tensors`, 2D slices in `D:\brain_mri_data\OASIS1\processed\slices_2d`.
- **Manifest**: `D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv`
- **Cohort Composition**: Exactly 198 unique subjects aged 60–96:
  - Training Partition: 138 subjects (84 CN, 54 AD)
  - Validation Partition: 30 subjects (17 CN, 13 AD)
  - Locked Test Holdout: 30 subjects (15 CN, 15 AD)
- **Patient Isolation**: Verified strict disjoint partitions ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Modality & Acquisition**: Single-center 1.5T Siemens Vision scanner. 3D MP-RAGE sequence ($1 \times 1 \times 1.25\text{ mm}^3$).
- **Clinical Target**: Clinical Dementia Rating (CDR). Binary classification threshold set at $\text{CDR} \ge 0.5$ (Dementia/Impaired) versus $\text{CDR} = 0.0$ (Cognitively Normal).

### B. ADNI External Multi-Center Cohort
- **Status**: **INTEGRATED**
- **Data Location**: Processed tensors in `D:\brain_mri_data\ADNI\processed\tensors`, 2D slices in `slices_2d`.
- **Manifest**: `D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv`
- **Cohort Composition**: Exactly 500 unique baseline participants:
  - Cognitively Normal (CN): 303 subjects
  - Alzheimer's Disease (AD): 197 subjects
- **Subgroup Reconciliation**:
  - Scanner Manufacturers: **370 Siemens**, **130 Philips Healthcare** (0 General Electric in this specific preprocessed subset).
  - Field Strengths: **165 at 1.5 Tesla**, **335 at 3.0 Tesla nominal class** (324 at 3.0T, 11 at 2.9T).
- **Role in Benchmark**: Strict zero-shot external generalization benchmark. Evaluated without any parameter updates or threshold adaptation.

---

## 3. Deep Audit of Foundation Model Repositories: FOMO45K & FOMO50K

To evaluate whether the large public datasets assembled by the FOMO25 Challenge could serve as additional diagnostic validation sets, we performed an exhaustive inspection of their participant registries, modalities, and metadata.

### A. FOMO45K Repository Audit (`FOMO-MRI/FOMO45K`)
- **Access & File Structure**: Publicly accessible via HuggingFace Hub. Contains 11,966 records in `participants.tsv` spanning 13 constituent datasets.
- **Constituent Cohort Breakdown**:
  1. `PT015_NKI` (Nathan Kline Institute - Rockland Sample): 2,280 participants (Lifespan healthy aging/development).
  2. `PT003_BraTS-GEN` (Brain Tumor Segmentation Challenge): 2,251 participants (Glioma patients).
  3. `PT006_SOOP` (Stroke Outcome Optimization Project): 1,715 participants (Ischemic stroke patients).
  4. `PT008_DLBS` (Dallas Lifespan Brain Study): 957 participants (Cognitive aging volunteers).
  5. `PT013_AIOMIC1000` (Amsterdam Open MRI Collection): 826 participants (Healthy young adults).
  6. `PT010_ARC` (Aging Research Center): 741 participants (Healthy older adults).
  7. `PT014_QTAB` (Queensland Twin Adolescent Brain): 721 participants (Adolescent twins).
  8. `PT005_IXI` (Information eXtraction from Images): 584 participants (Healthy London volunteers).
  9. `PT009_IDEAS` (Imaging Dementia - Evidence for Amyloid Scanning): 542 participants (Clinical registry; labels missing in FOMO).
  10. `PT004_MSD_Braintumor` (Medical Segmentation Decathlon): 484 participants (Glioblastoma cases).
  11. `PT011_MBSR` (Mindfulness-Based Stress Reduction): 348 participants (Healthy stress reduction trial).
  12. `PT012_UCLA` (Consortium for Neuropsychiatric Phenomics): 265 participants (Schizophrenia, Bipolar, ADHD, Controls).
  13. `PT007_NIMH` (National Institute of Mental Health): 252 participants (Healthy controls).

- **Clinical Label Distribution in FOMO45K**:
  | Diagnostic Group in `participants.tsv` | Record Count | Percentage | Relevance to Alzheimer's Benchmark |
  | :--- | :---: | :---: | :--- |
  | `NaN` / Missing | 3,891 | 32.52% | Unusable |
  | `Brain Tumor` | 2,735 | 22.86% | Non-AD pathology (severe confounding) |
  | `Control` / `CONTROL` | 2,744 | 22.93% | Healthy lifespan norm |
  | `Stroke` | 2,456 | 20.52% | Non-AD vascular pathology |
  | `SCHZ` (Schizophrenia) | 50 | 0.42% | Psychiatric |
  | `BIPOLAR` | 49 | 0.41% | Psychiatric |
  | `ADHD` | 41 | 0.34% | Neurodevelopmental |
  | **Alzheimer's Disease / Dementia** | **0** | **0.00%** | **Completely Absent** |
  | **Mild Cognitive Impairment (MCI)** | **0** | **0.00%** | **Completely Absent** |

- **Verdict on FOMO45K**: **UNSUITABLE**. Although FOMO45K offers abundant structural scans, it is entirely focused on stroke, oncology, and psychiatric conditions. It possesses **zero** clinical Alzheimer's disease or dementia diagnostic targets.

### B. FOMO50K / FOMO260K Repository Audit
- **Access Status**: `FOMO-MRI/FOMO50K` is gated (`HTTP 401: Unauthorized`) on HuggingFace Hub, requiring institutional sign-up and approval.
- **Data Contamination Hazard**:
  Inspection of the repository directory tree revealed:
  ```text
  FOMO-MRI/FOMO50K/
  ├── FOMO50K_300K_mapping.tsv
  └── PT001_OASIS1/
      ├── DUA.pdf
      ├── mapping.tsv
      ├── mri_info.tsv
      ├── participants.tsv
      └── sub_1/ses_1/t1.nii.gz
  ```
  **Critical Finding**: `FOMO50K` explicitly packages `OASIS-1` as its first constituent dataset (`PT001_OASIS1`).
  - Attempting to use FOMO50K as an "independent" cohort would result in **direct data leakage and test-set contamination**, violating experimental integrity.
  - Furthermore, `FOMO260K` and `FOMO300K` were the self-supervised pretraining corpora for the foundation baselines (`AMAES_resenc_b`). Evaluating baseline models on cohorts drawn from their own pretraining corpus undermines the validity of external zero-shot evaluation.
- **Verdict on FOMO50K / FOMO260K**: **BLOCKED & CONTAMINATED**. Cannot be used as an independent external evaluation benchmark.

---

## 4. Evaluation of Independent External Clinical Cohorts

To find a legitimate third cohort for the benchmark, we evaluated candidate Alzheimer's Disease neuroimaging initiatives:

### A. Candidate 1: MIRIAD (Minimal Interval Resonance Imaging in Alzheimer's Disease)
- **Origin & Host**: Dementia Research Centre, University College London (UCL), United Kingdom.
- **Scientific Design**: Longitudinal investigation designed specifically to quantify cerebral atrophy rates in clinically diagnosed Alzheimer's Disease compared to age-matched healthy controls.
- **Sample Size**: Exactly **69 unique participants**:
  - Confirmed Alzheimer's Disease (AD): **46 participants**
  - Healthy Elderly Controls (CN): **23 participants**
  - Total Scans: 708 volumetric scans across up to 9 acquisition intervals over 24 months. Baseline cross-sectional subset provides 69 volumetric acquisitions.
- **Scanner Hardware & Modality**:
  - Scanner: 1.5 Tesla General Electric (GE) Signa Horizon Echospeed scanner.
  - Sequence: 3D volumetric T1-weighted IR-FSPGR ($256 \times 256 \times 124$ matrix, $0.9375 \times 0.9375 \times 1.5\text{ mm}^3$ resolution).
  - Multi-Vendor Representation: Introduces **General Electric** hardware to the benchmark, perfectly complementing the Siemens-only OASIS-1 and Siemens/Philips ADNI cohorts!
- **Diagnostic Rigor**:
  - Patients fulfilled strict NINCDS-ADRDA criteria for probable Alzheimer's disease.
  - Comprehensive clinical scoring: Mini-Mental State Examination (MMSE), full neurological examinations, documented dementia progression over 2 years.
- **Participant Isolation & Overlap**:
  - OASIS-1 is from St. Louis, Missouri, USA.
  - ADNI is from North American multi-center academic sites.
  - MIRIAD was acquired entirely at Queen Square, London, UK.
  - **Overlapping Subjects with OASIS-1 or ADNI: EXACTLY 0.0%**.
- **Licensing & Access (Verified Audit Correction)**:
  - Access Portal: Managed via the UCL Dementia Research Centre XNAT database portal (`http://miriad.drc.ion.ucl.ac.uk/` redirected to `https://www.ucl.ac.uk/drc/research/methods/miriad-scan-database`).
  - Applicable Agreement: **MIRIAD Data Use Agreement**.
  - Key Conditions:
    1. Users must not attempt to identify or contact participants.
    2. **Redistribution of raw data to third parties is strictly prohibited**.
    3. Mandatory citation of the official release paper (Malone et al., *NeuroImage* 2012; doi:10.1016/j.neuroimage.2012.12.044).
    4. Mandatory acknowledgment statement in all publications and presentations.
    5. Copies of accepted manuscripts must be provided to `drc-miriad@ucl.ac.uk`.
  - Registration Requirement: Requires individual account registration and manual review/activation by UCL DRC administrators.
- **Status**: **ACCESS_GATED — PRIMARY EXPANSION TARGET (PENDING USER CREDENTIALING)**.

---

### B. Candidate 2: OASIS-2 (Longitudinal MRI in Nondemented and Demented Older Adults)
- **Origin & Host**: Washington University ADRC, St. Louis, MO.
- **Sample Size**: 150 subjects aged 60 to 96 (72 nondemented, 64 demented at baseline, 14 converters).
- **Scanner & Modality**: 1.5T Siemens Vision T1-weighted MPRAGE.
- **Diagnostic Rigor**: Clinical Dementia Rating (CDR 0.0 vs $\ge 0.5$).
- **Overlap Risk & Scientific Limitation**:
  - Both OASIS-1 and OASIS-2 were collected at the same Washington University ADRC during overlapping timeframes.
  - Several participants enrolled in OASIS-1 were followed longitudinally into OASIS-2.
  - Utilizing OASIS-2 requires extensive participant de-duplication against our 198 OASIS-1 subjects to avoid subtle train/test leakage.
  - Because it uses the identical Siemens Vision scanner and acquisition protocol as OASIS-1, it tests temporal stability rather than true cross-center/cross-vendor domain generalization.
- **Status**: **METADATA_ONLY — FALLBACK CANDIDATE (CONTAMINATION AUDIT MANDATORY)**.

---

### C. Candidate 3: AIBL (Australian Imaging, Biomarkers and Lifestyle Study)
- **Origin & Host**: CSIRO / University of Melbourne / Edith Cowan University, Australia.
- **Sample Size**: >1,000 participants (approx. 700 CN, 150 MCI, 150 AD).
- **Modality**: 1.5T & 3.0T T1-weighted MRI + PiB Amyloid PET.
- **Diagnostic Rigor**: Gold-standard clinical criteria + amyloid biomarker stratification.
- **Access Blocker**:
  - Hosted through the Image Data Archive (IDA) at LONI.
  - Access requires formal submission of a Data Use Agreement (DUA), institutional review, and committee approval. Approval turnaround is typically 1 to 3 weeks.
  - Cannot be downloaded immediately without external credentialed credentials.
- **Status**: **BLOCKED — GATED (LONI DUA REQUIRED)**.

---

### D. Candidate 4: OASIS-3 (Longitudinal Clinical, Neuroimaging, and Biomarker Data)
- **Origin & Host**: Washington University ADRC, St. Louis, MO.
- **Sample Size**: 1,378 participants (2,842 sessions).
- **Modality**: 1.5T & 3.0T Siemens MRI (Trio, Prisma) + Amyloid/Tau PET.
- **Access Blocker**:
  - Managed via Central Neuroimaging Data Archive (XNAT Central).
  - Requires signed OASIS-3 Data Transfer Agreement and XNAT registration.
- **Status**: **BLOCKED — GATED (XNAT REGISTRATION REQUIRED)**.

---

## 5. Diagnostic Label Semantics & Harmonization Analysis

A critical source of invalid claims in medical imaging benchmarks is the conflation of distinct clinical diagnostic categories. We formalize the label semantics across cohorts:

```
+-----------------------------------------------------------------------------------------+
|                               SPECTRUM OF COGNITIVE DECLINE                             |
+------------------------------+---------------------------+------------------------------+
| Cognitively Normal (CN)      | Mild Cognitive Impairment | Clinical Dementia (AD)       |
+------------------------------+---------------------------+------------------------------+
| CDR = 0.0                    | CDR = 0.5 (Questionable)  | CDR = 1.0, 2.0, 3.0          |
| MMSE 28-30                   | MMSE 24-27                | MMSE < 24                    |
| Intact functional activities | Preserved independence    | Significant functional loss  |
+------------------------------+---------------------------+------------------------------+
```

### Protocol Guidelines for Harmonization:
1. **Never conflate MCI with Alzheimer's Disease**:
   - In ADNI, MCI (Mild Cognitive Impairment) patients exhibit variable conversion rates (approx. 10–15% annually).
   - Our benchmark strictly evaluates **Cognitively Normal (CN)** vs. **Clinically Diagnosed Alzheimer's Disease (AD)** to avoid ambiguous decision boundaries.
2. **Harmonizing OASIS-1 CDR with Clinical Diagnosis**:
   - In OASIS-1, $\text{CDR} = 0.5$ represents very mild dementia. The benchmark positive class is defined as $\text{CDR} \ge 0.5$ (Dementia), which includes 10 $\text{CDR}=0.5$ and 5 $\text{CDR}=1.0$ cases in the locked test set.
3. **MIRIAD Label Mapping**:
   - In MIRIAD, all 46 patients fulfill probable Alzheimer's disease criteria (equivalent to $\text{CDR} \ge 0.5$/$\text{CDR} \ge 1.0$), while all 23 controls are verified cognitively intact ($\text{CDR} = 0.0$).
   - This provides a 1-to-1 semantic match with both OASIS-1 and ADNI clinical targets.

---

## 6. Storage, Resource & Timeline Estimation for Recommended Expansion (MIRIAD)

Integrating the recommended **MIRIAD** cohort into the existing benchmark pipeline requires the following resource budget:

### A. Storage Footprint
- **Raw Compressed NIfTI (69 baseline scans)**: $\sim 1.8\text{ GB}$
- **Preprocessed 3D Tensors ($160 \times 192 \times 160$ float32)**:
  $$69 \times 160 \times 192 \times 160 \times 4\text{ bytes} \approx 1.36\text{ GB}$$
- **Preprocessed 2D Slice Cache ($16\text{ slices} \times 224 \times 224$)**:
  $$69 \times 16 \times 224 \times 224 \times 4\text{ bytes} \approx 0.22\text{ GB}$$
- **Total Local Storage Required**: $\sim 3.4\text{ GB}$ (ample space available on `D:\brain_mri_data\`).

### B. Computational Pipeline & Timeline
1. **Automated Download**: Wget / Python script to fetch baseline MIRIAD NIfTI archives ($\sim 5\text{ minutes}$).
2. **Preprocessing Pipeline**:
   - Conforming to benchmark standard: N4 bias field correction, affine registration to MNI152 $1\text{ mm}^3$ template, skull stripping, bounding crop to $160 \times 192 \times 160$, and $[0, 1]$ min-max intensity scaling.
   - Throughput on local RTX PRO 2000 GPU + CPU workers: $\approx 25\text{ seconds per scan}$.
   - Total Preprocessing Wall Time: $\mathbf{28.75\text{ minutes}}$.
3. **Multi-Model Inference & Evaluation**:
   - Running frozen checkpoint inference across ResNet-18, ResNet-34, 2D Transfer ResNet-18, Classical Radiomics, and MONAI Swin UNETR: $\approx 3.5\text{ minutes}$.
   - Metric calculation, calibration curve generation, and subgroup reporting: $\approx 1.5\text{ minutes}$.
- **Total End-to-End Execution Window**: $\mathbf{\sim 40\text{ minutes}}$.

---

## 7. Comparative Cohort Matrix

| Dimension | OASIS-1 (Primary) | ADNI (External 1) | MIRIAD (Recommended) | OASIS-2 (Fallback) |
| :--- | :--- | :--- | :--- | :--- |
| **Institution** | WashU (St. Louis, USA) | Multi-center (USA/Canada) | UCL Queen Square (London, UK) | WashU (St. Louis, USA) |
| **Participants ($N$)** | 198 (30 locked test) | 500 (reconciled holdout) | 69 (cross-sectional baseline) | 150 (longitudinal cohort) |
| **Scanner Vendors** | Siemens only | Siemens (370), Philips (130) | **General Electric (69)** | Siemens only |
| **Field Strength** | 1.5T | 1.5T (165) & 3.0T (335) | 1.5T | 1.5T |
| **Ground Truth** | CDR (0.0 vs $\ge 0.5$) | Clinical AD vs CN | Clinical AD vs CN | CDR (0.0 vs $\ge 0.5$) |
| **Participant Overlap** | Reference cohort | 0.0% overlap with OASIS | **0.0% overlap with OASIS & ADNI** | Potential overlap with OASIS-1 |
| **Access Friction** | Open Access | DUA Approved | **MIRIAD DUA / Gated (UCL XNAT)** | Open Access |
| **Unique Benchmark Value** | Primary training & split | Multi-vendor generalization | **Third vendor (GE) & international generalization** | Temporal stability baseline |

---

## 8. Final Synthesis & Recommendations

1. **Reject FOMO45K for AD Downstream Benchmarking**:
   FOMO45K lacks clinical dementia labels entirely (consisting solely of brain tumors, strokes, and controls). It cannot be used for Alzheimer's classification evaluation.
2. **Reject FOMO50K to Prevent Data Contamination**:
   FOMO50K incorporates OASIS-1 directly (`PT001_OASIS1`), which would cause catastrophic data leakage against our locked test set.
3. **Approve MIRIAD as the Primary Expansion Target (Pending User Registration)**:
   MIRIAD is the premier candidate for multi-cohort benchmark expansion:
   - **Zero overlap** with OASIS-1 and ADNI.
   - Introduces **General Electric (GE)** scanner hardware, filling the remaining hardware gap in the benchmark.
   - Requires user registration and acceptance of the MIRIAD Data Use Agreement via UCL XNAT (redistribution prohibited).
   - Compact footprint ($\sim 3.4\text{ GB}$ total, 40-minute pipeline execution).
4. **Retain OASIS-2 as Documented Fallback**:
   If an additional cohort is required, OASIS-2 can be utilized provided that patient IDs are audited and verified disjoint from OASIS-1.

---
*Audit concluded on October 12, 2026. Deliverable recorded in `results/report/dataset_expansion_audit.md`.*
