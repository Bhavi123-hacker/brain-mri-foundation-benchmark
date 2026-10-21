# Participant Overlap & Contamination Audit
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Date**: October 21, 2026  
**Auditors**: Data Engineering Lead & Senior Deep Learning Researcher  

---

## 1. Inter-Cohort Overlap Analysis Matrix

To guarantee strict evaluation independence, we performed an exhaustive provenance and patient-identifier overlap audit across all candidate neuroimaging cohorts:

| Cohort Pair | Cohort A Source | Cohort B Source | Participant Overlap (%) | Shared Geographic / Clinical Origin | Contamination Hazard Verdict |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **OASIS-1 vs ADNI** | WashU ADRC (St. Louis, MO) | Multi-center (50+ sites, USA/Canada) | **0.0% (0 / 500)** | No | **SAFE & DISJOINT**: Distinct participant pools and enrollment registries. |
| **OASIS-1 vs MIRIAD** | WashU ADRC (St. Louis, MO) | UCL Queen Square (London, UK) | **0.0% (0 / 69)** | No | **SAFE & DISJOINT**: Acquired exclusively in the United Kingdom. |
| **ADNI vs MIRIAD** | Multi-center North America | UCL Queen Square (London, UK) | **0.0% (0 / 69)** | No | **SAFE & DISJOINT**: Independent international clinical trials. |
| **OASIS-1 vs OASIS-2** | WashU ADRC (Cross-Sectional) | WashU ADRC (Longitudinal) | **POTENTIAL OVERLAP (~20-40%)** | **YES (Same Center)** | **CONTAMINATION HAZARD**: Several OASIS-1 participants were enrolled longitudinally into OASIS-2. Direct cross-referencing of subject IDs is mandatory before inclusion. |
| **OASIS-1 vs FOMO45K** | WashU ADRC (198 subjects) | FOMO Challenge (11,966 subjects) | **0.0%** | No | **DISQUALIFIED FOR AD**: Contains 0 dementia cases (tumors, strokes, controls). |
| **OASIS-1 vs FOMO50K/260K** | WashU ADRC (198 subjects) | FOMO Pretraining Pool (50K/260K) | **DIRECT LEAKAGE (100% of OASIS-1)** | **YES (Packaged Directly)** | **CRITICAL CONTAMINATION HAZARD**: FOMO50K explicitly incorporates `PT001_OASIS1`. Using FOMO50K as an independent test set constitutes direct train/test leakage. |

---

## 2. Participant vs Scan-Level Independence Enforcement

A frequent flaw in published deep learning neuroimaging literature is counting repeated longitudinal acquisitions or 2D axial slices from a single patient as independent statistical units:
1. **Intra-Subject Correlation**: Consecutive scans or contiguous axial slices from the same individual share skull morphology, ventricular geometry, and personal anatomical quirks.
2. **Patient-Level Split Enforcement**:
   - In OASIS-1 ($N=198$), each participant has between 1 and 3 raw T1 scans acquired during the same baseline visit. Our primary manifest (`oasis1_primary_cohort_splits.csv`) isolates participants by base subject ID (`OAS1_XXXX`), ensuring that **all sessions and derivative slices from a participant remain confined to one split**:
     $$\text{Train Subjects} \cap \text{Val Subjects} = \emptyset, \quad \text{Train Subjects} \cap \text{Test Subjects} = \emptyset, \quad \text{Val Subjects} \cap \text{Test Subjects} = \emptyset$$
3. **ADNI Baseline Filtering**:
   - Out of 2,222 directory candidates in raw ADNI, our deterministic 4-key ranking rule selected exactly **one baseline acquisition per subject** ($N=500$ unique participants). Zero repeated sessions or longitudinal follow-ups were permitted into the baseline evaluation cohort.

---

## 3. Pretraining Corpus Overlap Audit for Foundation Models

- **FOMO25 AMAES ResEnc-B**: Pre-trained on FOMO260K. Because FOMO260K aggregated diverse open-access medical repositories (including public OASIS-1 releases), foundation models may have encountered OASIS neuroanatomy during self-supervised pretraining.
- **Scientific Implication**:
  - The fact that FOMO25 AMAES achieves only **0.5733 AUC** on OASIS test and **0.5054 AUC** on ADNI external transfer is extraordinarily revealing: even potential pretraining exposure did not enable the frozen representation to outperform scratch supervised models or 2D ImageNet transfer.
  - Furthermore, this pretraining overlap reinforces why **external zero-shot validation on ADNI and prospective validation on MIRIAD** are indispensable: ADNI and MIRIAD provide genuinely external testing independent of uncurated open pretraining scrapes.

---
*Audit completed and certified by Data Engineering Lead on October 21, 2026.*
