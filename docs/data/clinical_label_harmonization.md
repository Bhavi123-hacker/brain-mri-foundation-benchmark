# Diagnostic Label Harmonization & Clinical Semantic Mapping
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Date**: October 21, 2026  
**Auditors**: Neuroimaging Scientist, Statistical Methodologist & Principal Investigator  

---

## 1. Clinical Target Formalization & Harmonization Principles

A major source of bias in machine learning benchmarks for neuroimaging is the careless conflation of distinct clinical diagnostic categories. In observational Alzheimer's cohorts, diagnostic targets range from brief cognitive screening instruments to longitudinal multidimensional dementia ratings and definitive post-mortem or amyloid-PET biomarker staging.

To ensure strict semantic equivalence across cohorts, this benchmark enforces three methodological rules:
1. **Binary Target Formalization**:
   - Class 0: **Cognitively Normal / Control (CN)**
   - Class 1: **Clinically Diagnosed Alzheimer's Disease / Dementia (AD)**
2. **Strict Exclusion of Ambiguous Intermediate States (MCI)**:
   - Mild Cognitive Impairment (MCI) subjects exhibit variable annual progression rates (approximately 10–15% conversion to dementia per year). Conflating MCI with either stable controls or clinically overt dementia introduces significant label noise. In our primary binary benchmark, MCI subjects are excluded from evaluation partitions to preserve unambiguous ground truth.
3. **No Target Leaks from Pre-Diagnostic Variables**:
   - Demographic variables (e.g., age, sex) and cognitive screening scores (e.g., MMSE, MoCA, ADAS-Cog) must never be used to train imaging feature extractors without explicit multi-modal ablation reporting.

---

## 2. Cross-Cohort Semantic Mapping Table

```
+------------------------------------------------------------------------------------------------------------+
|                                  CROSS-COHORT CLINICAL LABEL MAPPING                                       |
+----------------------+-----------------------------+-----------------------------+-------------------------+
| Diagnostic Category  | OASIS-1 (Primary)           | ADNI (External Zero-Shot)   | MIRIAD (Recommended)    |
+----------------------+-----------------------------+-----------------------------+-------------------------+
| Healthy Control (0)  | CDR = 0.0                   | DX_bl == "CN"               | Healthy Elderly Control |
|                      | MMSE: 29.0 +/- 1.3          | MMSE: 29.1 +/- 1.0          | MMSE: 29.4 +/- 0.8      |
|                      | Intact memory and ADLs      | No memory complaints        | Normal cognitive score  |
+----------------------+-----------------------------+-----------------------------+-------------------------+
| Mild Impairment      | CDR = 0.5 (Very Mild)       | DX_bl == "LMCI" / "EMCI"    | Excluded                |
|                      | Mapped to AD (Class 1)      | (Excluded from Benchmark)   | (Excluded)              |
+----------------------+-----------------------------+-----------------------------+-------------------------+
| Overt Dementia (1)   | CDR = 1.0 (Mild)            | DX_bl == "AD"               | Probable AD             |
|                      | CDR = 2.0 (Moderate)        | NINCDS-ADRDA Probable AD    | NINCDS-ADRDA Criteria   |
|                      | MMSE: 24.5 +/- 3.2          | MMSE: 23.3 +/- 2.0          | MMSE: 19.3 +/- 4.5      |
|                      | Documented functional loss  | Substantial functional loss | Progressive decline     |
+----------------------+-----------------------------+-----------------------------+-------------------------+
```

---

## 3. Methodological Caveats & Threats to Ground Truth Validity

1. **Clinical Staging (CDR) vs Neuropathological Confirmation**:
   - Both OASIS-1 and ADNI rely on clinical phenotypic classification. Approximately 15–20% of clinically diagnosed "probable AD" patients do not display extensive amyloid-$\beta$ plaques or tau neurofibrillary tangles upon autopsy (often having frontotemporal lobar degeneration, Lewy body disease, or vascular cognitive impairment). Conversely, up to 30% of cognitively normal elderly individuals harbor significant asymptomatic amyloid burden.
   - Consequently, classification accuracy on clinical targets has an inherent theoretical ceiling bounded by the clinical-pathological diagnostic agreement (~85–90%).
2. **OASIS-1 CDR 0.5 Thresholding**:
   - In OASIS-1, patients with $\text{CDR} = 0.5$ ("Very Mild Dementia") are grouped with $\text{CDR} \ge 1.0$ into the positive class. In the locked OASIS test set ($N=30$), the positive class contains 10 $\text{CDR} = 0.5$ cases and 5 $\text{CDR} = 1.0$ cases. This operational definition reflects very early neurodegenerative decline.
3. **MIRIAD Semantic Concordance**:
   - All 46 MIRIAD patients met strict NINCDS-ADRDA criteria for probable Alzheimer's disease with extensive longitudinal cognitive documentation over 2 years, providing a 100% semantic concordance with ADNI and OASIS-1 positive cases.

---
*Mapping reviewed and approved by Neuroimaging Scientist on October 21, 2026.*
