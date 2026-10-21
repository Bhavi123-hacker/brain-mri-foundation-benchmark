# SHIFT-MRI: Multi-Cohort Dataset Discovery, Provenance, and Governance Audit Report

**Benchmark Version:** V3.0 (SHIFT-MRI Framework)  
**Date:** October 2026  
**Compliance Standard:** CLAIM 2024 / TRIPOD+AI 2024 / Institutional Data Governance  

---

## 1. Executive Summary

A critical priority identified during the independent scientific verification was addressing contradictory claims regarding the **independence of the OASIS-2 cohort**, auditing data-use agreements (DUAs), and evaluating immediately accessible external brain MRI datasets ahead of the October 22, 2026 deadline.

This report establishes:
1. **The Definitive Overlap Status of OASIS-2:** The OASIS-1 (cross-sectional) and OASIS-2 (longitudinal) cohorts both originate from the Washington University Alzheimer's Disease Research Center (ADRC) in St. Louis, Missouri. Because longitudinal studies routinely recruit from existing cross-sectional participant registries, differences in participant identifier namespaces (`OAS1_XXXX` vs `sub-OASIS2XXXX`) do **not** establish independent participant recruitment. Therefore, claims describing OASIS-2 as a "confirmed disjoint independent external cohort" are scientifically invalid and have been officially withdrawn. OASIS-2 is strictly classified as a **Secondary Exploratory Cohort with Unverified Participant Overlap**.
2. **Licensing and Data-Use Governance:** While repositories such as Hugging Face Radiata apply an Open Data Commons Attribution (ODC-By 1.0) license to their packaging and metadata, the underlying MRI scans remain bound to the original OASIS data-use agreements, which prohibit participant re-identification and enforce educational/research terms.
3. **Audit of Candidates for Immediate Expansion:** An exhaustive survey across Hugging Face, OpenNeuro, Zenodo, and academic repositories confirms that restricted providers (OASIS-3, AIBL) require 2–4 week approval timelines. Open access candidates (OpenNeuro lifespan datasets) lack matched dementia clinical endpoints. The benchmark appropriately locks its empirical validation to three verified volumetric cohorts ($N=848$ total scans).

---

## 2. Cohort Provenance and Characteristic Summary

| Cohort Name | Sample Size ($N$) | Clinical Classes | Acquisition Hardware | Field Strength | Benchmark Status | Overlap with OASIS-1 |
| :--- | :---: | :--- | :--- | :---: | :--- | :---: |
| **OASIS-1 Primary** | 198 | 134 CN, 64 AD (CDR 0.5: 44, CDR 1: 18, CDR 2: 2) | Siemens Vision (Single-Site) | 1.5 Tesla | **Primary Benchmark** (138 Train, 30 Val, 30 Locked Test) | Source Benchmark |
| **ADNI External** | 500 | 303 CN, 197 AD (Matched Baseline) | Multi-Site (GE, Siemens, Philips) | 1.5T (165) & 3.0T (335) | **External Zero-Shot Test** | **Zero Overlap** (Confirmed Disjoint) |
| **OASIS-2 Exploratory** | 150 | 85 Nondemented, 65 Demented | Siemens Vision (Single-Site) | 1.5 Tesla | **Secondary Exploratory** | **Unverified Overlap** (Shared ADRC Registry) |

---

## 3. Deep-Dive Audit: The OASIS-1 vs OASIS-2 Overlap Contradiction

### 3.1 Origin of the Contradiction
Earlier draft summaries asserted that OASIS-2 represented a "confirmed 100% disjoint external cohort" based solely on the fact that participant IDs in OASIS-1 follow the regex `^OAS1_\d{4}$` while OASIS-2 participant IDs in BIDS format follow `^sub-OASIS2\d{4}$`.

However, the repository's internal participant audit script (`scripts/validate_oasis1_integrity.py` and historical logs) previously noted that both cohorts drew from the longitudinal volunteer pool at the Washington University ADRC between 1988 and 2004. Washington University assigned unique cross-sectional study session codes upon initial visit, but longitudinal participants returning for subsequent waves were frequently re-coded under longitudinal study protocols without public cross-referencing keys.

### 3.2 Corrective Action and Scientific Disclosure
In accordance with CLAIM 2024 item 4 ("Describe characteristics of study cohorts, including geographic location, setting, and potential overlap") and PROBAST+AI Domain 1:
- We **withdraw all assertions** that OASIS-2 serves as an independent external validation of generalization across clinical institutions.
- We formally classify OASIS-2 as an **Exploratory Secondary Cohort (Same-Center Longitudinal Follow-Up / Unverified Participant Overlap)**.
- All primary conclusions regarding external domain shift and cross-cohort reliability are anchored strictly on **ADNI ($N=500$)**, which is geographically, institutionally, and operationally completely disjoint from OASIS.

---

## 4. Licensing and Ethical Data Governance

### 4.1 Hugging Face Radiata Mirror Terms
The Hugging Face dataset `radiata/oasis2` is distributed under an Open Data Commons Attribution (ODC-By 1.0) license. A common legal misunderstanding is assuming that ODC-By applies unconditionally to the underlying medical imaging data. 

In medical AI governance, collection-level licenses govern the metadata schema, feature tables, and hosting compilation. The raw volumetric imaging tensors remain governed by the **OASIS Data Use Agreement**:
1. Scans are provided strictly for scientific research and educational purposes.
2. Users agree not to attempt to re-identify any research participant.
3. No redistribution of raw protected health data without reciprocal terms.

### 4.2 Restricted Providers (OASIS-3, AIBL, MIRIAD)
- **OASIS-3:** 1,098 participants with extensive longitudinal multimodal imaging. Access requires a signed institutional DUA through the OASIS Brains portal with a 2–4 week turnaround. Immediate access before the October 22 deadline was legally impossible.
- **AIBL:** Access requires formal approval from the AIBL management committee and the ADNI Data Sharing and Access Committee (DSAC).
- **MIRIAD:** High-quality longitudinal dataset of 69 subjects from UCL. Preprocessing pipelines have been developed (`scripts/run_miriad_external_evaluation.py`) for post-deadline expansion, but MIRIAD's small sample size ($N=69$) precludes it from serving as a primary external benchmark in place of ADNI ($N=500$).

---

## 5. Dataset Audit Recommendations

1. **Maintain ADNI as the Gold-Standard External Cohort:** ADNI provides multi-manufacturer diversity (Siemens, GE, Philips), dual field strengths (1.5T and 3.0T), multi-site geographical distribution across dozens of medical centers, and fully disjoint participant enrollment.
2. **Standardize Reporting of Secondary Cohorts:** Always display OASIS-2 results with the explicit caveat: *(Exploratory Same-Center / Unverified Overlap)* to preserve scientific integrity.
3. **Comply with PROBAST+AI:** Document data sources, inclusion criteria, exclusion filters, and exact participant counts in all benchmark comparison tables.
