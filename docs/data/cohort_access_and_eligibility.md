# Cohort Access, Licensing & Eligibility Protocols
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Date**: October 21, 2026  
**Auditor**: Data Engineering and Reproducibility Lead  

---

## 1. Candidate Cohort Access Matrix & Licensing Terms

| Cohort Identifier | Host Institution | Access Portal URL | Applicable Data Use Agreement (DUA) | Redistribution Policy | Credentialing Turnaround | Current Local Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **OASIS-1** | Washington University ADRC | `https://www.oasis-brains.org/` | Creative Commons Attribution 4.0 (CC-BY 4.0) | Permitted with attribution | Open Instant Access | **FULLY INTEGRATED** ($N=198$) |
| **ADNI** | North American ADNI Consortium (LONI) | `https://adni.loni.usc.edu/` | ADNI Data Use Agreement | Strictly prohibited | Institutional DUA Approved | **FULLY INTEGRATED** ($N=500$) |
| **MIRIAD** | Dementia Research Centre, UCL | `https://www.ucl.ac.uk/drc/research/methods/miriad-scan-database` | MIRIAD Data Use Agreement | Strictly prohibited (No 3rd party redistribution) | UCL XNAT Registration & Manual Review | **ACCESS_GATED (Pipeline Ready)** |
| **OASIS-2** | Washington University ADRC | `https://www.oasis-brains.org/` | Creative Commons Attribution 4.0 (CC-BY 4.0) | Permitted with attribution | Open Instant Access | **METADATA_ONLY (Audit Required)** |
| **OASIS-3** | Washington University ADRC (XNAT) | `https://www.oasis-brains.org/#data` | OASIS-3 Data Transfer Agreement | Prohibited | 1-2 weeks (XNAT Central) | **DEFERRED_GATED** |
| **AIBL** | CSIRO / Univ. of Melbourne (LONI) | `https://aibl.csiro.au/` | LONI IDA AIBL DUA | Strictly prohibited | 2-3 weeks (Committee Review) | **DEFERRED_GATED** |

---

## 2. Deep Dive: MIRIAD Access Conditions & Required Human Registration

1. **Host Institution**: Dementia Research Centre, Institute of Neurology, University College London, UK.
2. **Access Route**: Database hosted on UCL XNAT portal (`http://miriad.drc.ion.ucl.ac.uk/`).
3. **Legal Compliance**:
   - Redistribution of raw or processed MIRIAD neuroimaging data to third parties or public repositories is **strictly forbidden**.
   - Automated downloading or scraping without explicit account authorization violates the UCL Data Use Agreement.
   - All publications and presentations utilizing MIRIAD data must cite the primary release paper:
     *Malone IB, Cash D, Ridgway GR, Macmanus DG, Ourselin S, Fox NC, Schott JM. MIRIAD-Public release of a multiple time point Alzheimer's MR imaging dataset. NeuroImage. 2012 Dec 28;70C:33-36. doi:10.1016/j.neuroimage.2012.12.044.*
4. **Actionable Step for User**:
   - The user must register an account on the UCL MIRIAD portal.
   - Upon administrator activation, download the 69 baseline NIfTI volumes into `D:\brain_mri_data\MIRIAD\raw\`.
   - The prepared benchmarking pipeline ([`scripts/run_miriad_external_evaluation.py`](file:///D:/ml%20project/scripts/run_miriad_external_evaluation.py)) will automatically execute preprocessing, feature extraction, and multi-model evaluation.

---

## 3. Explicit Research Tracks Design

To maintain strict experimental hygiene and avoid conflating zero-shot evaluation with domain adaptation, the benchmark is organized into two explicitly separate research tracks:

### Track A: Source-Only External Transfer (Zero-Shot Generalization)
- **Training Domain**: Trained and selected strictly on the source development cohort (OASIS-1: 138 train, 30 val).
- **Target Domain**: Deployed zero-shot on external cohorts (ADNI: 500 subjects; MIRIAD: 69 subjects).
- **Hyperparameter & Threshold Rules**:
  - Zero target-domain labels are observed during training, feature scaling, model selection, or threshold tuning.
  - Decision threshold is fixed strictly on source validation data ($t=0.50$ or source Youden index).
  - Target-domain metrics measure pure out-of-domain robustness under realistic scanner shifts.

### Track B: Multi-Cohort Learning (Domain Generalization via Multi-Site Training)
- **Training Domain**: Pooled training using patient-disjoint partitions from multiple acquisition sites:
  - OASIS-1 Train ($N=138$, 1.5T Siemens) + ADNI Development Train ($N=250$, multi-vendor Siemens/Philips 1.5T/3.0T).
- **Validation Domain**: Pooled multi-site validation (OASIS-1 Val $N=30$ + ADNI Val $N=50$).
- **Held-Out External Test**: Evaluated on completely unseen clinical sites or unseen cohorts (ADNI Held-Out Test $N=200$ and/or MIRIAD $N=69$ on GE scanner).
- **Scientific Value**: Tests whether exposing convolutional and foundation models to diverse scanner physics during training eliminates the cross-site calibration drift observed in Track A.

---
*Protocols established and documented on October 21, 2026.*
