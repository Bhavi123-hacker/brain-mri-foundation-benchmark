# SHIFT-MRI: Current (V3 Final) vs. Historical Results Audit

**Purpose:** Provide complete, transparent provenance separating historical pilot / pre-audit artifacts from the definitive, audited V3 research deliverables.

---

## 1. Summary of Major Methodological and Numerical Evolutions

| Research Item | Historical Pilot / Pre-Audit Status | Authoritative V3 Final Status | Rationale for Change |
| :--- | :--- | :--- | :--- |
| **Decision Threshold Selection ($t^*$)** | Derived post-hoc using OASIS-1 test labels ($N=30$) | Derived strictly from authentic OASIS-1 validation predictions (`v3_oasis_validation_predictions.csv`, $N=30$) | Eliminates test-label leakage; enforces prospective source-locked evaluation. |
| **ADNI Demographic Baselines (Age + Sex)** | Evaluated on pilot manifest ($N=50$), reporting $\text{AUC} = 0.5671$ | Evaluated on complete matched 500-subject ADNI cohort, reporting $\text{AUC} = 0.5123$ (Age alone: 0.5242) | Resolves small-sample pilot artifact; proves deep vision models (0.5834, 0.5778) outperform demographic priors. |
| **TrustGate In-Domain Validation Construction** | Combined validation feature distances with test-derived margin, disagreement, and QC scores | Reconstructed validation TrustGate composite strictly using validation inputs ($S_A^{\text{val}}, S_B^{\text{val}}, S_C^{\text{val}}, S_D^{\text{val}}$) | Enforces strict split isolation; $\tau^* = 0.6340$ derived without test data. |
| **Feature Distance (Component B) Saturation** | Clipped feature distance reported as AUC = 0.5000 with no diagnostic explanation | Diagnosed that 100% of ADNI samples exceed source $q_{95}$, saturating clipped distance to 1.0000; raw Mahalanobis reported (AUC = 0.4808) | Provides scientific explanation of zero-variance clipping under severe domain shift. |
| **Risk-Coverage Metrics Labeling** | Trapezoidal area over $[0.50, 1.0]$ reported as "AURC" | Explicitly labeled as Partial AURC ($\text{pAURC}_{[0.5, 1.0]}$) alongside complete Full AURC over $[0.0, 1.0]$ | Mathematical precision complying with selective classification literature. |
| **Out-of-Distribution (OOD) Evaluation** | Post-hoc label-dependent direction flipping (`if auc < 0.5: 1 - auc`) | Strictly prospective score direction (Mahalanobis AUC = 1.0000, Centroid L2 = 0.9930, Collapse = 1.0000) | Eliminates target-label knowledge in OOD evaluation. |
| **In-Domain Disagreement AUC** | Historical pre-audit script reported 0.6112 | Audited validation-aligned test disagreement is 0.5951 | Reconciled with authoritative CSV. |
| **OASIS-2 Cohort Status** | Described in earlier V2 notes as "confirmed independent external hospital" | Formally classified as an *Exploratory Secondary Cohort with Unverified Participant Overlap* | Acknowledges shared Washington University ADRC origin. |

---

## 2. Preservation of Historical Artifacts

To maintain full scientific auditability and reproducibility:
- All pre-audit V3 results and figures have been preserved in:  
  `results/v3_pre_audit_historical/`
- All original V2 master tables are preserved in:  
  `results/v2/tables/`
- All initial V1 locked test predictions are preserved in:  
  `results/report/`

The primary, default deliverables in `results/v3/`, `docs/submission/final/`, and the web application reflect strictly the **Authoritative V3 Final** numbers.
