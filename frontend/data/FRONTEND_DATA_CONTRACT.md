# Frontend Data Contract & Interface Specification

**Directory**: `results/final_campaign/canonical_frontend_data/`  
**Status**: FROZEN & CANONICAL  
**Readiness**: READY FOR FRONTEND CONSUMPTION  

---

## 1. Terminology and Scientific Communication Rules

All visual and textual UI components MUST strictly adhere to these vocabulary rules:

| Compliant Phrasing (USE) | Prohibited Phrasing (DO NOT USE) | Reason |
|---|---|---|
| **"Model-predicted probability"** | "Diagnosis" / "Diagnoses Alzheimer's" | System is a research benchmark, not a diagnostic medical device. |
| **"Prediction-associated regions"** / **"Model saliency"** | "Disease-generating regions" / "Proof of pathology" | Grad-CAM reflects statistical network attribution, not causal biology. |
| **"External transfer"** | "Clinically validated generalization" | Evaluated on an external research cohort ($N=8$ test scans). |
| **"Locked holdout test"** | "Final clinical benchmark" | Strictly controlled mathematical holdout partition ($N=8$). |
| **"Development cross-validation"** | "Universal training accuracy" | Evaluated across 5 folds on $N=42$ development scans. |
| **"Pilot evidence"** / **"Exploratory benchmark"** | "Definitive proof" / "Proven superiority" | Acknowledges statistical power bounds of small cohorts. |

---

## 2. Canonical JSON Schema Registry

1. **`study_metadata.json`**: Core research parameters, sample sizes, class definitions, and regulatory disclaimer.
2. **`models.json`**: Audited array of candidate models with parameter counts, paradigms, and frozen metrics.
3. **`leaderboard.json`**: Multi-tiered leaderboard cleanly distinguishing Development CV, Locked OASIS Holdout, and External ADNI Transfer.
4. **`data_efficiency.json`**: Sample scaling trajectories across 10%, 25%, 50%, and 100% development fractions.
5. **`label_efficiency.json`**: Absolute subject-count scaling ($N=4, 9, 18, 35$) and scaling category flags.
6. **`ssap.json`**: Seed-Stability Adjusted Performance scores across $\lambda \in \{0.25, 0.50, 1.00\}$.
7. **`parameter_efficiency.json`**: Total vs trainable parameter counts, Pareto optimality flags, and capacity ratios.
8. **`compute_efficiency.json`**: Wall-clock training times and inference latency per sample.
9. **`calibration.json`**: Raw vs temperature-scaled Brier scores, ECE, MCE, and fitting data records.
10. **`domain_shift.json`**: Cross-scanner transfer performance comparing 1.5T OASIS to 3.0T ADNI.
11. **`fusion.json`**: Hybrid foundation + CNN concatenation specifications and gain metrics.
12. **`representation_analysis.json`**: Latent feature dimensionality, PCA variance ratios, and silhouette scores.
13. **`error_analysis.json`**: Subject-level false positive and false negative anatomical failure audits.
14. **`dataset_characteristics.json`**: Cohort-level demographic and acquisition comparisons.
15. **`statistics.json`**: Multi-seed RNG registries, bootstrap parameters, and subgroup $p$-values.
16. **`interpretability_index.json`**: Index linking every Grad-CAM case to subject IDs, slice views, and composite images.
17. **`methodology.json`**: Machine-readable summary of end-to-end experimental protocols.
18. **`limitations.json`**: 12 mandatory scientific caveats and boundary conditions.
19. **`audit_status.json`**: Formal verification flags and readiness declaration (`READY`).

---

## 3. Image and Media Path Contract
- All image links must use portable relative paths starting from project root:
  - Figures: `results/final_campaign/figures/<filename>.png`
  - Dataset Figures: `results/final_campaign/figures/dataset/<filename>.png`
  - Grad-CAM Panels: `results/final_campaign/interpretability/<filename>.png`
- Absolute Windows paths (e.g. `D:\...`) must NEVER be embedded in user-facing UI elements.
