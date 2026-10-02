# Frontend Quality Assurance & Verification Report

**Project Title**: *Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines*  
**Date**: September 2026  
**Auditor**: Senior UI/UX Designer, Frontend Architect & Scientific QA Engineer  
**Frontend Application URL**: `http://127.0.0.1:8000` (via `python run_frontend.py`)  
**Frontend Directory**: `frontend/`  
**Single Source of Truth**: `results/final_campaign/canonical_frontend_data/`  

---

## 1. Executive Status

**FRONTEND STATUS**: **READY**

The research benchmark interactive web application has been fully implemented, verified against the frozen canonical research dataset, and validated across 53 automated tests and specialized terminology audits. Zero experimental numbers were modified or fabricated.

---

## 2. Architecture & Framework
- **Architecture**: Single-Page Application (SPA) utilizing semantic HTML5, modern CSS3, native ES modules, and a standalone vendored Chart.js distribution.
- **Offline Self-Containment**: 100% offline functionality with zero external CDN dependencies (`frontend/vendor/chart.min.js`, 205 KB local bundle).
- **Backend Serving Engine**: `run_frontend.py` provides high-performance static asset routing, canonical JSON mapping (`/data/...`), and results directory serving (`/results/...`), with an integrated automated `--test` smoke check.
- **Build Bottleneck Elimination**: Zero npm build or node_modules dependencies required at runtime; immediate launch via Python runtime.

---

## 3. Implemented Sections & Journey (20 Modules)

| # | Section ID | Title | Core Features & Canonical Data Sources |
|---|---|---|---|
| **01** | `section-overview` | Overview & Landing | 6 KPI cards strictly separating Dev CV, Locked Holdout, and External Transfer; Key Findings synthesis callout; multi-context model comparison bar chart. |
| **02** | `section-question` | Research Question | Verbatim central scientific hypothesis; interactive 5-step end-to-end benchmark workflow pipeline. |
| **03** | `section-datasets` | Datasets & Cohorts | Side-by-side profile cards for OASIS-1 (1.5T) and ADNI (3.0T); detailed characteristics comparison table (`dataset_characteristics.json`). |
| **04** | `section-models` | Evaluated Models | 7 model inspection cards with total vs trainable parameter ratios, training modes, and multi-context AUC scores (`models.json`). |
| **05** | `section-cnn-benchmark` | 3D CNN Benchmark | Dedicated comparative matrix for ResNet-18 vs ResNet-34 across OASIS-1 and ADNI; subject-level classification outcomes table. |
| **06** | `section-leaderboard` | Research Leaderboard | Multi-tiered sortable and filterable table with contextual ranking flags (`leaderboard.json`). |
| **07** | `section-data-efficiency` | Data Efficiency | Sample scaling curves (10% N=4, 25% N=9, 50% N=18, 100% N=35); exact hover tooltips with multi-seed std error bounds (`data_efficiency.json`). |
| **08** | `section-parameter-compute`| Parameter & Compute | Trainable parameters vs CV AUC Pareto frontier scatter plot; latency and training throughput table (`parameter_efficiency.json`, `compute_efficiency.json`). |
| **09** | `section-calibration` | Calibration & ECE | Reliability error comparison; post-hoc temperature scaling audit ($T=1.35-1.85$, dev validation only; `calibration.json`). |
| **10** | `section-domain-shift` | Domain Shift (ADNI) | Cross-scanner transfer performance from 1.5T to 3.0T; non-causal compound shift breakdown (`domain_shift.json`). |
| **11** | `section-representation` | Representation Geometry | PCA explained variance, raw vs projected silhouette scores, and latent dimensionality table (`representation_analysis.json`). |
| **12** | `section-fusion` | Feature Fusion | Architecture specification for 1,280-d concatenated representation (Swin + ResNet-18); development CV gain (+0.0800 AUC; `fusion.json`). |
| **13** | `section-error-analysis` | Error Analysis | Filterable failure audit table by cohort (OASIS/ADNI) and outcome (FP/FN/TP/TN); anatomical mechanisms (`error_analysis.json`). |
| **14** | `section-interpretability`| CNN Grad-CAM | Multi-view orthogonal slice inspector (Axial, Coronal, Sagittal; Original, CAM, Overlay, 3×3 Composite); zoom modal; spatial consistency metrics (`interpretability_index.json`). |
| **15** | `section-statistics` | Statistics & Uncertainty | 1000-resample bootstrap 95% confidence intervals; SSAP stability bar chart; $N=8$ discrete resolution caveat ($1 \text{ subject} = 12.5\%$). |
| **16** | `section-methodology` | Methodology | Complete audited experimental protocol covering curation, preprocessing, splits, and training (`methodology.json`). |
| **17** | `section-reproducibility` | Reproducibility | Environment parameters (Python 3.12.4, PyTorch 2.5.1), 53/53 passing tests badge, cryptographic SHA-256 artifact verification. |
| **18** | `section-limitations` | Research Limitations | 12 mandatory scientific caveats prominently formatted as numbered cards (`limitations.json`). |
| **19** | `section-conclusion` | Final Conclusion | Balanced thesis synthesis addressing in-domain performance, parameter efficiency, external transfer, and hybrid fusion. |
| **20** | `section-gallery` | Publication Gallery | Interactive gallery of all 28 publication-quality figures with category filtering, search, modal zoom, and asset download triggers. |

---

## 4. Specialized Interactive Components
1. **Presentation Mode**:
   - Toggled via the sidebar or header button (or keyboard shortcut `Ctrl+P`).
   - Maximizes viewport width, eliminates sidebar distraction, enlarges fonts and KPIs, and expands chart dimensions for viva/thesis screen demonstrations.
2. **Dedicated 3D CNN Benchmark Matrix**:
   - Allows instant dataset switching (OASIS-1 vs ADNI) and architecture selection (ResNet-18 vs ResNet-34).
   - Renders balanced accuracy, sensitivity, specificity, Brier scores, and subject-level classifications without page reloads.
3. **Grad-CAM Multi-Plane Inspector**:
   - Dynamically loads axial, coronal, and sagittal slice overlays and high-resolution 3×3 composites.
   - Click-to-zoom modal for inspecting fine periventricular and temporal lobe saliency contours.
4. **Figure Gallery**:
   - Real-time search and category filtering across Performance, Data Efficiency, Calibration, Representations, Domain Shift, and Ablations.

---

## 5. Terminology & Ethical Compliance Audit
- Executed `scripts/audit_frontend_terminology.py` scanning all HTML, CSS, and JS files.
- **Audit Result**: **100% PASS (Zero Violations)**.
- Strictly enforced vocabulary:
  - *"Model-predicted probability"* (prohibited: *"diagnostic probability"*).
  - *"Prediction-associated regions"* (prohibited: *"disease-generating regions"*).
  - *"Locked holdout"* and *"Development CV"* (never conflated).
  - Persistent non-diagnostic disclaimer in global footer and section headers.

---

## 6. Verification & Automated Test Suite

```
pytest tests/ -v
```

**Results**: **53 passed, 0 failed, 9 warnings in 53.12s**
- `tests/test_frontend_application.py`: 4/4 passing (Structure, Sections, Server Smoke Test, Terminology).
- `tests/test_canonical_frontend_data.py`: 6/6 passing (Canonical Schema, Zero NaN/Inf, Unique IDs, Ranks, Portable Paths, Ready Status).
- Model and pipeline tests: 43/43 passing (ResNet3D, Foundation Adapters, Dataloader, Caching, Splits, Evaluator, Diagnostics).

---

## 7. Known Boundaries & Limitations
1. **Pilot Sample Sizes**: All holdout test sets consist of $N=8$ scans per cohort; discrete accuracy steps equal 12.5%.
2. **Cross-Sectional Classification**: Does not evaluate longitudinal progression or prodromal MCI (CDR 0.5) staging.
3. **Compound Domain Shifts**: Cross-scanner performance differences encompass hardware, site protocols, and demographic factors.
4. **Local Asset Serving**: High-resolution Grad-CAM composites require local server execution (`python run_frontend.py`) for file URI resolution.

---

## 8. Final Verdict
The frontend research application is **READY** for local deployment, publication demonstration, and thesis evaluation.
