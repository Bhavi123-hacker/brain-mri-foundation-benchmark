# SHIFT-MRI: Future Research Program & Methodological Roadmap

**Project:** SHIFT-MRI Benchmark Platform  
**Document:** Future Research Program (Tracks A through I)  
**Author:** Bhavy Garg, VIT Chennai  
**Date:** October 2026  
**Status:** Canonical Living Document  

---

## 1. Overview and Scientific Motivation

The SHIFT-MRI benchmark has systematically exposed critical vulnerabilities in brain MRI foundation models and supervised deep learning baselines when evaluated across independent acquisition sites (OASIS-1 $\to$ ADNI $N=500$). 

Most notably, **SHIFT-MRI TrustGate** produced an empirical **negative result**: an equal-weighted composite risk estimator combining predictive entropy, feature distance, ensemble disagreement, and quality anomalies achieved an $\text{AUROC} = 0.4697$—performing *worse* than random guessing and severely underperforming simple classification margin ($\text{AUROC} = 0.5829$).

This roadmap establishes nine prioritized, scientifically rigorous research tracks designed to address the foundational limitations uncovered in V3.

---

## 2. Prioritized Research Tracks

### Track A — SHIFT-MRI TrustGate V2: Non-Linear & Shift-Robust Abstention
- **Scientific Question:** Why does multi-model consensus invert under severe domain shift, and can selective prediction be rehabilitated without external target labels?
- **Hypothesis:** Under severe scanner shift, all six ensemble architectures undergo correlated prediction collapse toward the majority class, causing ensemble disagreement ($S_C$) to penalize correct minority predictions ($\text{AUROC} = 0.4674$). Furthermore, high-dimensional latent distances ($S_B$) saturate ($100\%$ saturation in ADNI).
- **Proposed Protocol:**
  1. Replace linear equal weighting ($0.25 S_A + 0.25 S_B + 0.25 S_C + 0.25 S_D$) with rank-based aggregation and Pareto-optimal abstention.
  2. Implement regularized minimum covariance determinant (FastMCD) and robust Mahalanobis distances with dimensionality reduction (PCA whitening / UMAP projection).
  3. Replace simple ensemble variance with diversity-weighted disagreement that accounts for inter-architecture prediction correlation.
- **Data Requirements:** OASIS-1 source validation set ($N=30$) for parameter fitting; unlabeled target cohorts for validation.
- **Verification Criteria:** TrustGate V2 must achieve $\text{AUROC} \ge 0.60$ and $\text{pAURC}_{[0.5, 1.0]} < 0.25$ without using any target ground-truth labels.
- **Maturity:** Implementation-ready (framework established in `experiments/experiment_lab.py`).

---

### Track B — Scanner and Preprocessing Distribution Shift
- **Scientific Question:** How much of the cross-cohort generalization drop is attributable to scanner manufacturer/field strength vs anatomical differences?
- **Hypothesis:** Scanner field strength (1.5T vs 3.0T) and manufacturer point spread functions (Siemens vs Philips) account for significant shortcut feature extraction.
- **Proposed Protocol:**
  1. Stratify ADNI ($N=500$) into pure scanner subsets: Siemens 1.5T, Siemens 3.0T, and Philips 3.0T.
  2. Measure performance deltas under matched demographic groups (age- and sex-matched sub-cohorts).
  3. Evaluate controlled harmonization algorithms (e.g. ComBat neuroimaging harmonization) vs raw foreground Z-scoring.
- **Confounding Factors:** Participant age and disease severity differences across scanner sites.
- **Verification Criteria:** Demonstrate statistically significant performance variance across scanner subsets using permutation tests.
- **Maturity:** Partially executed (V3 scanner stratification baseline committed).

---

### Track C — Foundation Model Adaptation: Linear Probes vs Low-Rank Adaptation (LoRA)
- **Scientific Question:** Does parameter-efficient fine-tuning (PEFT) enable foundation models to overcome domain collapse without catastrophic forgetting?
- **Hypothesis:** Frozen linear probing on self-supervised ViT representations (e.g. MONAI Swin UNETR) restricts adaptation, whereas low-rank adaptation (LoRA, $r=8$) of attention projection matrices improves out-of-distribution transfer under matched labeled budgets.
- **Proposed Protocol:**
  1. Compare three regimes: (a) Frozen linear probe, (b) Attention-layer LoRA ($r=4, 8, 16$), (c) Full fine-tuning.
  2. Evaluate on sample-budget curves ($N \in \{20, 50, 100, 138\}$).
- **Baselines:** 3D ResNet-10 trained from scratch.
- **Verification Criteria:** LoRA must achieve superior external AUC on ADNI while utilizing $< 5\%$ of trainable parameters.
- **Maturity:** Proposed study.

---

### Track D — Multi-Seed Uncertainty & Split Stability
- **Scientific Question:** How sensitive are small-sample neuroimaging benchmarks to random initialization and train/val partition choices?
- **Hypothesis:** Supervised 3D CNNs with high parameter-to-sample ratios (e.g. 3D ResNet-34, 63.5M parameters on $N=138$) exhibit high test AUC variance ($\sigma > 0.05$) across random seeds.
- **Proposed Protocol:**
  1. Run 5-fold cross-validation repeated across 5 independent random seeds ($K=25$ runs) for all 10 architectures.
  2. Report inter-split and intra-split confidence intervals and Delong tests.
- **Maturity:** Implementation-ready (supported by `experiments/experiment_lab.py`).

---

### Track E — Independent External Cohort Ingestion (AIBL / MIRIAD)
- **Scientific Question:** Do foundation models generalize to completely independent international cohorts with certified zero subject overlap?
- **Hypothesis:** Models demonstrating above-chance AUC on ADNI will maintain consistent ranking on Australian Imaging, Biomarker & Lifestyle Flagship Study of Ageing (AIBL).
- **Proposed Protocol:**
  1. Formal DUA acquisition for AIBL and MIRIAD cohorts.
  2. Preprocessing standardization through the verified isotropic 128^3 pipeline.
  3. Evaluation using locked validation thresholds derived from OASIS-1.
- **Governance:** Full ethical DUA compliance prior to tensor generation.
- **Maturity:** Proposed study (adapter contract defined in `adapters/dataset_adapter.py`).

---

### Track F — Conformal Prediction & Risk-Coverage Optimization
- **Scientific Question:** Can split-conformal prediction guarantee coverage under covariate and label shift in 3D brain MRI?
- **Hypothesis:** Standard split-conformal prediction guarantees marginal coverage on source data, but coverage guarantees violate target bounds under domain shift unless weighted by likelihood ratios.
- **Proposed Protocol:**
  1. Compute non-conformity scores on OASIS-1 validation set.
  2. Implement Covariate-Shift-Adapted Conformal Prediction using density ratio estimation.
  3. Plot selective classification risk-coverage curves ($AURC$).
- **Maturity:** Proposed study.

---

### Track G — Synthetic MRI Perturbation Robustness Auditing
- **Scientific Question:** Which architectures are most resilient to clinical acquisition artifacts (motion, Rician noise, intensity non-uniformity)?
- **Hypothesis:** 2.5D multi-view models and foundation models exhibit higher tolerance to slice-direction motion artifacts than fully 3D convolutional models.
- **Proposed Protocol:**
  1. Implement parameterized perturbations: Rician noise ($\sigma \in [0.01, 0.10]$), Gaussian blur ($\text{FWHM} \in [1, 3]\text{ mm}$), and k-space motion artifact simulation.
  2. Measure degradation curves across increasing perturbation severity.
- **Maturity:** Implementation-ready.

---

### Track H — Clinical-Label Sensitivity & Diagnostic Borderlines
- **Scientific Question:** How does diagnostic grouping (CDR 0.5 very mild vs CDR 1.0 moderate, and ADNI Mild Cognitive Impairment) affect model confidence?
- **Hypothesis:** Model misclassifications correlate strongly with borderline clinical dementia ratings (CDR 0.5) rather than scanner artifacts.
- **Proposed Protocol:**
  1. Subdivide positive classes by CDR impairment severity.
  2. Correlate predicted probabilities with Mini-Mental State Examination (MMSE) scores.
- **Maturity:** Proposed study.

---

### Track I — Longitudinal Disease Progression & Atrophy Velocity
- **Scientific Question:** Can baseline 3D MRI feature representations predict longitudinal progression rate over 24-36 months?
- **Hypothesis:** Foundation model embeddings capture structural atrophy trajectories superior to cross-sectional volumetric biomarkers.
- **Proposed Protocol:**
  1. Ingest multi-session OASIS-2 and ADNI longitudinal series.
  2. Implement participant-disjoint longitudinal progression prediction.
- **Maturity:** Proposed study.
