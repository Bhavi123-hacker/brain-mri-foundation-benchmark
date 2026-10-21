# SHIFT-MRI TrustGate: Comprehensive Prior Art Comparison and Methodological Context

**Component:** SHIFT-MRI TrustGate Failure Risk Estimation Framework  
**Scope:** Literature Survey, Mathematical Formulations, and Prior Art Mapping  
**Benchmark Version:** V3.0  
**Date:** October 2026  

---

## 1. Motivation and Problem Formulation

When a medical imaging AI model is trained on a source cohort (e.g., single-site 1.5T OASIS-1) and deployed to an external medical center (e.g., multi-site ADNI), ground-truth clinical diagnostic labels are unavailable at inference time. Clinicians cannot know whether a model's prediction on an individual patient is accurate or erroneous.

The goal of **failure risk estimation** (also termed error detection or selective prediction) is to construct a scoring function $R(x) \in \mathbb{R}^+$ based entirely on information accessible at test time without true labels $y$, such that:
$$R(x) > \tau \implies \mathbb{P}(\hat{y}(x) \neq y) \gg \mathbb{P}_{\text{marginal}}(\hat{y} \neq y)$$
where scans with $R(x) > \tau$ are flagged as potentially unreliable and routed for manual neuroradiological review.

---

## 2. Thorough Survey of Existing Prior Art (2017–2026)

### 2.1 Selective Classification and Failure Detection in Medical Imaging
- **Geifman & El-Yaniv (NeurIPS 2017, ICML 2019: *SelectiveNet*):**  
  Introduced formal selective classification using the Softmax Response ($SR(x) = \max_k p_k(x)$) and margin. Demonstrated that under independent and identically distributed (i.i.d.) conditions, thresholding maximum probability monotonically decreases risk.  
  *Limitation under medical distribution shift:* Modern neural networks are notoriously overconfident out-of-distribution (Guo et al., 2017). On external ADNI, models output $p \approx 0.99$ for false positives, rendering raw Softmax Response uninformative for error detection.
- **Kompa, Snoek, & Beam (Nature Communications 2021: *"Second opinion strategies in medical AI"*):**  
  Evaluated abstention strategies across diabetic retinopathy and cardiology datasets. Found that abstaining based on entropy or epistemic uncertainty substantially improves clinical safety, but noted severe performance drops under domain shifts.
- **Healthcare Failure Detection Benchmarks (Ding et al., 2023; Roy et al., 2022):**  
  Benchmarked failure detection algorithms across clinical EHR and radiology data, establishing that multi-modal uncertainty signals outperform single-metric epistemic indicators.

### 2.2 Label-Free Foundation-Model Selection and Pseudo-Label Agreement
- **MICCAI 2026: Label-Free Foundation Model Selection via Pseudo-Label Discrepancy:**  
  Recent work in medical foundation models proposed selecting optimal downstream backbones on unlabeled target datasets by measuring cross-model prediction consensus and pseudo-label agreement. If multiple structurally diverse models (e.g., 3D CNN, 2D Transfer, and Foundation SSL) converge on the same prediction for an external scan, prediction reliability is substantially higher.
- **BrainFound (BioRxiv / MICCAI 2026):**  
  A neuroimaging foundation model benchmark that explored representation consistency across multiple downstream tasks, noting that representations that drift far from the source pretraining distribution exhibit elevated error rates.
- **Mandoline & Reverse Testing (Chen et al., 2021; Garg et al., 2021):**  
  Proposed estimating test error without labels using model agreement across validation slices and importance-weighted consistency.

### 2.3 Feature-Space Out-of-Distribution (OOD) Detection
- **Lee et al. (NeurIPS 2018: *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*):**  
  Demonstrated that fitting empirical Gaussian distributions to penultimate layer activations and computing the Mahalanobis distance provides powerful OOD detection:
  $$D_{\text{Mahalanobis}}(z) = \min_c (z - \mu_c)^T \Sigma^{-1} (z - \mu_c)$$
- **Regularized Feature Distances in High Dimensions (Ledoit & Wolf, 2004):**  
  When feature dimension $D$ is comparable to training sample size $N$ (e.g., FOMO25 $D=320$ vs OASIS train $N=138$), empirical sample covariance $\hat{\Sigma}$ is rank-deficient and ill-conditioned. Regularization via Ledoit-Wolf shrinkage or spherical regularization ($\Sigma_{\text{reg}} = \hat{\Sigma} + \lambda I$) is mandatory to prevent numerical instability.

### 2.4 Conformal Prediction and Its Distribution Shift Limitations
- **Split Conformal Prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021):**  
  Provides rigorous finite-sample coverage guarantees ($1 - \alpha$) for prediction sets under the assumption of **exchangeability**.
- **Conformal Prediction under Covariate Shift (Tibshirani, Candès, et al., 2019):**  
  Extended conformal guarantees under covariate shift $p_{\text{target}}(x) = w(x) p_{\text{source}}(x)$ by reweighting non-conformity scores using likelihood ratios $w(x)$.
- **Critical Limitation in High-Dimensional Neuroimaging:**  
  In 3D brain MRI, estimating the exact density ratio $w(x)$ over millions of voxels across multi-site scanners is intractable. Under unmodeled acquisition shifts, finite-sample coverage guarantees break down. Therefore, **we explicitly avoid claiming guaranteed conformal coverage** and instead treat failure estimation as an empirical risk mitigation framework.

### 2.5 Confidence Aggregation and Ensemble Disagreement
- **Lakshminarayanan, Pritzel, & Blundell (NeurIPS 2017: *Deep Ensembles*):**  
  Demonstrated that prediction disagreement among multiple independently trained models provides calibrated estimates of epistemic uncertainty.
- **Mukhoti et al. (2023):**  
  Showed that disagreement across heterogeneous model families (e.g., ViT vs CNN) captures architectural inductive bias differences, outperforming homogeneous deep ensembles.

---

## 3. Systematic Prior Art Comparison Matrix

The table below contrasts existing paradigms with the proposed TrustGate composite framework:

| Method / Paradigm | Core Signal Source | Requires Target Labels? | Robust to Target Logit Shift? | Sensitive to Scanner QC Shifts? | Uses Foundation Features? | Conformal Guarantees Under Arbitrary Shift? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Maximum Softmax Probability (MSP)** (Geifman 2017) | Output probability $\max_k p_k$ | No | **No** (Severe Overconfidence) | No | No | No |
| **Predictive Entropy** (Shannon 1948) | Output distribution dispersion $H(p)$ | No | **No** (Shifted distributions) | No | No | No |
| **Feature Mahalanobis OOD** (Lee et al. 2018) | Penultimate representation distance | No | Yes (Operates in feature space) | Partially | Optional | No |
| **Homogeneous Ensemble Disagreement** (Lakshminarayanan 2017) | Variance among identical CNN seeds | No | Partially | No | No | No |
| **Label-Free Pseudo-Label Discrepancy** (MICCAI 2026) | Multi-model prediction agreement | No | Yes | No | Yes | No |
| **Conformal Prediction w/ Likelihood Weighting** (Tibshirani 2019)| Non-conformity quantiles | No | Only if $w(x)$ is exact | No | No | **Guaranteed only if $w(x)$ known** |
| **SHIFT-MRI TrustGate (Proposed Prototype)** | **Composite:** Margin + Reg. Feature Distance + Heterogeneous Disagreement + QC Anomaly | **No (Locked to Source)** | **Yes (Multi-Signal Composite)** | **Yes (Foreground Intensity Anomaly)** | **Yes (FOMO25 Latents)** | **No (Transparent Empirical Prototype)** |

---

## 4. Scientific Humility and Novelty Boundary

We explicitly establish that **TrustGate is not a brand-new fundamental mathematical discovery**, but rather a principled, source-locked composite architecture that synergistically combines four established statistical signals:
1. Classification Margin (Decision boundary proximity).
2. Regularized Feature Distance (Latent foundation manifold drift).
3. Heterogeneous Multi-Model Disagreement (Consensus discrepancy across 3D CNN, 2D Transfer, and ViT).
4. Morphometric / Intensity QC Anomaly (Scanner acquisition outlier detection).

Our contribution is:
- Formulating a source-locked combination protocol that prevents target-data leakage.
- Conducting rigorous empirical ablations comparing the composite against every individual baseline.
- Honestly reporting whether the combination yields statistically defensible error detection under real-world clinical domain shifts.
