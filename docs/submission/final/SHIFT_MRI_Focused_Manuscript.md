# SHIFT-MRI: Source-Locked, Label-Aware Cross-Cohort Reliability and Failure Analysis of Brain MRI Models

**Author:** Bhavy Garg  
**Affiliation:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Chennai, Tamil Nadu, India  
**Date:** October 2026  
**Software Repository:** [brain-mri-foundation-benchmark](https://github.com/Bhavi123-hacker/brain-mri-foundation-benchmark) (Branch: `research/v3-final-ui-paper-submission`)  
**Audited Commit:** `934c86c`  

---

## Abstract

Pretrained self-supervised vision foundation models are widely hypothesized to confer robust generalization in neuroimaging. However, clinical validation under real-world multi-center acquisition shifts remains scarce. Here we present **SHIFT-MRI**, a source-locked, label-aware benchmarking framework evaluating ten diverse deep learning architectures trained on the in-domain OASIS-1 cohort ($N=198$) and evaluated zero-shot on the multi-center ADNI cohort ($N=500$). 

Our benchmark reveals a pervasive **Foundation Generalization Deficit**: while the self-supervised MONAI Swin UNETR SSL leads cross-cohort discrimination ($\text{ROC-AUC} = 0.5834$, $95\%\text{ CI: } [0.5331, 0.6337]$), its advantage over a 2D slice-aggregated ResNet-18 ($\text{AUC} = 0.5778$) and a lightweight 3D ResNet-10 ($\text{AUC} = 0.5444$) is modest. Crucially, all vision models exhibit performance within the margin of a simple demographic baseline (Age alone: $\text{AUC} = 0.5242$). 

Furthermore, we investigate **SHIFT-MRI TrustGate**, a source-calibrated composite selective prediction system combining predictive entropy, latent Mahalanobis distance, multi-architecture disagreement, and quality anomalies. In a rigorous prospective test, TrustGate produced an empirical **negative result**: the composite risk estimator yielded $\text{AUROC} = 0.4697$, performing *worse* than random guessing and lagging far behind simple classification margin ($\text{AUROC} = 0.5829$). We demonstrate that multi-model consensus inverts under severe scanner shift, causing ensemble disagreement to penalize correct minority predictions. These findings demonstrate that unadapted brain MRI foundation models do not reliably transfer across clinical sites, and multi-component failure estimators must be domain-calibrated to avoid catastrophic selective prediction failure.

---

## 1. Introduction

Deep learning models for structural 3D brain magnetic resonance imaging (MRI) frequently report near-perfect discrimination ($>95\%$ ROC-AUC) on isolated, single-institution cohorts. However, clinical deployment requires generalization across different MRI scanner manufacturers, magnetic field strengths, and patient demographics.

Recent medical imaging research has promoted self-supervised foundation models (e.g. MONAI Swin UNETR SSL, trained on over 5,000 multi-site brain volumes) as a potential solution to domain shift. Yet rigorous, source-locked benchmarks directly comparing foundation models against well-tuned supervised 3D CNNs under matched preprocessing and unpolluted external cohorts remain rare.

In this work, we investigate two primary questions:
1. **The Foundation Generalization Deficit:** Do self-supervised foundation models generalize better to external multi-site MRI acquisitions than compact supervised 3D CNNs and hybrid anatomical models?
2. **Selective Failure Detection (TrustGate):** Can a model's prediction unreliability on an unlabeled target cohort be detected using a source-locked composite gate without access to target ground-truth labels?

---

## 2. Experimental Methodology

### 2.1 Cohorts and Governance
- **OASIS-1 (Primary In-Domain Benchmark):** $N=198$ participants (138 training, 30 validation, 30 locked test) acquired on a single Siemens Vision 1.5T scanner. Label: Clinical Dementia Rating (CDR 0 vs CDR $\ge 0.5$).
- **ADNI (Zero-Shot External Multi-Site Cohort):** $N=500$ participants ($303$ Cognitively Normal, $197$ Alzheimer's Disease) acquired across dozens of clinical sites on Siemens ($N=370$) and Philips ($N=130$) scanners (1.5T and 3.0T). Label: NINCDS-ADRDA diagnostic consensus.
- **OASIS-2 (Secondary Exploratory Cohort):** $N=150$ participants ($85$ CN, $65$ AD) on Siemens Vision 1.5T. Disclosed as an exploratory cohort due to unresolved participant overlap with OASIS-1.

### 2.2 Preprocessing Standardization
All scans underwent isotropic spatial resampling to $128 \times 128 \times 128$ ($1.5 \text{ mm}^3$ voxels) followed by non-zero foreground Z-score normalization. No non-linear atlas registration or unverified N4 bias correction was applied, preserving authentic scanner variance.

### 2.3 Evaluated Model Architectures (10 Paradigms)
1. **MONAI Swin UNETR SSL:** 3D Swin Transformer ViT backbone pretrained on 5,050 brain MRIs with masked autoencoding; frozen linear probe ($8.06\text{M}$ params).
2. **2D Transfer ResNet-18:** Slice-aggregated ImageNet pretrained ResNet-18 ($11.18\text{M}$ params).
3. **Dual-Branch Hybrid Model:** 3D ResNet-10 spatial CNN concatenated with 6 morphometric anatomical proxies via late fusion MLP ($14.53\text{M}$ params).
4. **3D ResNet-10 (Pipeline B):** Lightweight 3D residual network ($14.62\text{M}$ params).
5. **Classical Radiomics:** 25 morphological and intensity features classified by linear SVM.
6. **3D ResNet-18 (Scratch):** Full 3D convolutional baseline ($33.16\text{M}$ params).
7. **FOMO25 AMAES ResEnc-B:** Foundation model linear probe ($90.30\text{M}$ params).
8. **3D DenseNet-121:** Dense 3D feature reuse network ($11.77\text{M}$ params).
9. **2.5D Multi-View ResNet-18:** Triplanar orthogonal view transfer ($11.57\text{M}$ params).
10. **3D ResNet-34 (Scratch):** Overparameterized 3D baseline ($63.49\text{M}$ params).

### 2.4 SHIFT-MRI TrustGate Formulation
TrustGate computes a composite unreliability score $R(x)$ on an unlabeled target scan $x$ by combining four source-calibrated components:
$$R(x) = 0.25 S_A(x) + 0.25 S_B(x) + 0.25 S_C(x) + 0.25 S_D(x)$$
where:
- $S_A(x) = 1 - 2|p(x) - 0.5|$: Classification Margin Risk.
- $S_B(x) = \min(1.0, d_M(z_x) / q_{0.95})$: Clipped Mahalanobis Feature Distance to source representations.
- $S_C(x) = \text{Var}_{m \in \mathcal{M}}(p_m(x))$: Inter-Architecture Ensemble Disagreement.
- $S_D(x)$: MRI Intensity and Slice Quality Anomaly.

The rejection threshold $\tau^* = 0.6340$ is calibrated strictly as the 80th percentile of composite risk on the source OASIS-1 validation set (flagging the top 20% highest-risk scans on the source domain). In contrast, the 95th-percentile threshold ($q_{95} = 4.3905$) is used for standardization of the latent Mahalanobis feature distance ($S_B$). Under external ADNI domain shift ($N=500$), feature saturation pushes all composite risk scores above $\tau^* = 0.6340$ ($\min R(x) = 0.8033$), resulting in 100% of external scans (500/500) being flagged. Scans that pass or fail TrustGate are exploratory research flags and must not be treated as clinically reliable.

---

## 3. Results and Empirical Findings

### 3.1 Cross-Cohort Benchmark Leaderboard
Table 1 presents the prospective performance on external ADNI ($N=500$), demonstrating the cross-cohort generalization drop from in-domain OASIS-1 locked test:

| Rank | Model Architecture | Paradigm | Params | ADNI ROC-AUC [95% CI] | ADNI Brier | OASIS-1 Test AUC | Gen. Drop |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | MONAI Swin UNETR SSL | 3D Foundation Model | 8.06M | **0.5834** [0.5331, 0.6337] | 0.2569 | 0.6044 | -0.0210 |
| 2 | 2D Transfer ResNet-18 | 2D Pretrained Transfer | 11.18M | **0.5778** [0.5280, 0.6276] | 0.5315 | 0.6222 | -0.0444 |
| 3 | Dual-Branch Hybrid | Deep Spatial + Radiomics | 14.53M | **0.5554** [0.5044, 0.6066] | 0.3163 | 0.6311 | -0.0757 |
| 4 | 3D ResNet-10 | Supervised CNN (3D) | 14.62M | **0.5444** [0.4909, 0.5953] | 0.5724 | 0.6133 | -0.0689 |
| Ref | Demographic: Age Alone | Non-Imaging Baseline | 1 | 0.5242 [0.4727, 0.5753] | 0.2516 | 0.5000 | +0.0242 |
| 5 | Classical Radiomics | Engineered Morphometry | 25 | 0.5149 [0.4638, 0.5660] | 0.4491 | 0.5689 | -0.0540 |
| 6 | 3D ResNet-18 | Supervised Baseline | 33.16M | 0.5128 [0.4612, 0.5644] | 0.2738 | 0.6178 | -0.1050 |
| 7 | FOMO25 AMAES ResEnc-B | 3D Foundation Model | 90.30M | 0.5054 [0.4516, 0.5552] | 0.3442 | 0.5733 | -0.0679 |
| 8 | 3D DenseNet-121 | Supervised CNN (3D) | 11.77M | 0.5048 [0.4537, 0.5592] | 0.2603 | 0.4133 | +0.0915 |
| 9 | 2.5D Multi-View ResNet-18 | Triplanar Transfer | 11.57M | 0.4943 [0.4434, 0.5464] | 0.2654 | 0.6444 | -0.1501 |
| 10 | 3D ResNet-34 | Supervised Baseline | 63.49M | 0.4777 [0.4261, 0.5293] | 0.2534 | 0.5311 | -0.0534 |

### 3.2 Key Findings
1. **Foundation Model Margin:** Swin UNETR SSL leads the benchmark ($0.5834$), but its discrimination is only $0.0056$ higher than 2D ImageNet transfer ($0.5778$) and overlaps substantially with a non-imaging demographic predictor ($0.5242$).
2. **Severe Overparameterization Penalty:** 3D ResNet-34 ($63.5\text{M}$ parameters) degraded to sub-chance performance on ADNI ($0.4777$), demonstrating that excessive capacity on small medical cohorts leads to severe shortcut learning.

### 3.3 TrustGate Empirical Negative Result
Table 2 details the failure detection performance of TrustGate components on ADNI ($N=500$):

| Failure Risk Estimator | Error Detection AUROC [95% CI] | pAURC [0.5, 1.0] | Full AURC | Status / Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Classification Margin Risk ($S_A$)** | **0.5829** [0.5337, 0.6306] | **0.2871** | **0.5488** | **Top Error Detector** |
| Maximum Probability (MSP) Risk | 0.5829 [0.5337, 0.6306] | 0.2871 | 0.5488 | Rank-equivalent to Margin |
| Predictive Entropy | 0.5829 [0.5337, 0.6306] | 0.2871 | 0.5488 | Rank-equivalent to Margin |
| Raw Mahalanobis Feature Distance | 0.4808 [0.4288, 0.5332] | - | - | Sub-chance within-cohort error detection |
| Saturated Distance ($S_B$) | 0.5000 [0.5000, 0.5000] | - | - | 100% ADNI points saturated to 1.0 |
| Model Disagreement ($S_C$) | 0.4674 [0.4155, 0.5190] | 0.3086 | - | Inverted consensus under collapse |
| **Proposed TrustGate Composite** | **0.4697** [0.4172, 0.5221] | **0.3088** | **0.6470** | **Documented Negative Result** |

Under domain shift, all architectures collapse toward the majority class ($60.6\%$ CN in ADNI). Consequently, instances where models disagree actually correspond to rare correct predictions on AD patients, causing ensemble disagreement ($S_C$) to penalize correct predictions. Combined with distance saturation ($S_B$), the composite gate is dragged below random chance ($\text{AUROC} = 0.4697$).

---

## 4. Discussion and Limitations

1. **Diagnostic Label Mismatch:** OASIS-1 CDR impairment (functional status) is not clinically equivalent to ADNI's multi-modal clinical Alzheimer's disease diagnosis. Cross-cohort generalization reflects joint covariate and label shift.
2. **Scanner Confounding:** The ADNI sample comprises Siemens ($N=370$) and Philips ($N=130$) scanners, but no GE systems. Scanner-specific point spread functions and B1 field variations create systematic acquisition shortcuts.
3. **Negative Result Value:** Disclosing TrustGate's failure is essential for medical AI safety: complex multi-component abstention heuristics cannot be assumed to generalize across hospitals without domain calibration.

---

## 5. Conclusion

SHIFT-MRI provides an audited, source-locked evaluation demonstrating that 3D brain MRI foundation models do not automatically overcome cross-scanner distribution shift. Furthermore, multi-model failure estimation fails catastrophically under domain collapse. Future work must prioritize non-linear rank-based abstention and parameter-efficient domain adaptation.

---

## References

1. Hatamizadeh, A. et al. (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. *BrainLes / Springer LNCS*.
2. Marcus, D. S. et al. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults. *Journal of Cognitive Neuroscience*, 19(9), 1498–1507.
3. Jack, C. R. et al. (2008). The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *Journal of Magnetic Resonance Imaging*, 27(4), 685–691.
4. Hendrycks, D. & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. *ICLR*.
5. Garg, B. (2026). SHIFT-MRI: Source-Locked, Label-Aware Framework for Cross-Cohort Reliability of Brain MRI Models. *Antigravity Research Repository*.
