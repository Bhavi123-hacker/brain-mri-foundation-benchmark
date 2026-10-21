# SHIFT-MRI: Comprehensive Literature Review, Theoretical Foundations, and Scientific Novelty

**Project Title:** SHIFT-MRI: Systematic Benchmark and Cross-Cohort Reliability Audit of Brain-MRI Foundation Models versus Supervised Baselines  
**Lead Authors:** Deep Learning & Neuroimaging AI Research Team  
**Benchmark Version:** V3.0 (SHIFT-MRI Framework)  
**Date:** October 2026  

---

## 1. Executive Summary and Theoretical Grounding

Medical image analysis has recently experienced an influx of 3D foundation models—large-scale neural networks pretrained on tens or hundreds of thousands of unlabelled volumetric scans via self-supervised learning (SSL) or masked autoencoding (MAE). Prominent examples include **BrainIAC** (BioRxiv 2025; Nature Neuroscience 2026), **FOMO-25** (Foundation Models for MRI, CVPR/MICCAI 2026), and **MONAI Swin UNETR SSL** (Tang et al., 2022). These models promise universal anatomical representations, label-efficient transfer, and out-of-the-box domain generalization.

However, clinical translation requires answering a foundational methodological question:

> **Under matched label budgets, controlled preprocessing, and rigorous cross-cohort evaluation, do 3D brain-MRI foundation-model representations provide statistically verifiable advantages over well-tuned supervised deep learning baselines, engineered morphometric features, or simple demographic shortcuts?**

Recent methodological guidelines—including **CLAIM 2024** (Checklist for Artificial Intelligence in Medical Imaging), **TRIPOD+AI 2024** (Transparent Reporting of multivariable prediction models for Individual Prognosis Or Diagnosis), and **PROBAST+AI 2025** (Prediction model Risk Of Bias Assessment Tool)—mandate that medical AI benchmarks must evaluate external calibration, decision threshold stability, demographic confounding, and statistical multiplicity control.

The **SHIFT-MRI** framework introduces seven interconnected methodological novelties (N01–N07) to rigorously audit 10 diverse models across 3 distinct cohorts (OASIS-1, ADNI, and OASIS-2; total $N=848$ volumetric scans), uncovering systemic failure modes that are invisible under conventional single-metric in-domain reporting.

---

## 2. Deep Grounding in Priority Literature

### 2.1 Brain-MRI Foundation Models
1. **BrainIAC (Nature Neuroscience 2026 / BioRxiv 2025)**  
   *Pretrained on >70,000 heterogeneous 3D brain MRI scans across healthy lifespan and neurological cohorts.*  
   - *Architecture & Objective:* 3D ViT backbone trained via masked autoencoding and contrastive representation alignment. BrainIAC demonstrates superior downstream few-shot age regression and brain disease classification in homogeneous research settings.  
   - *Benchmark Role & Gap:* BrainIAC represents the frontier of large-scale SSL. However, its transfer representations have rarely been compared against well-regularized 2.5D multi-view ImageNet transfer or calibrated demographic baselines under unaligned scanner acquisitions.

2. **FOMO-25 AMAES (CVPR/MICCAI 2026: Foundation Models for MRI)**  
   *Pretrained on the FOMO-260k volumetric MRI collection (~260,000 brain and body scans).*  
   - *Architecture:* Asymmetric Masked Autoencoder for 3D Segmentation and Classification (AMAES) built on a 90,296,897-parameter Residual Encoder backbone (`resenc_unet_b`).  
   - *Findings in SHIFT-MRI:* When evaluated via a frozen linear probe on in-domain OASIS-1, FOMO25 achieves ROC-AUC = 0.5733 [95% CI: 0.3664, 0.7989]. When evaluated zero-shot on external ADNI ($N=500$), it achieves ROC-AUC = 0.5054 [95% CI: 0.4516, 0.5552] with a Brier score of 0.3442. This confirms that massive pretraining alone does not insulate linear probe decoders from feature drift caused by scanner acquisition variations.

3. **MONAI Swin UNETR SSL (Tang et al., CVPR 2022; Hatamizadeh et al., 2022)**  
   *Self-supervised 3D Swin Transformer (8,062,771 parameters) pretrained on large-scale public CT/MRI data with masked voxel inpainting, contrastive loss, and rotation prediction.*  
   - *Findings in SHIFT-MRI:* Achieves the highest raw discrimination on external ADNI among all deep models: **ROC-AUC = 0.5834** [95% CI: 0.5331, 0.6337]. However, calibration analysis reveals extreme overconfidence: at standard decision threshold $t=0.50$, predicted positive rate is **100.0%** (500/500 cases predicted AD), demonstrating complete decision collapse.

### 2.2 Supervised Deep Learning and Transfer Baselines
4. **MedicalNet (Chen et al., arXiv 2019)**  
   *3D ResNet architectures (ResNet-10, ResNet-18, ResNet-34, ResNet-50) pretrained across 8 diverse medical imaging datasets.*  
   - Provided foundational 3D convolutional baselines. In SHIFT-MRI, we demonstrate that a lightweight 3D ResNet-10 (14.6M parameters) matches or outperforms heavy 3D ResNet-34 (63.5M parameters) while avoiding severe overfitting under small-cohort training regimes.

5. **2.5D Multi-View Triplanar Transfer Learning (ImageNet Pretraining)**  
   *Decomposing 3D volumes into orthogonal axial, coronal, and sagittal slice projections processed by 2D pretrained backbones.*  
   - In SHIFT-MRI, 2.5D Multi-View ResNet-18 (11.6M parameters) achieves highest in-domain validation AUC (**0.8267**) and robust performance on OASIS-2 (**0.7500**). Because 2D natural image features encode low-level edge and texture primitives, they exhibit strong resistance to domain-specific volumetric artifacts.

6. **Engineered Morphometric Proxies and Dual-Branch Hybrid Architectures**  
   - Combining deep spatial representations with morphometric volume/intensity proxies (ventricular CSF expansion, medial temporal lobe / hippocampal volume ratio, and intensity skewness).  
   - Dual-Branch Hybrid (14.5M parameters) achieves **ROC-AUC = 0.6311** on OASIS-1 test and **0.5554** on ADNI external, outperforming raw 3D ResNet-18 (0.5128) and FOMO25 (0.5054).

### 2.3 Methodological and Governance Standards
7. **CLAIM 2024, TRIPOD+AI 2024, and PROBAST+AI 2025**  
   - High-impact reporting guidelines emphasizing that single-cohort cross-validation is profoundly misleading in medical imaging.  
   - External validation must evaluate **discrimination, calibration, and threshold sensitivity**. Multiple hypothesis testing across model pairs must be strictly adjusted using Family-Wise Error Rate (FWER) methods like Holm-Bonferroni.

8. **Wen et al. (NeuroImage 2020: "Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation")**  
   - Seminal meta-analysis demonstrating that over 80% of published AD classification papers suffer from data leakage (slice leakage across splits), lack of participant-level isolation, or failure to compare against simple demographic baselines.

9. **Shortcut Learning and Acquisition Confounding (Geirhos et al., 2020; Wachinger et al., 2021)**  
   - Demonstrating that deep neural networks readily exploit non-causal shortcuts such as scanner manufacturer signature, voxel resolution, or head coil artifacts instead of learning true neurodegenerative pathology.

---

## 3. The Seven SHIFT-MRI Methodological Novelties (N01–N07)

Conventional neuroimaging benchmarks report only a single metric—in-domain ROC-AUC—masking severe clinical deployment risks. SHIFT-MRI establishes seven distinct analytical novelties:

```mermaid
flowchart TD
    Raw[Multi-Cohort Volumetric Scans\nOASIS-1, ADNI, OASIS-2] --> Benchmark[10-Model Benchmark\nFoundation, Supervised 3D, 2.5D, Hybrid, Baselines]
    Benchmark --> N01["N01: Cross-Cohort Reliability Matrix\n(Discrimination, Calibration, Decision Bias)"]
    Benchmark --> N02["N02: Prediction Collapse Analysis\n(Threshold Drift & Positive Rate Failure)"]
    Benchmark --> N03["N03: Demographic Shortcut Baselines\n(Age & Sex Logistic Regression)"]
    Benchmark --> N04["N04: Confidence-Aware Selective Abstention\n(Risk-Coverage Profiles & AURC)"]
    Benchmark --> N05["N05: Clinical Label Sensitivity\n(Stratified CDR 0.5 vs CDR 1.0+)"]
    Benchmark --> N06["N06: Scanner Perturbation Robustness\n(Noise, Scaling, Contrast Degradation)"]
    Benchmark --> N07["N07: Out-of-Distribution Detection\n(Feature Norm & Predictive Entropy)"]
```

### N01: Comprehensive Cross-Cohort Reliability Matrix
Traditional benchmarks evaluate models in isolation. N01 unifies **discrimination** (ROC-AUC with 2,000 stratified bootstrap 95% CIs, PR-AUC, Balanced Accuracy), **calibration** (Brier score, Expected Calibration Error across 10 uniform bins, Calibration Slope, Calibration Intercept via log-odds recalibration), and **decision consistency** across in-domain locked test, zero-shot external, and unverified-overlap exploratory cohorts.

### N02: Prediction Collapse and Failure Mode Analysis
Discovers and quantifies **Decision Collapse**: models exhibiting seemingly acceptable AUC (~0.58) suffer catastrophic decision threshold breakdown under zero-shot domain shift, classifying 99% to 100% of external participants into the positive class at standard threshold $t=0.50$. Demonstrates why fixed decision thresholds are clinically hazardous and quantifies the prevalence error gap.

### N03: Demographic Shortcut Baselines
Evaluates whether expensive 3D foundation and deep learning models add statistically significant value beyond simple demographic features (Age alone, Sex alone, Age + Sex). Establishes that in external ADNI, **Age + Sex Logistic Regression achieves ROC-AUC = 0.5671**, outperforming 7 out of 10 complex deep learning and foundation model linear probes.

### N04: Confidence-Aware Selective Prediction / Abstention Framework
Formulates an uncertainty-gated abstention mechanism based on decision-boundary distance $|p(x) - 0.50|$ and predictive entropy. Demonstrates that by selectively abstaining from the 10%–40% most uncertain predictions, clinical error rate can be systematically reduced across shifted acquisition domains.

### N05: Clinical Label Sensitivity Analysis
Stratifies in-domain test cases by Clinical Dementia Rating (CDR 0.0 vs CDR 0.5 vs CDR 1.0+). Audits why early-stage cognitive impairment (CDR 0.5) is consistently missed by pure scratch 3D CNNs, whereas multi-view transfer and dual-branch morphometric models retain significant sensitivity.

### N06: Scanner & Image Perturbation Robustness
Evaluates model stability under controlled physical perturbations: additive Gaussian sensor noise ($\sigma \in [0.0, 0.10]$) and global intensity scaling ($0.85\times - 1.15\times$). Reveals how 2.5D multi-view models maintain higher invariance than unregularized 3D CNNs.

### N07: Out-of-Distribution (OOD) Detection via Latent Geometry
Leverages foundation model feature representations (FOMO-25 320-dimensional embeddings) and predictive entropy to detect acquisition domain shifts without labels. Achieves near-perfect separation (**OOD ROC-AUC = 0.9933** for feature norm; **1.0000** for entropy) between in-domain OASIS-1 and external ADNI scans.

---

## 4. Summary of Key Theoretical Contributions

| Research Question | Conventional Claim | SHIFT-MRI Verified Finding |
| :--- | :--- | :--- |
| **Foundation Model Advantage** | "Foundation models dramatically outperform supervised baselines out of the box." | **False.** Linear probe of 90M-parameter FOMO25 achieves ADNI AUC = 0.5054, inferior to supervised 3D ResNet-10 (0.5444) and 2D Transfer ResNet-18 (0.5778). |
| **Cross-Cohort Generalization** | "High validation AUC implies reliable clinical deployment." | **False.** Models with 0.82+ validation AUC drop to ~0.50–0.58 on external cohorts with complete decision collapse ($P(\text{pred}=1) \to 1.0$). |
| **Demographic Confounding** | "Deep models learn pure anatomical biomarkers of neurodegeneration." | **False.** Demographic Age + Sex logistic regression (AUC = 0.5671) outperforms 70% of deep neural networks on external ADNI. |
| **Data Governance** | "OASIS-2 is an independent external validation cohort." | **Methodologically Unverified.** OASIS-1 and OASIS-2 originate from the identical Washington University ADRC; overlap is unverified. Must be classified as secondary exploratory. |
