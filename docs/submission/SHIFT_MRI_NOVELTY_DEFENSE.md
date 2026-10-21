# SHIFT-MRI: Viva Voce Defense Guide and Novelty Defense

**Project:** SHIFT-MRI Benchmark Framework  
**Target:** Academic & Industry Technical Examination Panels  
**Date:** October 2026  

---

## 1. Primary Thesis Statement

> "In 3D volumetric brain MRI classification of Alzheimer's disease, massive self-supervised foundation-model pretraining does not guarantee out-of-the-box cross-cohort generalization or calibration. When audited under matched label budgets and multi-site acquisition shift, well-tuned supervised 2D transfer and hybrid deep-morphometric baselines achieve equivalent or superior discrimination, while all deep models suffer catastrophic decision collapse at default thresholds and fail to significantly outperform non-imaging demographic priors (Age + Sex)."

---

## 2. Twenty Rigorous Viva Defense Questions and Model Answers

### Category A: Foundation Models vs Supervised Baselines

#### Q1: "Why did the 90-million parameter FOMO25 foundation model linear probe achieve an AUC of only 0.5054 on external ADNI, performing no better than random guessing?"
**Model Answer:**  
FOMO25 AMAES was pretrained via masked autoencoding on a broad multi-organ MRI collection (FOMO-260k). While the upstream encoder learns diverse volumetric edge and contrast representations, a frozen linear probe trained on a single-center cohort (OASIS-1, 1.5T Siemens Vision) locks its decision hyperplane to the specific latent feature coordinates induced by that scanner's acquisition protocol. When evaluated zero-shot on ADNI—which spans multiple manufacturers (GE, Philips, Siemens) and 3.0T field strengths—the global feature distribution drifts. Because the backbone is frozen, the linear classifier cannot adapt to the covariate shift, resulting in near-chance discrimination (AUC = 0.5054 [95% CI: 0.4516, 0.5552]).

#### Q2: "MONAI Swin UNETR SSL achieved the highest AUC on ADNI (0.5834). Doesn't that prove foundation models are superior?"
**Model Answer:**  
While Swin UNETR SSL achieved an external AUC of 0.5834, our paired bootstrap statistical audit with Holm-Bonferroni correction ($p_{\text{Holm}} = 1.000$) confirms that this difference is **not statistically significant** compared to a lightweight 2D Transfer ResNet-18 (AUC = 0.5778) or our Dual-Branch Hybrid model (AUC = 0.5554). Furthermore, reliability analysis reveals that Swin UNETR SSL is severely miscalibrated: at the default threshold $t=0.50$, it predicts positive (AD) for 100.0% of external participants, yielding a specificity of 0.0% and balanced accuracy of 0.5000. Higher ranking on a single rank-order metric does not translate to clinical utility without calibration.

#### Q3: "Why did slice-aggregated 2D Transfer ResNet-18 generalize so well to ADNI compared to complex 3D CNNs?"
**Model Answer:**  
2D ResNet-18 leverages features pretrained on 1.28 million natural images (ImageNet-1K). Low-level 2D convolutional filters (Gabor-like edge detectors, texture gradients) are highly regularized and scale-invariant. By extracting central axial slices and aggregating predictions across slices, the 2D model bypasses the volumetric noise, anisotropic voxel artifacts, and through-plane distortion that plague 3D convolutions trained from scratch on small medical datasets.

---

### Category B: Generalization, Calibration, and Decision Collapse

#### Q4: "What is 'Decision Collapse' (Novelty N02), and why does it occur?"
**Model Answer:**  
Decision Collapse describes a critical failure mode where a classifier's predicted probability distribution shifts entirely to one side of the decision threshold when tested on an external domain. On ADNI ($N=500$; true prevalence = 39.4%), 7 out of 10 evaluated models predicted positive (AD) for 99% to 100% of cases at $t=0.50$. This occurs because scanner differences (e.g., higher SNR at 3.0T, differing RF coil profiles) induce a positive shift in intermediate logits. Even though the model can still weakly rank subjects (AUC ~0.55–0.58), the uncalibrated probability shift causes the hard decision boundary to completely fail.

#### Q5: "Why couldn't you fix Decision Collapse by using the optimal Youden's threshold ($t^*$) computed on OASIS-1?"
**Model Answer:**  
We empirically tested this in Novelty N02. On OASIS-1, Youden's optimal threshold $t^*$ was near 0.50 (e.g., $t^* = 0.513$ for Swin UNETR; $0.390$ for Dual-Branch Hybrid). Because the external ADNI logit distribution shifted upwards across the board, applying the source-locked threshold $t^*$ still resulted in predicted positive rates of 100.0%. This proves that **threshold transfer is non-viable** across scanner domains; threshold recalibration must be performed using target-domain unlabeled or calibration data.

#### Q6: "Why did you measure Brier score and Expected Calibration Error (ECE) instead of just ROC-AUC?"
**Model Answer:**  
ROC-AUC measures only discrimination (the probability that a randomly chosen positive case is ranked higher than a randomly chosen negative case). It is invariant to monotonic logit transformations. In clinical medicine, however, predicted probabilities guide triage decisions, risk assessment, and biopsy referrals. A model with AUC = 0.85 that outputs probabilities clustered between 0.98 and 0.99 for all patients is completely useless for individual risk stratification. Brier score and ECE quantify probability accuracy, exposing external calibration degradation ($\Delta \text{Brier} > 0.20$) that ROC-AUC masks.

---

### Category C: The Demographic Shortcut Confound (Novelty N03)

#### Q7: "How do demographic baselines compare against deep vision models on the full ADNI external cohort (N=500)?"
**Model Answer:**  
Alzheimer's disease risk increases exponentially with age. Because age influences brain atrophy, deep neural networks readily learn brain age as a proxy shortcut for dementia. When evaluated on external cohorts with slightly different demographic balances, the deep models' imaging features provide less discriminative signal than simple demographic variables. Any benchmark that does not benchmark against an Age + Sex tabular baseline cannot verify whether a deep learning model has learned actual neurodegenerative pathology or merely an expensive, imprecise proxy for patient age.

#### Q8: "If Age + Sex is so effective, should clinicians stop using MRI for Alzheimer's classification?"
**Model Answer:**  
No. MRI provides essential structural and differential diagnostic information (excluding vascular strokes, subdural hematomas, and normal pressure hydrocephalus). However, our finding proves that current end-to-end 3D deep classifiers are underperforming their clinical potential. Clinical imaging models must demonstrate that their incremental diagnostic value ($\Delta \text{AUC}$) beyond demographic risk factors is statistically significant, rather than assuming that high deep learning capacity automatically captures unique pathology.

---

### Category D: Model Architecture, Capacity, and Overfitting

#### Q9: "Why did lightweight 3D ResNet-10 (14.6M params) outperform heavy 3D ResNet-34 (63.5M params) on ADNI?"
**Model Answer:**  
In medical imaging where training set sizes are modest (OASIS-1 train $N=138$), deep 3D convolutions suffer from the curse of dimensionality. 3D ResNet-34 has 63.5 million parameters—an overparameterization ratio of nearly 500,000 parameters per training volume. This causes the network to memorize scanner-specific high-frequency noise and site-specific skull artifacts, resulting in cross-cohort failure (ADNI AUC = 0.4777). 3D ResNet-10 (14.6M parameters) constrains capacity, acting as an architectural regularizer that generalizes better (ADNI AUC = 0.5444).

#### Q10: "How does the Dual-Branch Hybrid architecture combine deep learning with morphometry?"
**Model Answer:**  
Dual-Branch Hybrid combines a 3D ResNet-10 convolutional spatial branch with a tabular morphometric branch. The tabular branch takes 6 biologically grounded anatomical proxies extracted from the MRI volume: total brain voxel volume, ventricular CSF volume, medial temporal / hippocampal tissue volume, ventricle-to-brain ratio, hippocampus-to-brain ratio, and foreground intensity skewness. The convolutional and morphometric embeddings are concatenated before the final classification head. This anchors the deep representations to proven macroscopic hallmarks of Alzheimer's disease.

---

### Category E: Preprocessing Truth and Controlled Ablations

#### Q11: "Previous project summaries claimed that N4 bias correction, SynthStrip skull-stripping, and MNI-152 registration were applied. Why did your audit withdraw these claims?"
**Model Answer:**  
Scientific integrity requires reporting what was physically executed, not what was aspirational. When we inspected the raw dataset generation scripts and cached tensors in `D:\brain_mri_data\*\processed\tensors`, we verified that the tensors were produced solely via trilinear spatial resampling to $128^3$ and nonzero foreground Z-score normalization. No N4 bias field correction, SynthStrip, or non-linear atlas registration tools were ever executed. In accordance with CLAIM 2024, we immediately corrected the scientific record, withdrew the false preprocessing claims, and re-evaluated the benchmark under the verified physical baseline.

#### Q12: "What did your controlled intensity outlier clipping ablation prove?"
**Model Answer:**  
We evaluated whether clipping extreme foreground voxel intensities to the 0.5%–99.5% percentile range before Z-scoring improved generalization. The ablation showed that while clipping artificially boosted in-domain validation AUC from 0.6622 to 0.7400, it actually degraded external ADNI AUC from 0.5185 down to 0.4973. This demonstrated that percentile clipping acts as an in-domain over-regularizer that truncates meaningful contrast differences across external scanner manufacturers.

---

### Category F: Dataset Governance and Independence

#### Q13: "Why is it scientifically invalid to claim OASIS-2 as an independent external validation cohort?"
**Model Answer:**  
Both OASIS-1 (cross-sectional) and OASIS-2 (longitudinal) were collected at the same institution: the Washington University Alzheimer's Disease Research Center (ADRC) in St. Louis, Missouri, using the same 1.5T Siemens Vision scanner. Longitudinal studies routinely recruit returning participants from existing cross-sectional registries. While their public IDs follow different namespaces (`OAS1_XXXX` vs `sub-OASIS2XXXX`), the absence of a cross-referencing key means participant overlap cannot be disproven. Calling OASIS-2 an 'independent external cohort' violates CLAIM 2024 and PROBAST+AI standards. It must be disclosed as an exploratory secondary cohort with unverified overlap.

#### Q14: "Why couldn't you simply download OASIS-3 or AIBL before the deadline?"
**Model Answer:**  
OASIS-3 and AIBL require formal institutional Data Use Agreements (DUAs) with human review and approval timelines typically requiring 2 to 4 weeks. Downloading restricted medical imaging data without signed DUA compliance violates federal health data privacy laws and academic integrity codes. Our audit confirmed that adhering strictly to the verified accessible cohorts (OASIS-1, ADNI, OASIS-2) was the only legally and scientifically compliant path.

---

### Category G: Statistical Methodology and Multiplicity Control

#### Q15: "Why did you use paired bootstrap hypothesis testing with Holm-Bonferroni correction instead of independent t-tests?"
**Model Answer:**  
Predictions from different models evaluated on the same test participants are paired and highly correlated, violating the independence assumption of Student's t-test. Furthermore, parametric assumptions do not hold for non-linear rank metrics like ROC-AUC. Paired bootstrap resampling ($B=2,000$) preserves the correlation structure across subjects. Holm-Bonferroni adjustment controls the Family-Wise Error Rate (FWER) under sequential testing, preventing false discoveries across multiple model comparisons.

#### Q16: "What happened to the previously reported p-value of p = 0.641?"
**Model Answer:**  
Our statistical audit revealed a historical misattribution: $p = 0.641$ was previously described as comparing FOMO25 against Classical Radiomics. In our paired bootstrap audit, $p = 0.641$ was confirmed to belong strictly to the comparison between the **Dual-Branch Hybrid model and Classical Radiomics** on the OASIS-1 locked test set ($\Delta \text{AUC} = +0.0606$). After Holm-Bonferroni correction across all seven benchmark comparisons, this difference is non-significant ($p_{\text{Holm}} = 1.000$).

---

### Category H: Advanced Clinical Novelties (N04–N07)

#### Q17: "How does your Confidence-Aware Selective Prediction framework (N04) function in practice?"
**Model Answer:**  
In clinical deployment, forced binary decisions on highly uncertain cases cause dangerous diagnostic errors. We quantify uncertainty as boundary distance $|p(x) - t^*|$. The system accepts predictions where confidence exceeds an operational threshold and abstains on ambiguous scans, referring them to expert neuroradiologists. On ADNI, abstaining from the 30% most uncertain scans reduced the diagnostic error rate of MONAI Swin UNETR from 60.6% down to 56.3% (AURC = 0.2871).

#### Q18: "Why do models struggle specifically with CDR 0.5 (Novelty N05)?"
**Model Answer:**  
CDR 0.5 represents very mild cognitive impairment or prodromal Alzheimer's disease. At this stage, macroscopic brain atrophy is subtle and localized primarily to the entorhinal cortex and subfields of the hippocampus. Convolutional filters downsampling volumes to $128^3$ blur these minute morphological changes. Dual-Branch Hybrid models mitigate this limitation by computing explicit medial temporal volume ratios, achieving 86.7% sensitivity on CDR 0.5 compared to 20%–70% for pure scratch 3D CNNs.

#### Q19: "How can foundation models achieve an OOD detection AUC of 1.0000 (Novelty N07) while failing at downstream classification?"
**Model Answer:**  
Foundation models trained on diverse MRI data map volumetric scans into rich geometric manifolds. Scanner acquisition shifts (differing RF coils, B0 field inhomogeneities, gradient non-linearities) systematically alter feature vectors, shifting their L2 norm and predictive entropy. The representation is sensitive enough to detect that an external scan does not match the training distribution (OOD AUC = 1.0000 for FOMO25 Mahalanobis distance; 0.9930 for centroid Euclidean distance), even though a linear probe trained on in-domain data cannot disentangle those scanner shifts from disease pathology.

#### Q20: "What is the single most important recommendation of the SHIFT-MRI study for future neuroimaging AI research?"
**Model Answer:**  
Future neuroimaging AI benchmarks must abandon the practice of evaluating models solely on in-domain, single-scanner cross-validation. True clinical translation requires multi-site external validation (e.g., ADNI), reporting calibration metrics and decision boundary stability alongside ROC-AUC, testing against simple demographic shortcut baselines, and incorporating uncertainty-gated selective abstention.
