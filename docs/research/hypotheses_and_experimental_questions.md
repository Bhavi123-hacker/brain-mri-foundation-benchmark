# Research Hypotheses & Experimental Inquiries (Version 2)
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Date**: October 21, 2026  
**Authors**: Principal Investigator, Senior Deep Learning Lead, Neuroimaging Scientist & Statistical Methodologist  

---

## Central Research Question
> **Under controlled label budgets, realistic acquisition-domain shifts, and participant-independent evaluation, when do pretrained brain-MRI foundation models outperform—or fail to outperform—well-tuned supervised, transfer-learning, and anatomical-feature baselines?**

---

## Core Hypotheses

### Hypothesis 1 (H1: Representation Specificity vs. Domain Relevance)
* **Statement**: Authentic 3D medical pretraining (e.g. MONAI Swin UNETR SSL, FOMO25 AMAES) provides superior label efficiency compared to randomly initialized 3D CNNs in low-label regimes ($N \le 35$ training participants), but does not outperform natural image 2D transfer learning (ImageNet-pretrained ResNet-18) unless the 3D pretext task explicitly encourages global whole-brain atrophy representation.
* **Mechanism**: Voxel-level masked autoencoding (AMAES) optimizes localized texture and boundary reconstruction (geared toward dense segmentation), which preserves high-frequency local anatomy but fails to compress macro-structural patterns (e.g., ventricular dilation, hippocampal volumetric loss) into linearly separable diagnostic vectors without task-specific fine-tuning.
* **Empirical Test**: Compare 10%, 25%, 50%, and 100% label regimes across 3D scratch ResNet-18, authentic MedicalNet 3D ResNet-18, 2D Transfer ResNet-18, MONAI Swin UNETR SSL, and FOMO25 AMAES ResEnc-B.

---

### Hypothesis 2 (H2: Model Capacity vs. Overfitting in Volumetric Neuroimaging)
* **Statement**: Under sample-limited neuroimaging regimes ($N_{\text{train}} \approx 100-200$), compact 3D architectures (e.g. 3D ResNet-10 with ~14M parameters or 3D ResNet-18 with ~33M parameters) achieve superior validation generalization and cross-cohort transfer compared to deeper architectures (3D ResNet-34 with ~63M parameters or 100M+ parameter foundation transformers).
* **Mechanism**: High parameter counts relative to the number of independent patient training volumes cause over-parameterized models to memorize scanner-specific acquisition artifacts and non-biological noise rather than generalized neurodegenerative patterns.
* **Empirical Test**: Benchmark 3D ResNet-10, 3D ResNet-18, 3D ResNet-34, and 3D DenseNet-121 under matched training hyperparameters and regularization on OASIS-1.

---

### Hypothesis 3 (H3: 2.5D Multi-View Orthogonal Aggregation vs. Pure 3D Volumetric CNNs)
* **Statement**: Multi-planar 2.5D representations (combining orthogonal axial, coronal, and sagittal views pre-trained on diverse 2D visual corpora) provide a superior discriminative performance-to-compute ratio than fully volumetric 3D scratch models, while exhibiting lower sensitivity to cross-scanner field-strength variations.
* **Mechanism**: 2D slice backbones leverage massive visual priors from millions of images (ImageNet, DINOv2), while multi-planar orthogonal aggregation preserves spatial 3D neuroanatomical context without incurring the high memory and parameter burden of 3D volumetric convolutions.
* **Empirical Test**: Compare single-slice 2D, 16-slice axial aggregation, 2.5D orthogonal multi-view fusion, and 3D volumetric ResNet-18 on in-domain test and ADNI external transfer.

---

### Hypothesis 4 (H4: Preprocessing Harmonization & Domain Randomization vs. Model Scale)
* **Statement**: Enhancing MRI preprocessing (robust N4 bias correction, strict affine MNI registration, robust skull stripping, and intensity quantile normalization) combined with acquisition-domain randomization (random intensity scaling, spatial blurring, and noise injection) improves cross-cohort transfer to ADNI more effectively than increasing model scale or pretraining dataset size.
* **Mechanism**: Cross-cohort transfer failure in brain MRI is dominated by physical acquisition domain shifts (scanner RF coil profile, field strength, voxel spacing, and cranial tissue contamination). Aligning physical image distributions and forcing models to be invariant to acquisition perturbations mitigates calibration drift.
* **Empirical Test**: Preprocessing ablation comparing Baseline Pipeline A, Enhanced Pipeline B (N4 + MNI affine + SynthStrip), Pipeline C (Anatomical ROI extraction), and Pipeline D (Domain Randomization Augmentation).

---

### Hypothesis 5 (H5: Whole-Brain Global Features vs. Anatomical ROI Fusion)
* **Statement**: Combining global whole-brain deep features with explicit anatomical morphometric features (intracranial-volume-adjusted hippocampal volume and ventricular enlargement metrics) creates a synergistic classifier that outperforms either modality alone and provides interpretable biological grounding.
* **Mechanism**: Deep learning models excel at capturing distributed, subtle cortical thinning patterns, while explicit volumetric radiomics provide unassailable, robust markers of macroscopic neurodegeneration that are invariant to deep convolutional feature drift.
* **Empirical Test**: Benchmark whole-brain CNN embeddings alone, regional hippocampal/ventricular radiomics alone (SVM/XGBoost), and a fused multi-branch classifier.
