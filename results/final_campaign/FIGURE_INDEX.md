# Comprehensive Scientific Figure Index: Master Autonomous Campaign

**Project Title**: *Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines*  
**Directory**: `results/final_campaign/figures/`  
**Total Cataloged Artifacts**: 50+ Visualizations  

---

## 1. Section 31 Master Visualization Suite (`figures/`)

| Filename | Figure Number | Scientific Purpose & Methodology | Core Finding / Takeaway |
|---|---|---|---|
| `01_model_auc_comparison.png` | Fig 01 | Cross-model ROC-AUC comparison: Development CV vs Locked Holdout Test | Supervised CNNs lead in development; holdout test shows strong discrimination across models. |
| `02_model_accuracy_comparison.png` | Fig 02 | Balanced accuracy under frozen development-selected decision thresholds | ResNet-34 and Ensemble achieve 100% test balanced accuracy; ResNet-18 achieves 87.5%. |
| `03_model_f1_comparison.png` | Fig 03 | F1-Score comparison for the positive (Dementia) diagnostic class | Demonstrates precision-recall balance; ResNets maintain perfect harmonic scores. |
| `04_data_efficiency.png` | Fig 04 | Data-efficiency scaling curves across 10%, 25%, 50%, 100% training pool fractions | Supervised CNNs exhibit steep sample scaling; foundation linear probes show flat trajectories. |
| `05_learning_curves_resnet18.png` | Fig 05 | Supervised 3D ResNet-18 training & validation BCE loss convergence trajectories | Stable convergence with minimum validation loss achieved at Epoch 18. |
| `06_learning_curves_resnet34.png` | Fig 06 | Supervised 3D ResNet-34 training & validation BCE loss convergence trajectories | Smooth convergence to Epoch 22 checkpoint without over-fitting divergence. |
| `07_roc_curves.png` | Fig 07 | Receiver Operating Characteristic curves on holdout test partition ($N=8$) | ResNet-18/34 achieve AUC 1.000; BrainIAC achieves 0.938; FOMO25/Swin achieve 0.812. |
| `08_precision_recall_curves.png` | Fig 08 | Precision-Recall curves across all candidate network architectures | Average Precision (AP) demonstrates high positive predictive value above 50% prior baseline. |
| `09_confusion_matrices.png` | Fig 09 | 3-Panel confusion matrices (ResNet-34, ResNet-18, Swin UNETR) on test set | Depicts exact TP, TN, FP, FN counts under pre-test frozen thresholds. |
| `10_calibration_curves.png` | Fig 10 | Reliability diagrams and Expected Calibration Error (ECE) curves | Temperature scaling aligns predicted probabilities with true empirical frequencies. |
| `11_parameter_efficiency.png` | Fig 11 | 2D Pareto frontier mapping validation CV AUC against trainable parameters | Foundation linear probes require <0.1% parameters, dominating the low-capacity frontier. |
| `12_compute_efficiency.png` | Fig 12 | Wall-clock training duration (seconds, log scale) vs CV performance | Foundation linear probes train in seconds; full 3D CNNs require backpropagation compute. |
| `13_seed_stability.png` | Fig 13 | Multi-seed predictive variance ($\sigma_{\text{AUC}}$ across Seeds 42, 43, 44, 45, 46) | ResNet-18 achieves highest stability ($\pm 0.034$); foundation probes display higher fold variance. |
| `14_representation_pca.png` | Fig 14 | 2D PCA projection of penultimate latent spaces labeled by diagnostic class | Supervised CNNs form linearly separable clusters; foundation latents are diffuse. |
| `15_representation_umap.png` | Fig 15 | 2D UMAP non-linear manifold projection of latent embeddings | Confirms task-directed cluster separation for supervised CNN representations. |
| `16_fine_tuning.png` | Fig 16 | Frozen linear probing vs end-to-end foundation backbone fine-tuning | Fine-tuning multi-million parameter backbones on $N=35$ subjects causes severe overfitting. |
| `17_fusion_comparison.png` | Fig 17 | Standalone CNN vs Foundation + CNN feature concatenation fusion | Joint feature fusion elevates CV AUC to 0.9800, proving architectural complementarity. |
| `18_augmentation_ablation.png` | Fig 18 | Augmentation ablation (None, Light affine, Standard, Aggressive elastic) | Light 3D rotation and scaling optimizes CV AUC (0.8950); elastic warping degrades boundaries. |
| `19_preprocessing_ablation.png` | Fig 19 | Intensity normalization ablation (Min-Max, Z-score, Robust Median/IQR) | Volumetric Median/IQR robust scaling (Pipeline C) delivers highest multi-scanner stability. |
| `20_domain_shift.png` | Fig 20 | Cross-scanner domain generalization gap (1.5T OASIS vs 3.0T ADNI) | Foundation models transfer with 1.0000 AUC, outperforming un-adapted CNNs. |
| `21_subgroup_analysis.png` | Fig 21 | Subgroup robustness across Sex (M/F), Age brackets, and Scanners | Confirms no significant subgroup disparity, with explicit small-N power caveats. |
| `22_prediction_confidence.png` | Fig 22 | Predicted probability confidence distribution for CN vs Dementia scans | Bimodal risk distribution separating CN controls (<0.20) from Impaired subjects (>0.80). |
| `23_dataset_characteristics.png`| Fig 23 | Composite 4-panel dataset characterization comparison (OASIS vs ADNI) | Synthesizes Age, Sex, Field Strength, and Class balance across cohorts. |
| `24_cross_dataset_gradcam.png` | Fig 24 | Cross-dataset 3D Grad-CAM comparison between OASIS-1 and ADNI subjects | Visualizes shared medial temporal and lateral ventricular focus across domains. |
| `25_error_analysis.png` | Fig 25 | Aggregate error analysis breakdown (TP, TN, FP, FN) | Quantifies benign ventriculomegaly (FP) and early focal atrophy (FN) failure modes. |
| `26_trainable_parameters.png` | Fig 26 | Log-scale horizontal bar chart of trainable parameters per model | Highlights the 4-orders-of-magnitude parameter disparity between paradigms. |
| `27_label_efficiency.png` | Fig 27 | High-resolution label efficiency sample scaling trajectory | Documents performance inflection points as labeled data grows from 4 to 35 subjects. |
| `prediction_confidence_distribution.png` | Supp | Standalone probability density distribution for clinical calibration auditing | Confirms proper logit spread under development-selected decision boundaries. |

---

## 2. Section 25 Dataset Characterization Figures (`figures/dataset/`)

| Filename | Description | Takeaway |
|---|---|---|
| `01_age_distribution.png` | Histogram overlay comparing OASIS-1 ($71.0 \pm 15.0\text{y}$) and ADNI ($73.8 \pm 6.4\text{y}$) | OASIS-1 has wider age spread; ADNI is strictly clustered in older adulthood. |
| `02_sex_distribution.png` | Grouped bar chart of sex distributions (OASIS: 60% F; ADNI: 66% F) | Both cohorts reflect female predominance typical of Alzheimer's epidemiology. |
| `03_class_distribution.png` | Class balance bar chart comparing CN vs Impaired cohorts | Balanced 50%/50% diagnostic setup across both benchmark evaluations. |
| `04_field_strength.png` | Magnetic field strength comparison (1.5 Tesla OASIS vs 3.0 Tesla ADNI) | Major domain shift parameter: 1.5T Vision vs 3.0T Trio/Skyra systems. |
| `05_dataset_comparison.png` | 4-Panel composite summary combining Age, Sex, Magnet, and Class splits | Comprehensive multi-cohort demographic overview for publication. |

---

## 3. Interpretability & 3D Grad-CAM Suite (`interpretability/`)

| Filename | Directory | Contents & Purpose |
|---|---|---|
| `figure_A_oasis_resnet18.png` | `interpretability/` | $3 \times 3$ Orthogonal Grad-CAM panel for ResNet-18 on OASIS-1 |
| `figure_B_adni_resnet18.png` | `interpretability/` | $3 \times 3$ Orthogonal Grad-CAM panel for ResNet-18 on ADNI external transfer |
| `figure_C_oasis_resnet34.png` | `interpretability/` | $3 \times 3$ Orthogonal Grad-CAM panel for ResNet-34 on OASIS-1 |
| `figure_D_adni_resnet34.png` | `interpretability/` | $3 \times 3$ Orthogonal Grad-CAM panel for ResNet-34 on ADNI external transfer |
| `cross_dataset_cnn_interpretability_paper.png` | `interpretability/` | Side-by-side publication composite comparing ResNet-18 and ResNet-34 across cohorts |
| `oasis_resnet18_gradcam/` | `interpretability/` | Individual orthogonal slice PNGs for all evaluated OASIS subjects |
| `adni_resnet18_gradcam/` | `interpretability/` | Individual orthogonal slice PNGs for all evaluated ADNI subjects |
| `oasis_resnet34_gradcam/` | `interpretability/` | Individual orthogonal slice PNGs for all evaluated OASIS subjects |
| `adni_resnet34_gradcam/` | `interpretability/` | Individual orthogonal slice PNGs for all evaluated ADNI subjects |
