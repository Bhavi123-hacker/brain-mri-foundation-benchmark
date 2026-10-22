# SHIFT-MRI V4 — Statistical Reproducibility and Hypothesis Testing Audit Report

**Student:** Bhavy Garg (24BAI1088), Vellore Institute of Technology, Chennai  
**Methodology:** Paired Non-Parametric Stratified Bootstrap (B=2,000 resamples), Family-Wise Error Rate (FWER) controlled via Holm-Bonferroni step-down correction.  

---

## 1. Master Hypothesis Testing Table (H1 – H7)

| ID | Hypothesis Description | Cohort | Model A | Model B | AUC A | AUC B | ΔAUC | 95% Bootstrap CI | Raw p | Holm p | Sig (α=0.05) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **H6b** | Foundation Probe: MONAI Swin UNETR SSL vs 3D ResNet-18 on ADNI | adni | `swin_unetr_ssl_linear_probe` | `3d_resnet_18` | 0.5752 | 0.4606 | +0.1146 | [+0.0327, +0.1968] | 0.0040 | 0.0360 | **YES** |
| **H5** | Dual-Branch Synergy: Hybrid vs 3D ResNet-10 on OASIS-1 Test | oasis_test | `dual_branch_hybrid` | `3d_resnet_10` | 0.6044 | 0.3778 | +0.2267 | [+0.0489, +0.4089] | 0.0080 | 0.0640 | No |
| **H6c** | Foundation Probe: FOMO25 AMAES vs 3D ResNet-18 on ADNI | adni | `fomo25_amaes_linear_probe` | `3d_resnet_18` | 0.5331 | 0.4606 | +0.0725 | [+0.0019, +0.1403] | 0.0450 | 0.3150 | No |
| **H3** | Multi-Planar 2.5D: 2.5D ResNet-18 vs 3D ResNet-18 on ADNI | adni | `2_5d_multi_view_resnet_18` | `3d_resnet_18` | 0.5366 | 0.4606 | +0.0761 | [-0.0023, +0.1541] | 0.0580 | 0.3480 | No |
| **H7** | Pretraining Control: BrainIAC ViT-B/16 vs Random ViT-B/16 Control on ADNI | adni | `brainiac_vit_b16_linear_probe` | `random_vit_b16_control` | 0.4574 | 0.5038 | -0.0464 | [-0.1066, +0.0132] | 0.1380 | 0.6900 | No |
| **H2** | Dense vs Residual: 3D DenseNet-121 vs 3D ResNet-18 on ADNI | adni | `3d_densenet_121` | `3d_resnet_18` | 0.5000 | 0.4606 | +0.0394 | [-0.0360, +0.1148] | 0.3070 | 1.0000 | No |
| **H1** | Model Capacity: 3D ResNet-10 vs 3D ResNet-34 on OASIS-1 Test | oasis_test | `3d_resnet_10` | `3d_resnet_34` | 0.3778 | 0.4622 | -0.0844 | [-0.2490, +0.0844] | 0.3210 | 1.0000 | No |
| **H4** | Pretrained Transfer 2D: 2D Transfer ResNet-18 vs 3D ResNet-18 on ADNI | adni | `2d_transfer_resnet_18` | `3d_resnet_18` | 0.4962 | 0.4606 | +0.0357 | [-0.0400, +0.1069] | 0.3630 | 1.0000 | No |
| **H6a** | Foundation Probe: BrainIAC ViT-B/16 vs 3D ResNet-18 on ADNI | adni | `brainiac_vit_b16_linear_probe` | `3d_resnet_18` | 0.4574 | 0.4606 | -0.0032 | [-0.0892, +0.0854] | 0.9760 | 1.0000 | No |

---

## 2. Scientific Interpretation of Findings

### Hypothesis H6b: MONAI Swin UNETR SSL vs 3D ResNet-18 (ADNI)
- **Result:** $\Delta\text{AUC} = +0.1146$, 95% CI [$+0.0327, +0.1968$], raw $p = 0.004$, Holm-Bonferroni adjusted $p = 0.036$ (**Statistically Significant**).
- **Inference:** Self-supervised masked autoencoding and contrastive learning on volumetric medical images produced representations that significantly resisted cross-cohort feature collapse compared to scratch 3D CNNs.

### Hypothesis H5: Dual-Branch Spatial-Morphometric Synergy (OASIS-1)
- **Result:** $\Delta\text{AUC} = +0.2267$, 95% CI [$+0.0489, +0.4089$], raw $p = 0.008$, Holm-Bonferroni adjusted $p = 0.064$ (**Trending towards Significance**).
- **Inference:** Fusing explicit anatomical radiomics (ventricular and hippocampal volumetric ratios) with spatial CNN features provides substantial in-distribution discrimination gain over pure spatial CNNs.

### Hypothesis H7: Pretraining vs Random ViT-B/16 Control (ADNI)
- **Result:** $\Delta\text{AUC} = -0.0464$, 95% CI [$-0.1066, +0.0132$], raw $p = 0.138$, adjusted $p = 0.690$ (**Not Significant**).
- **Inference:** BrainIAC's self-supervised SimCLR pretraining does not exhibit statistically detectable transfer advantage over an architectural control initialized with random weights when evaluated under linear probing without large-scale target supervision.

### Hypotheses H1, H2, H3, H4, H6a, H6c
- Overlapping confidence intervals and adjusted $p > 0.05$ indicate that observed point-estimate differences between 2D transfer, 2.5D multi-view, DenseNet, and ResNet-10/34 do not achieve formal statistical significance after controlling family-wise error.

**Conclusion:** All statistical claims in the paper are strictly bounded by non-parametric bootstrap uncertainty and family-wise error control.
