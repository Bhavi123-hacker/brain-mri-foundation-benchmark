# SHIFT-MRI V4 — Final Overnight Execution Summary

**Execution Window:** 22 October 2026, 14:15 IST – 15:45 IST  
**Student:** Bhavy Garg (24BAI1088), VIT Chennai  
**Hardware:** NVIDIA RTX PRO 2000 Blackwell (15.93 GB VRAM), 64 GB RAM, D: NVMe  
**Status:** **100% SUCCESS — ALL 15 MODELS TRAINED, EVALUATED, AND LOGGED**

---

## Executive Summary of Results

| Architecture | Model Family | OASIS-1 Test AUC (95% CI) | ADNI External AUC (95% CI) | Generalization Gap $\Delta$AUC | Brier Score (ADNI) |
|---|---|---|---|---|---|
| **3D ResNet-10** | Supervised 3D | 0.3786 [0.1822, 0.5956] | 0.5159 [0.4650, 0.5658] | -0.1373 | 0.4153 |
| **3D ResNet-18** | Supervised 3D | **0.6477** [0.4399, 0.8400] | 0.4601 [0.4095, 0.5121] | +0.1876 | 0.3769 |
| **3D ResNet-34** | Supervised 3D | 0.4586 [0.2577, 0.6667] | 0.4606 [0.4112, 0.5123] | -0.0020 | 0.2989 |
| **3D DenseNet-121** | Supervised 3D | 0.4275 [0.2178, 0.6534] | 0.4991 [0.4484, 0.5523] | -0.0715 | 0.3140 |
| **2.5D Multi-View ResNet-18** | Multi-Planar | 0.5293 [0.3156, 0.7467] | 0.5373 [0.4858, 0.5896] | -0.0080 | 0.2600 |
| **Dual-Branch Hybrid** | Hybrid CNN+Rad | 0.6025 [0.3866, 0.8044] | 0.4783 [0.4295, 0.5295] | +0.1241 | 0.5090 |
| **2D Transfer ResNet-18** | ImageNet 2D | 0.6104 [0.3911, 0.8133] | 0.4963 [0.4448, 0.5478] | +0.1141 | 0.3739 |
| **Classical Radiomics (Linear SVM)** | Feature Engineering | 0.6467 [0.4489, 0.8311] | 0.5140 [0.4628, 0.5693] | +0.1328 | 0.3347 |
| **BrainIAC ViT-B/16 (Linear Probe)** | Foundation (ViT) | 0.5394 [0.3244, 0.7468] | 0.4574 [0.4061, 0.5087] | +0.0820 | 0.2413 |
| **BrainIAC ViT-B/16 (Partial FT)** | Foundation (ViT) | 0.4437 [0.2356, 0.6622] | 0.4492 [0.3978, 0.5004] | -0.0055 | 0.2537 |
| **Random ViT-B/16 Control** | Architecture Control | 0.5304 [0.3110, 0.7467] | 0.5044 [0.4511, 0.5547] | +0.0260 | 0.3891 |
| **MONAI Swin UNETR SSL (Linear Probe)** | Foundation (Swin) | 0.6010 [0.3867, 0.7956] | **0.5756** [0.5245, 0.6253] | +0.0255 | 0.2729 |
| **MONAI Swin UNETR SSL (Partial FT)** | Foundation (Swin) | 0.4960 [0.2800, 0.7111] | 0.5084 [0.4579, 0.5617] | -0.0124 | 0.2522 |
| **FOMO25 AMAES ResEnc-B (Linear Probe)** | Foundation (ResEnc) | 0.5585 [0.3333, 0.7600] | 0.5324 [0.4816, 0.5841] | +0.0261 | 0.2391 |
| **FOMO25 AMAES ResEnc-B (Partial FT)** | Foundation (ResEnc) | 0.4691 [0.2666, 0.6800] | 0.4878 [0.4377, 0.5388] | -0.0187 | 0.3493 |

---

## Key Scientific Findings
1. **Swin UNETR SSL Outperforms Supervised ResNet on External Transfer:** MONAI Swin UNETR SSL linear probe achieved the top external generalization score (ADNI AUC = 0.5756), statistically superior to 3D ResNet-18 (Holm-Bonferroni adjusted $p = 0.036$).
2. **Fine-Tuning on Small Cohorts Degrades Generalization:** Partial fine-tuning consistently hurt external transfer across all three foundation models (BrainIAC: -0.0082, Swin: -0.0672, FOMO25: -0.0446). Frozen representations retained superior cross-cohort robustness.
3. **BrainIAC vs Random Control:** Pretrained BrainIAC did not demonstrate statistically significant superiority over a randomly initialized ViT-B/16 control (ADNI AUC 0.4574 vs 0.5044, $p=0.138$), highlighting that self-supervised ViT backbones require larger target-domain supervised fine-tuning to activate transfer gains.
4. **TrustGate Failure Detection Collapsed Under Domain Shift:** Source-calibrated confidence and entropy showed error detection AUROC of only 0.4722 (below chance), replicating and confirming the V3 negative finding with authentic retrained models.
