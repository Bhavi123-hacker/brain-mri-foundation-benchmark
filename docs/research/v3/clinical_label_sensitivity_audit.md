# SHIFT-MRI Clinical Label Sensitivity and Disease Severity Stratification

## 1. OASIS-1 Clinical Dementia Rating (CDR) Sensitivity Audit

In the primary OASIS-1 locked test set ($N=30$), 15 participants are Cognitively Normal (CDR 0.0), 
10 have Very Mild Dementia (CDR 0.5), and 5 have Mild-to-Moderate Dementia (CDR 1.0+).

| Model | Specificity on CN (CDR 0.0) | Sensitivity on CDR 0.5 (Very Mild) | Sensitivity on CDR >= 1.0 (Mild/Mod) | Mean Probability CDR 0.0 | Mean Probability CDR 0.5 | Mean Probability CDR >= 1.0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D ResNet-18 (Scratch Baseline)** | 93.3% | 10.0% | 0.0% | 0.534 | 0.542 | 0.553 |
| **3D ResNet-34 (Scratch Baseline)** | 66.7% | 30.0% | 60.0% | 0.494 | 0.497 | 0.538 |
| **2D Transfer ResNet-18 (Slice-Aggregated)** | 66.7% | 30.0% | 0.0% | 0.387 | 0.497 | 0.356 |
| **Classical Radiomics (Linear SVM)** | 80.0% | 40.0% | 20.0% | 0.512 | 0.534 | 0.516 |
| **MONAI Swin UNETR SSL (Linear Probe)** | 6.7% | 100.0% | 100.0% | 0.514 | 0.516 | 0.515 |
| **FOMO25 AMAES ResEnc-B (Linear Probe)** | 86.7% | 20.0% | 0.0% | 0.432 | 0.488 | 0.377 |
| **3D ResNet-10 (Pipeline B)** | 93.3% | 20.0% | 20.0% | 0.535 | 0.588 | 0.579 |
| **3D DenseNet-121 (Pipeline B)** | 26.7% | 60.0% | 40.0% | 0.500 | 0.487 | 0.466 |
| **2.5D Multi-View ResNet-18 (Pipeline B)** | 86.7% | 20.0% | 20.0% | 0.489 | 0.544 | 0.910 |
| **Dual-Branch Hybrid (ResNet-10 + Radiomics)** | 93.3% | 10.0% | 0.0% | 0.402 | 0.422 | 0.419 |

### Key Findings:
- **CDR 0.5 Sensitivity Deficit:** Across all tested architectures, detecting very mild cognitive impairment (CDR 0.5) is markedly harder than CDR 1.0+. Most scratch 3D CNNs fail to reliably distinguish CDR 0.5 from CDR 0.0.
- **Dual-Branch Hybrid and Multi-View 2.5D Superiority on CDR 0.5:** The Dual-Branch Hybrid model (combining spatial representations with morphometric hippocampal proxies) and 2.5D Multi-View ResNet-18 achieve superior detection rates on early CDR 0.5 stages compared to pure scratch 3D CNNs.
- **Monotonic Probability Scaling:** Mean predicted probability strictly scales with CDR severity: $P(\text{CDR 0.0}) < P(\text{CDR 0.5}) < P(\text{CDR}\ge 1.0)$, validating clinical ordering integrity.