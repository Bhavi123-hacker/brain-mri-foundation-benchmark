# Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11+cu128-red.svg)](https://pytorch.org/)
[![MONAI 1.4](https://img.shields.io/badge/MONAI-1.4.0-brightgreen.svg)](https://monai.io/)
[![Hardware: NVIDIA RTX PRO 2000 Blackwell](https://img.shields.io/badge/GPU-RTX_PRO_2000_Blackwell-green.svg)](https://www.nvidia.com/)

A rigorous, reproducible, open-science research benchmark evaluating modern **3D Brain-MRI Foundation Models** against **well-tuned supervised 3D convolutional neural networks**, **2D transfer learning**, and **classical radiomics** for Alzheimer’s disease classification across controlled label budgets, multi-center scanner domain shifts, paired bootstrap uncertainty quantification, and strict multiple-comparison corrections.

---

## 📊 Master Benchmark Comparison Table

All models evaluated under strictly isolated patient partitions on **OASIS-1** ($N=198$; 138 train, 30 val, 30 locked test) and external zero-shot transfer on **ADNI** ($N=500$; 303 Cognitively Normal, 197 Alzheimer's Disease):

| Architecture | Family / Paradigm | Parameters | Weights Source | Status | OASIS Val AUC | OASIS Test ROC-AUC (95% CI) | OASIS Test BalAcc | ADNI External ROC-AUC (95% CI) | ADNI Ext BalAcc |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D ResNet-18 (Scratch)** | Supervised 3D CNN | 33.2M | Random (MedicalNet init) | **COMPLETED** | 0.5600 | **0.6178** [0.4133, 0.8133] | 0.5333 | **0.5128** [0.4612, 0.5644] | 0.4940 |
| **3D ResNet-34 (Scratch)** | Supervised 3D CNN | 63.5M | Random (MedicalNet init) | **COMPLETED** | 0.5822 | **0.5311** [0.3200, 0.7333] | 0.5000 | **0.4777** [0.4261, 0.5293] | 0.5133 |
| **2D Transfer ResNet-18** | 2D Pretrained Transfer | 11.2M | Torchvision ImageNet-1K | **COMPLETED** | **0.7778** | **0.6222** [0.4133, 0.8178] | 0.4333 | **0.5778** [0.5280, 0.6276] | 0.5000 |
| **Classical Radiomics (SVM)** | Engineered Radiomics | 25 | 24 Volume/Texture Features | **COMPLETED** | **0.6978** | **0.5689** [0.3600, 0.7689] | 0.5667 | **0.5149** [0.4638, 0.5660] | 0.4957 |
| **MONAI Swin UNETR SSL** | 3D Foundation Transformer | 8.1M | MONAI SSL (`model_swinvit.pt`)| **COMPLETED** | 0.4467 | **0.6044** [0.3956, 0.8044] | 0.5000 | **0.5834** [0.5331, 0.6337] | 0.5000 |
| **FOMO25 AMAES ResEnc-B** | 3D Foundation Residual Enc | 90.3M | FOMO-260K (`resenc_unet_b.ckpt`)| **COMPLETED** | **0.6667** | **0.5733** [0.3664, 0.7989] | 0.5333 | **0.5054** [0.4516, 0.5552] | 0.5000 |
| **BrainIAC Foundation Model** | 3D Foundation Transformer | ~65M | AIM-Harvard/BrainIAC | **BLOCKED** | N/A | N/A | N/A | N/A | N/A |
| **FOMO25 Winner (FOMO2JOMO)** | 3D Foundation Transformer | ~86M | jbanusco/FOMO25-Winner | **BLOCKED** | N/A | N/A | N/A | N/A | N/A |

*Note*: BrainIAC and FOMO25 Winner were audited and categorized as **BLOCKED** due to private token access requirements (`HTTP 401 Unauthorized`) on Hugging Face. Zero mock or randomly initialized weights were substituted.

---

## 🔬 Core Scientific Findings

1. **No Foundation Model Superiority from Frozen Probes**: Modern 3D foundation models (FOMO25 AMAES ResEnc-B, Test AUC **0.5733**; MONAI Swin UNETR SSL, Test AUC **0.6044**) do not outperform well-tuned supervised 3D ResNets (Test AUC **0.6178**) or 2D ImageNet transfer learning (Test AUC **0.6222**) for structural T1 Alzheimer's classification.
2. **Pretext-Task Misalignment**: The AMAES pretext task is volumetric masked autoencoding designed for dense voxel-level segmentation reconstruction. Its spatial representations retain localized boundary textures rather than compressing whole-brain neurodegenerative atrophy patterns into linearly separable diagnostic manifolds.
3. **Small-Cohort Statistical Equivalence**: Paired non-parametric bootstrap resampling ($B=2,000$) reveals that **all 15 pairwise comparisons on the OASIS locked test set ($N=30$) cross zero in their 95% confidence intervals** (all raw $p > 0.49$; all Holm-Bonferroni $p = 1.000$). Small-sample rank orderings are statistically indistinguishable.
4. **Severe Multi-Center Calibration Drift**: On external zero-shot transfer to ADNI ($N=500$), cross-scanner physical differences shift predicted probability distributions strongly positive ($\bar{p} = 0.9398$ for 2D ResNet; $\bar{p} = 0.7157$ for FOMO AMAES). Under an uncalibrated default decision threshold ($t=0.50$), 100% of cases are predicted positive, collapsing specificity to 0.0000. Post-hoc Youden thresholding recovers balanced classification (0.5741 for 2D ResNet; 0.5146 for FOMO AMAES).
5. **Multiple-Comparison Correction**: Under strict Holm-Bonferroni Family-Wise Error Rate control across the 15 planned pairwise tests on ADNI, only **MONAI Swin UNETR SSL vs 3D ResNet-34** retains statistical significance ($\Delta \text{AUC} = +0.1060$, $p_{\text{Holm}} = 0.0150$). FOMO25 AMAES is statistically indistinguishable from a scratch 3D ResNet-18 ($p = 0.8070$).

---

## 📑 Flagship Research Deliverables

All authoritative research reports, statistical audits, literature matrices, and manifests are available under `results/report/`:

- 📖 **[Final Comprehensive Research Report](results/report/final_research_report.md)**: Full manuscript detailing motivation, 10-reference bibliographic audit, methods, results, calibration analysis, and discussion.
- 📊 **[Master Model Comparison Table (CSV)](results/report/master_model_comparison_table.csv)**: Machine-readable comparison table across all evaluated architectures and evaluation splits.
- 📐 **[Statistical, Uncertainty & Calibration Audit Summary](results/report/statistical_audit_summary.md)**: Detailed paired bootstrap tables, Holm-Bonferroni adjustments, and calibration drift analyses.
- 🗺️ **[Research Gap Matrix (CSV)](results/report/research_gap_matrix.csv)**: Systematic matrix mapping literature claims against empirical benchmark findings.
- 🌐 **[Dataset Expansion Audit (CSV)](results/report/dataset_expansion_audit.csv)**: Exhaustive audit of candidate cohorts (OASIS-1/2/3, ADNI, MIRIAD, AIBL, FOMO45K/50K, UK Biobank).
- ⚙️ **[Reproducibility & Environment Manifest](results/report/reproducibility_manifest.md)**: Cryptographic checksums (SHA-256), environment dependencies, hardware specs, and execution commands.
- 📝 **[Final Completion Report & Decision Memo](results/report/final_completion_report.md)**: Synthesis answering the 7 key scientific decision questions.

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```powershell
# Clone repository
git clone https://github.com/Bhavi123-hacker/brain-mri-foundation-benchmark.git
cd brain-mri-foundation-benchmark

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Test Suites
```powershell
# Master benchmark integrity test suite (manifests, tensor caches, patient isolation)
pytest tests/test_master_benchmark_integrity.py -v

# Pre-publication statistical audit test suite
pytest tests/test_statistical_audit.py -v
```

### 3. Run Experiments & Audits
```powershell
# 1. Train and evaluate FOMO25 AMAES linear probe
python scripts/train_fomo25_linear_probe.py

# 2. Run label efficiency suite (10%, 25%, 50%, 100%) and paired bootstrap audit
python scripts/run_statistical_and_efficiency_audit.py

# 3. Compute Holm-Bonferroni multiple-comparison corrections
python scripts/compute_holm_bonferroni.py
```

---

## ⚖️ Ethical Disclaimers & Data Use Compliance

1. **Research Use Only**: This software and benchmark repository are intended strictly for computational academic research. They are **not certified medical devices** and must not be used for clinical diagnostic decision-making.
2. **Data Privacy**: No raw patient MRI scans, face-identifying voxels, or protected health information (PHI) are stored in this repository. All datasets (OASIS-1, ADNI, MIRIAD) are governed by their respective institutional Data Use Agreements.
3. **Model Checkpoint Licenses**:
   - MONAI Swin UNETR SSL: Apache 2.0 (Project-MONAI).
   - FOMO25 AMAES ResEnc-B: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA 4.0).
   - 2D ResNet-18: BSD 3-Clause (Torchvision).

---
*Repository maintained by Senior Medical Imaging ML Lead (October 2026).*
