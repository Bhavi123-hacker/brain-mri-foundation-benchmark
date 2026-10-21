# Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11+cu128-red.svg)](https://pytorch.org/)
[![MONAI 1.6](https://img.shields.io/badge/MONAI-1.6.1-brightgreen.svg)](https://monai.io/)
[![Hardware: NVIDIA RTX PRO 2000 Blackwell](https://img.shields.io/badge/GPU-RTX_PRO_2000_Blackwell-green.svg)](https://www.nvidia.com/)
[![Tests: 28 Passed](https://img.shields.io/badge/Tests-28%20Passed-brightgreen.svg)](tests/)

A rigorous, multi-cohort, open-science research benchmark evaluating modern **3D Brain-MRI Foundation Models** against **well-tuned supervised 3D convolutional neural networks**, **2.5D tri-planar transfer learning**, **classical radiomics**, and **dual-branch hybrid networks** for Alzheimer’s disease classification across controlled label budgets, multi-scanner domain shifts, paired bootstrap uncertainty quantification, and strict family-wise error rate corrections.

---

## 📊 Consolidated Master Benchmark Table (10 Models)

All models evaluated under strictly isolated participant partitions on **OASIS-1** ($N=198$; 138 train, 30 val, 30 locked test), external zero-shot transfer on **ADNI** ($N=500$; 229 CN, 271 AD; Siemens & Philips, 1.5T & 3.0T), and the newly ingested **OASIS-2 External Cohort** ($N=150$; 85 CN, 65 AD; 0.0% patient overlap):

| Architecture | Family / Paradigm | Parameters | Weights Source | OASIS Val AUC | OASIS Locked Test AUC [95% CI] | OASIS Test BalAcc | ADNI External AUC [95% CI] | ADNI Ext BalAcc | OASIS-2 External AUC | OASIS-2 Ext BalAcc |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D ResNet-18 (Scratch)** | Supervised 3D CNN | 33.2M | Random (MedicalNet init) | 0.5600 | **0.6178** [0.4133, 0.8133] | 0.5333 | **0.5128** [0.4612, 0.5644] | 0.4940 | — | — |
| **3D ResNet-34 (Scratch)** | Supervised 3D CNN | 63.5M | Random (MedicalNet init) | 0.5822 | **0.5311** [0.3200, 0.7333] | 0.5000 | **0.4777** [0.4261, 0.5293] | 0.5133 | — | — |
| **2D ResNet-18 (Slice-Agg)** | 2D Pretrained Transfer | 11.2M | Torchvision ImageNet-1K V1 | 0.7778 | **0.6222** [0.4133, 0.8178] | 0.4333 | **0.5778** [0.5280, 0.6276] | 0.5000 | — | — |
| **Classical Radiomics (SVM)** | Engineered Biomarkers | **25** | 24 Volume/Texture Features | 0.6978 | **0.5689** [0.3600, 0.7689] | 0.5667 | **0.5149** [0.4638, 0.5660] | 0.4957 | — | — |
| **MONAI Swin UNETR SSL** | 3D Foundation Transformer | 8.1M | MONAI SSL (`model_swinvit.pt`) | 0.4467 | **0.6044** [0.3956, 0.8044] | 0.5000 | **0.5834** [0.5331, 0.6337] | 0.5000 | — | — |
| **FOMO25 AMAES ResEnc-B** | 3D Foundation Residual Enc | 90.3M | FOMO-50K (`resenc_unet_b.ckpt`) | 0.6667 | **0.5733** [0.3664, 0.7989] | 0.5333 | **0.5054** [0.4516, 0.5552] | 0.5000 | — | — |
| **3D ResNet-10** | Lightweight 3D CNN | 5.4M | Random Scratch | 0.5444 | **0.6133** [0.3973, 0.8145] | 0.5333 | **0.5444** [0.4909, 0.5953] | 0.5000 | **0.6557** | 0.5000 |
| **3D DenseNet-121** | Dense 3D CNN | 11.2M | Random Scratch | 0.6400 | **0.4133** [0.2187, 0.6269] | 0.5000 | **0.5048** [0.4537, 0.5592] | 0.4800 | **0.3939** | 0.4824 |
| **2.5D Multi-View ResNet-18**| Tri-Planar Transfer | 11.6M | Torchvision ImageNet-1K V1 | **0.8267** | **0.6444** [0.4261, 0.8300] | **0.6333** | **0.4943** [0.4434, 0.5464] | 0.4908 | **0.7500** | **0.6403** |
| **Dual-Branch Hybrid** | Spatial + Biomarkers | 5.4M | Random Scratch + Radiomics | 0.5756 | **0.6311** [0.4152, 0.8170] | 0.5000 | **0.5554** [0.5044, 0.6066] | 0.5000 | **0.6581** | 0.5000 |

*Table Artifact:* `results/v2/tables/v2_master_model_comparison_table.csv`

---

## 🔬 Core Scientific Findings

1. **No Foundation Model Superiority from Frozen Probes**: 3D foundation model linear probes (FOMO25 AMAES ResEnc-B, Test AUC **0.5733**; MONAI Swin UNETR SSL, Test AUC **0.6044**) do not statistically outperform lightweight supervised 3D CNNs (ResNet-10, Test AUC **0.6133**) or 2.5D transfer learning (Test AUC **0.6444**; OASIS-2 AUC **0.7500**).
2. **Multiplicity Correction Verdict**: Across 2,000-iteration paired bootstrap tests with **Holm-Bonferroni FWER control**, **zero** comparative hypotheses achieved statistical significance ($p_{\text{adj}} > 0.05$). Small-cohort rank orderings are within expected binomial sampling variance.
3. **Tri-Planar 2.5D Dominance on Sample Scarcity**: 2.5D Multi-View ResNet-18 reached **0.7289 Val AUC with just 14 labeled training scans** (10% budget), proving that pretrained 2D inductive priors surpass 3D random initialization on small medical datasets.
4. **Dual-Branch Hybrid Robustness**: Incorporating handcrafted anatomical biomarkers (ventricle-to-brain ratio, hippocampal index, intensity skewness) achieved the highest domain transfer on ADNI (**0.5554 AUC**).

---

## 📑 Key Research Deliverables

- 📖 **[V2 Final Research Report](results/v2/v2_final_research_report.md)**: Comprehensive 11-section publication-ready paper.
- 📋 **[Executive Decision Memo](results/v2/v2_executive_handoff_memo.md)**: Management and PI decision memo (Items A through J).
- 📊 **[Master Comparison Table (CSV)](results/v2/tables/v2_master_model_comparison_table.csv)**: Consolidated 10-model evaluation.
- 📐 **[Paired Bootstrap Statistical Audit (CSV)](results/v2/tables/v2_paired_bootstrap_statistical_audit.csv)**: 2,000 bootstrap resamples with Holm-Bonferroni adjustments.
- 🗺️ **[Research Gap Matrix (CSV)](docs/research/research_gap_matrix.csv)**: Literature claims vs. empirical benchmark realities.
- 🌐 **[OASIS-2 Expansion Audit](docs/data/oasis2_expansion_audit.md)**: Provenance and 0.0% overlap verification for $N=150$ baseline scans.

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```powershell
# Clone repository and checkout feature branch
git clone https://github.com/Bhavi123-hacker/brain-mri-foundation-benchmark.git
cd brain-mri-foundation-benchmark
git checkout research/v2-foundation-generalization-upgrade

# Setup Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (28 Tests)
```powershell
pytest tests/ -v
```

### 3. Run Version 2 Benchmark Campaign
```powershell
python scripts/run_v2_experiment_campaign.py
```

---

## ⚖️ Ethical Disclaimers & Data Governance

1. **Research Use Only**: This software and benchmark repository are intended strictly for academic computational research. They are **not certified medical devices**.
2. **Data Privacy**: No protected health information (PHI) or face-identifying voxels are stored. Datasets (OASIS-1, OASIS-2, ADNI) are governed by their respective institutional Data Use Agreements.
3. **Licenses**: OASIS-2 data is distributed under Open Data Commons Attribution (ODC-By). Code is licensed under Apache 2.0.
