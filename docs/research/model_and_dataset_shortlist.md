# Model Architecture & Dataset Expansion Shortlist (Version 2)
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Date**: October 21, 2026  
**Auditors**: Senior Deep Learning Lead, Data Engineering Lead & Scientific Literature Researcher  

---

## 1. Prioritized Model Architecture Shortlist

Based on GPU memory constraints (16 GB VRAM on NVIDIA RTX PRO 2000 Blackwell), code availability, authentic weights accessibility, and scientific relevance, we prioritize models into three distinct tiers:

### Tier 1 — Core High-Value Executable Models (Immediate Implementation & Benchmarking)
| Model Identifier | Architecture Family | Parameter Count | Pretrained Source / Weights | Implementation Strategy | GPU Memory / Compute Viability | Scientific Rationale |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Model 1: 3D ResNet-18 (Scratch vs MedicalNet)** | 3D Residual CNN | 33,160,513 | Random init vs MedicalNet (Tencent Med3D, 23 medical datasets) | MONAI `resnet18` (spatial_dims=3); test random vs authentic Med3D initialization | ~4.2 GB VRAM at batch_size=2; highly feasible | Tests Hypothesis H1 directly: does authentic multi-dataset 3D medical pretraining outperform random initialization? |
| **Model 2: 3D DenseNet-121** | 3D Densely Connected CNN | 11,213,249 | Random init (He normal) | MONAI `DenseNet121` (spatial_dims=3, in_channels=1, out_channels=1) | ~6.8 GB VRAM with mixed precision; feasible | Compact parameter footprint (~11.2M), rich feature reuse through dense connections, highly effective in medical imaging. |
| **Model 3: 3D ResNet-10** | Compact 3D Residual CNN | 14,411,457 | Random init | MONAI `resnet10` (spatial_dims=3) | ~2.5 GB VRAM; extremely fast | Tests Hypothesis H2: does lowering 3D model capacity prevent catastrophic overfitting in sample-limited regimes ($N=138$)? |
| **Model 4: 2.5D Multi-View ResNet-18** | Multi-Planar Slice Aggregation | 11,177,025 | ImageNet-1K V1 (Torchvision) | Orthogonal slice extraction (Axial, Coronal, Sagittal central slices), shared 2D ResNet-18 backbone, late fusion head | ~1.8 GB VRAM; very fast | Tests Hypothesis H3: does multi-planar 2.5D visual transfer outperform volumetric 3D scratch models with lower compute? |

### Tier 2 — Foundation & Modern Self-Supervised Models
| Model Identifier | Architecture Family | Parameter Count | Pretrained Source / Weights | Implementation Strategy | GPU Memory / Compute Viability | Scientific Rationale |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Model 5: MONAI Swin UNETR SSL** | 3D Shifted-Window ViT | 8,062,771 (backbone) | `weights/model_swinvit.pt` (Project-MONAI, CVPR 2022) | Frozen linear probe vs partial fine-tuning of top 2 stages | ~5.5 GB VRAM (linear probe: ~1.2 GB); verified | Benchmark gold standard for 3D hierarchical self-supervised transformers; evaluated with authentic weights. |
| **Model 6: FOMO25 AMAES ResEnc-B** | 3D Residual Encoder | 90,296,897 (encoder) | `weights/resenc_unet_b.ckpt` (FOMO-260K, arXiv:2408.00640) | Frozen linear probe vs parameter-efficient adapter head | ~7.8 GB VRAM (linear probe: ~2.1 GB); verified | Official FOMO25 challenge baseline pre-trained on 260,000 multi-sequence brain MRIs. |
| **Model 7: ConvNeXt-Tiny (2D Transfer)** | Modern Depthwise Convolutional 2D | 28,629,057 | ImageNet-1K (Torchvision) | 16-slice axial aggregation, average pooling, linear classification head | ~2.4 GB VRAM; feasible | Tests modern 2D architectural improvements (7x7 depthwise convolutions, inverted bottleneck) against classical ResNet. |
| **Model 8: DINOv2 ViT-S/14 (2D SSL)** | Self-Supervised 2D ViT | 22,056,576 | Meta AI DINOv2 (`dinov2_vits14`) | 16-slice axial feature extraction, frozen token pooling, linear probe | ~3.1 GB VRAM; feasible | Evaluates whether 2D self-supervised visual representations extract better subtle brain texture than supervised ImageNet. |

### Tier 3 — Anatomical & Hybrid Fusion Models
| Model Identifier | Architecture Family | Parameter Count | Feature Inputs | Implementation Strategy | GPU Memory / Compute Viability | Scientific Rationale |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Model 9: Regional Anatomical Radiomics** | Morphometric Machine Learning | 25 - 50 | 24 - 40 Volumetric & Texture Features (Hippocampus, Ventricles, Cortex) | FastSurfer / Morphometric feature extraction + L2 Logistic Regression / SVM / XGBoost | CPU execution (< 100 MB RAM); instant | Pure anatomical baseline; unassailable biological grounding invariant to deep CNN texture shortcuts. |
| **Model 10: Dual-Branch Hybrid Fusion** | Multimodal Deep + Anatomical Fusion | ~11.2M + 30 | 3D DenseNet-121 / ResNet-18 + Regional Radiomics | Feature concatenation of global spatial embedding (128-d) and anatomical radiomics (24-d) | ~7.0 GB VRAM; feasible | Tests Hypothesis H5: does fusing global deep features with macroscopic morphometric biomarkers create optimal synergy? |

---

## 2. Prioritized Dataset Expansion Shortlist

| Cohort Identifier | Source & Center | Eligible Subjects ($N$) | Scanner Hardware | Ground Truth Label | Data Access & Licensing Status | Priority & Feasibility Verdict |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **MIRIAD** | Dementia Research Centre, UCL (London, UK) | 69 (46 AD, 23 CN) | 1.5T General Electric (GE) Signa | Clinical Probable AD vs Healthy Elderly | MIRIAD DUA via UCL XNAT (redistribution prohibited) | **PRIORITY 1 (RECOMMENDED)**: Introduces third scanner vendor (GE); 0.0% overlap; automated pipeline ready; pending user registration. |
| **OASIS-2** | Washington University ADRC (St. Louis, USA) | 150 (72 CN, 64 AD, 14 Conv) | 1.5T Siemens Vision | Longitudinal CDR (0.0 vs $\ge 0.5$) | Open Access (Creative Commons) | **PRIORITY 2 (FALLBACK)**: Open access; identical hardware to OASIS-1; requires participant de-duplication; tests longitudinal stability. |
| **ADNI Extended Baseline** | ADNI Consortium (Multi-Center) | Additional ~300 eligible scans | 1.5T & 3.0T Siemens/Philips/GE | Clinical AD vs CN vs MCI | Approved under existing ADNI DUA | **PRIORITY 3 (IN-COHORT EXPANSION)**: Can expand external cohort from 500 to ~800 subjects under approved DUA without cross-border friction. |
| **AIBL** | CSIRO / Univ of Melbourne (Australia) | ~1,000 | 1.5T & 3.0T Multi-vendor + Amyloid PET | Clinical AD, MCI, CN + Amyloid | Gated (LONI IDA DUA committee approval) | **DEFERRED**: 2-3 week institutional review committee turnaround. |
| **OASIS-3** | Washington University ADRC (St. Louis, USA) | 1,378 | 1.5T & 3.0T Siemens Trio/Prisma | Longitudinal CDR + Biomarkers | Gated (XNAT Central Data Transfer Agreement) | **DEFERRED**: Requires signed DTA and XNAT Central registration. |
| **FOMO45K / FOMO50K** | FOMO Challenge Consortium | 11,966 / 50,000 | Multi-sequence MRI | Tumors, Strokes, Controls | Open Access / Gated | **DISQUALIFIED**: Zero AD cases (FOMO45K); direct OASIS-1 test leakage hazard (`PT001_OASIS1` in FOMO50K). |
| **HuggingFace `Falah/Alzheimer_MRI`** | Kaggle Mirror via Hugging Face | 6,400 images | Unspecified | 2D Slice Categories | Open Access | **DISQUALIFIED**: Uncurated 2D JPG slices without 3D volumetric geometry, patient IDs, or slice-leakage prevention. |

---
*Shortlist ratified on October 21, 2026. Models 1, 2, 3, 4, 9, 10 scheduled for implementation in Version 2.*
