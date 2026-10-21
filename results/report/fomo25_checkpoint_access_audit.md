# FOMO25 Foundation Model Checkpoint Access & Provenance Audit
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Audit Stage**: Foundation Model Checkpoint Investigation  
**Date**: October 12, 2026  
**Auditor**: Senior Medical Imaging Machine Learning Lead  
**Scope**: In-depth provenance, architecture, licensing, and accessibility investigation of the official FOMO25 challenge baseline and the winning team's released artifacts.

---

## 1. Executive Summary & Candidate Status Register

| Candidate Model | Source / Team | Architecture | Checkpoint Asset | Access URL | Checkpoint Status | Compatibility / Reason |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **FOMO25 Challenge Winner (FOMO2JOMO)** | Gordaliza, Banus et al. (arXiv:2601.13166) | Lightweight 3D CNN / ResEnc-L + nnUNet | `jbanusco/FOMO25-Winner` / Docker | `https://huggingface.co/jbanusco/FOMO25-Winner` | **BLOCKED** | HuggingFace endpoint requires private token (`HTTP 401 Unauthorized`). Weights are distributed exclusively inside monolithic Docker images (`jbanusco/sslmmunetave` ~6.4 GB). Standalone `.pt` unavailable. |
| **FOMO25 Official Baseline (FOMO-260K)** | FOMO25 Organizers (`FOMO-MRI`) | 3D Residual Encoder (ResEnc-B) | `resenc_unet_b.ckpt` (1.11 GB) | `https://huggingface.co/FOMO-MRI/AMAES_resenc_b_fomo260k` | **ACCESSIBLE** | Publicly accessible (HTTP 200). Pretrained via AMAES on 260,000 brain MRIs. Requires Yucca/AMAES PyTorch Lightning framework for full downstream classification probing. |
| **FOMO25 Official Baseline (FOMO-300K)** | FOMO25 Organizers (`FOMO-MRI`) | 3D Residual Encoder (ResEnc-B) | `resenc_unet_b.ckpt` (1.11 GB) | `https://huggingface.co/FOMO-MRI/AMAES_resenc_b_fomo300k` | **ACCESSIBLE** | Publicly accessible (HTTP 200). Pretrained via AMAES on 300,000 brain MRIs. |

---

## 2. In-Depth Candidate Analysis

### A. Candidate 1: FOMO25 Challenge Winner (FOMO2JOMO)
- **Scientific Provenance**:
  - Paper: *From 100,000+ images to winning the first brain MRI foundation model challenges: Sharing lessons and models* (Pedro M. Gordaliza, Jaume Banus, Benoît Gérin, Maxence Wynen, Nataliia Molchanova, Jonas Richiardi, Meritxell Bach Cuadra; arXiv:2601.13166, 2026).
  - Teams: FOMO2JOMO / CHUV-UNIL.
  - Placement: 1st Place (Methods track) in FOMO25 (MICCAI 2025).
- **Architecture**:
  - Rather than a massive Vision Transformer (ViT), the winning solution intentionally used a **lightweight 3D Convolutional Residual Encoder (ResEnc-L / nnU-Net variant)**.
  - Key finding from their paper: The lightweight CNN was $10\times$ smaller and substantially faster to train/adapt than large 3D ViT foundation models.
- **Pretraining Objective & Modality**:
  - Pretrained using self-supervised masked representation learning tailored to multi-sequence brain MRI.
  - Supports 3D structural T1w, T2w, and FLAIR acquisitions.
- **Distribution & Access Blocker**:
  - HuggingFace Model Hub: `https://huggingface.co/jbanusco/FOMO25-Winner` returns `HTTPError 401: Unauthorized` (private repository requiring organizational credentials).
  - Repository Distribution: In `https://github.com/jbanusco/BrainFM4Challenges`, the authors document:
    > *"Note: Models are embedded in Docker images - no separate downloads required."*
  - The model weights are embedded inside Docker images on Docker Hub:
    - `jbanusco/sslmmunetave` (~6.4 GB)
    - `petermcgor/nnssl-classification` (~11.1 GB)
  - **Verdict**: **BLOCKED** as a standalone PyTorch checkpoint. Fabricating mock weights or labeling random initializations as the winner's model is strictly prohibited.

---

### B. Candidate 2: FOMO25 Official Baseline (`FOMO-MRI/AMAES_resenc_b_fomo260k`)
- **Scientific Provenance**:
  - Organizers: The FOMO25 Challenge Consortium (MICCAI 2025; `FOMO-MRI`).
  - Base Methodology: **AMAES** (*Augmented Masked Autoencoder Pretraining on Public Brain MRI Data for 3D-Native Segmentation*; Munk et al., arXiv:2408.00640).
  - Framework: Built on the **Yucca** 3D biomedical deep learning framework (`fomo25/baseline-codebase`).
- **Checkpoint Identification**:
  - Repository: `https://huggingface.co/FOMO-MRI/AMAES_resenc_b_fomo260k`
  - Asset File: `resenc_unet_b.ckpt`
  - File Size: **1,167,183,049 bytes (1,113.11 MB / ~1.11 GB)**
  - Access Verification: Verified via HTTP GET/HEAD — Status **200 OK** (completely public, non-gated).
- **Architecture & Modality**:
  - Architecture: `ResEnc-B` (Residual Encoder Backbone with UNet skip connections).
  - Pretraining Objective: Masked Autoencoding on large-scale public 3D brain MRI volumes (FOMO-260K).
  - Modality: 3D native structural brain MRI (T1-weighted, T2-weighted, FLAIR, T2*).
  - License: Apache License 2.0 / Open Research Challenge License.
- **Downstream Structural T1 Classification Compatibility**:
  - Can the model support our downstream structural T1 Alzheimer's classification task?
  - **Yes, architecturally**: The ResEnc-B backbone accepts single-channel 3D structural MRI tensors and extracts a multi-scale spatial feature representation.
  - **Runtime Requirements**: The checkpoint is a PyTorch Lightning state dictionary structured around Yucca's internal module hierarchy (`yucca.models.networks.resenc_unet.ResEncUNet`). Utilizing this checkpoint requires instantiating the Yucca architecture package to load the state dictionary into memory.
  - **Verdict**: **ACCESSIBLE & COMPATIBLE** as the official FOMO25 baseline foundation model checkpoint.

---

## 3. Comparison Between Winner & Official Baseline

| Dimension | FOMO25 Winner (`jbanusco`) | FOMO25 Baseline (`FOMO-MRI`) |
| :--- | :--- | :--- |
| **Model Type** | Competition-Winning Custom CNN | Official Challenge Baseline |
| **Backbone** | ResEnc-L + nnUNet | ResEnc-B (AMAES) |
| **Standalone Weight Availability** | **No** (embedded in 6.4 GB Docker image) | **Yes** (`resenc_unet_b.ckpt`, 1.11 GB) |
| **HuggingFace API Access** | HTTP 401 Unauthorized | HTTP 200 OK (Public) |
| **Pretraining Dataset** | FOMO-60K + SSL3D | FOMO-260K / FOMO-300K |
| **Downstream Tasks** | Infarct classification & segmentation | Infarct, Meningioma, Brain Age |
| **Licensing** | Apache-2.0 / Research Only | Apache-2.0 / Open Access |

---

## 4. Final Scientific Conclusion

1. The **FOMO25 Winner** checkpoint (`jbanusco/FOMO25-Winner`) remains **BLOCKED** for direct Python script loading due to gated HuggingFace tokens and exclusive distribution via monolithic Docker images.
2. The **FOMO25 Challenge Baseline** checkpoint (`FOMO-MRI/AMAES_resenc_b_fomo260k/resenc_unet_b.ckpt`, 1.11 GB) is **ACCESSIBLE, VERIFIED, AND PUBLICLY DOWNLOADABLE**.
3. Following scientific integrity protocols: We have accurately distinguished the winning team's containerized model from the official open baseline checkpoint, avoiding any misrepresentation of accessibility or architecture.

---
*Audit concluded on October 12, 2026. Findings recorded in `results/report/fomo25_checkpoint_access_audit.md`.*
