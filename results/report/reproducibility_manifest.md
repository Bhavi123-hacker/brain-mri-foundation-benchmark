# Reproducibility & Environment Manifest
**Project**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines  
**Date**: October 21, 2026  
**Repository**: `https://github.com/Bhavi123-hacker/brain-mri-foundation-benchmark`  
**Local Workspace**: `D:\ml project`  

---

## 1. Hardware & System Configuration

- **GPU**: NVIDIA RTX PRO 2000 Blackwell (16,376 MiB VRAM, Architecture `sm_120`)
- **Driver / CUDA**: NVIDIA Studio Driver, CUDA Version 12.8
- **Host OS**: Microsoft Windows 11 Enterprise (x86_64)
- **Primary Data Mount**: `D:\` (`D:\ml project` and `D:\brain_mri_data`) mapped via `subst D: E:\`

---

## 2. Software & Python Environment

- **Python Interpreter**: `D:\ml project\.venv\Scripts\python.exe` (Python 3.12.10)
- **Core ML Libraries**:
  - `torch`: 2.11.0+cu128
  - `torchvision`: 0.26.0+cu128
  - `monai`: 1.4.0
  - `scikit-learn`: 1.6.1
  - `scipy`: 1.15.2
  - `numpy`: 2.2.3
  - `pandas`: 2.2.3
  - `nibabel`: 5.3.2
  - `pytest`: 9.1.1
  - `pytorch-lightning`: 2.5.0.post0

---

## 3. Model Weights, Checkpoints & Cryptographic Hashes

| Model | Local Weight File Path | Size (Bytes) | SHA-256 Checksum | Architecture / Parameters | License / Provenance |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **FOMO25 AMAES ResEnc-B** | `weights/resenc_unet_b.ckpt` | 1,167,183,049 | `115d509241ad3a421a851dda3a1e5d98ae8377973ea9b29bbb2c2a181c277749` | ResEnc-B Backbone (90,296,897 params) | CC-BY-NC-SA 4.0 / FOMO-MRI |
| **MONAI Swin UNETR SSL** | `weights/model_swinvit.pt` | 411,162,269 | `1cf19eca3ed0b4278c7c98692c8e3ca91438902506b3a3c915f7b4ee7be88f4e` | 3D SwinViT Backbone (8,062,771 params) | Apache 2.0 / Project-MONAI |
| **2D Transfer ResNet-18** | Torchvision Cache | N/A | Standard PyTorch hub download | ResNet-18 (11,177,025 params) | BSD-3-Clause / ImageNet-1K |
| **3D ResNet-18 (Scratch)**| Checkpoint from training | ~132 MB | Generated during training run | 3D ResNet-18 (33,160,513 params) | Open Research / MedicalNet init |
| **3D ResNet-34 (Scratch)**| Checkpoint from training | ~254 MB | Generated during training run | 3D ResNet-34 (63,489,217 params) | Open Research / MedicalNet init |
| **FOMO Linear Probe Model**| `results/fomo_linear_probe/fomo25_model.pkl` | 9,976 | Stored fitted model | $L_2$ Logistic Regression ($C=0.001$, 321 params) | MIT / Fitted on OASIS Train |

---

## 4. Cohort Manifests & Tensor Caches

| Cohort | Manifest Path | Eligible Count | Preprocessed Tensor Path | Grid Resolution |
| :--- | :--- | :---: | :--- | :--- |
| **OASIS-1 Primary** | `D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv` | 198 (138 tr, 30 val, 30 test) | `D:\brain_mri_data\OASIS1\processed\tensors` | $128 \times 128 \times 128$ isotropic (1.25 mm) |
| **ADNI External** | `D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv` | 500 (303 CN, 197 AD) | `D:\brain_mri_data\ADNI\processed\tensors` | $128 \times 128 \times 128$ isotropic (1.25 mm) |
| **MIRIAD Template** | `D:\ml project\results\miriad_external_validation\miriad_manifest_template.csv` | 69 (23 CN, 46 AD) | `D:\brain_mri_data\MIRIAD\processed\tensors` | Template ready for user credentialing |

---

## 5. Exact Execution Commands

### A. Preprocessing & Data Validation
```powershell
# Verify OASIS-1 and ADNI manifest and cache integrity
& "D:\ml project\.venv\Scripts\python.exe" -m pytest "D:\ml project\tests\test_master_benchmark_integrity.py" -v
```

### B. FOMO25 AMAES Feature Extraction & Linear Probing
```powershell
# Extract bottleneck representations and train linear probe
& "D:\ml project\.venv\Scripts\python.exe" "D:\ml project\scripts\train_fomo25_linear_probe.py"
```

### C. Label Efficiency Suite & Paired Statistical Bootstrap
```powershell
# Run label efficiency (10%, 25%, 50%, 100%) and paired bootstrap resampling
& "D:\ml project\.venv\Scripts\python.exe" "D:\ml project\scripts\run_statistical_and_efficiency_audit.py"
```

### D. Holm-Bonferroni Multiple-Comparison Correction
```powershell
# Run Holm-Bonferroni correction across all 15 pairwise comparisons
& "D:\ml project\.venv\Scripts\python.exe" "D:\ml project\scripts\compute_holm_bonferroni.py"
```

### E. Master Benchmark Integrity & Statistical Test Suite
```powershell
Set-Location "D:\ml project"
& "D:\ml project\.venv\Scripts\python.exe" -m pytest "tests\test_master_benchmark_integrity.py" -v
& "D:\ml project\.venv\Scripts\python.exe" -m pytest "tests\test_statistical_audit.py" -v
```

---
*Manifest generated and verified on October 21, 2026.*
