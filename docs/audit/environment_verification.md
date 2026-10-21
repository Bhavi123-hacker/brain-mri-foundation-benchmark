# System & Environment Verification Report
**Date**: October 21, 2026  
**Auditor**: Principal Investigator & Research Infrastructure Lead  
**Scope**: Full validation of hardware, operating system, CUDA, driver, virtual environment, and deep learning library stacks.

---

## 1. Hardware Specification
- **GPU Device**: NVIDIA RTX PRO 2000 Blackwell
- **Architecture**: `sm_120` (Compute Capability 12.0)
- **VRAM Total**: 15.93 GB (16,310 MiB)
- **Host OS**: Microsoft Windows 11 Enterprise (x86_64)
- **Storage Substituted Drives**: Drive `D:\` is mapped to `E:\` via `subst D: E:\`.
  - Used: ~29.9 GB
  - Free: ~311.9 GB

---

## 2. Python Environment & Core Dependencies
- **Interpreter Path**: `D:\ml project\.venv\Scripts\python.exe`
- **Python Version**: 3.12.10 (64-bit)

### Verified Library Versions:
| Library | Installed Version | Verification Status |
| :--- | :---: | :---: |
| `torch` | `2.11.0+cu128` | **VERIFIED** (CUDA 12.8 enabled) |
| `torchvision` | `0.26.0+cu128` | **VERIFIED** |
| `monai` | `1.6.1` | **VERIFIED** |
| `SimpleITK` | `2.5.6` | **VERIFIED** |
| `nibabel` | `5.4.2` | **VERIFIED** |
| `scikit-learn` | `1.9.1` | **VERIFIED** |
| `scipy` | `1.15.2` | **VERIFIED** |
| `numpy` | `2.2.3` | **VERIFIED** |
| `pandas` | `2.2.3` | **VERIFIED** |
| `pytest` | `9.1.1` | **VERIFIED** |
| `pytorch-lightning` | `2.5.0.post0` | **VERIFIED** |

---

## 3. Path Portability and Drive Mapping Note
The workspace uses `D:\ml project` and `D:\brain_mri_data`. Because `D:` is mapped via `subst D: E:\`, hardcoded absolute paths to `D:\` work reliably within the current environment. However, to guarantee cross-platform portability on POSIX systems or unmapped environments, all new code in the `v2` campaign must accept configurable dataset roots via environment variables (`BRAIN_MRI_DATA_DIR` and `BENCHMARK_PROJECT_ROOT`) with fallback to relative paths and local configuration YAML files.
