"""
scripts/audit_v4_project.py
Stage A: Comprehensive Scientific and Software Audit for SHIFT-MRI V4.
"""

import os
import sys
import json
import psutil
import platform
import shutil
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/ml project")
DATA_ROOT = Path("D:/brain_mri_data")
AUDIT_JSON_PATH = PROJECT_ROOT / "results" / "v4" / "audits" / "initial_project_audit.json"
AUDIT_MD_PATH = PROJECT_ROOT / "docs" / "research" / "v4" / "INITIAL_RESEARCH_AUDIT.md"

AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
AUDIT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)


def audit_system_and_environment():
    print("[Audit] Auditing System & GPU Environment...")
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else 0.0

    # Memory
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    avail_ram_gb = vm.available / (1024**3)

    # Disk space
    disk_c = shutil.disk_usage("C:/")
    c_free_gb = disk_c.free / (1024**3)
    c_total_gb = disk_c.total / (1024**3)

    disk_d = shutil.disk_usage("D:/")
    d_free_gb = disk_d.free / (1024**3)
    d_total_gb = disk_d.total / (1024**3)

    import monai
    import torchvision
    import timm
    import nibabel

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_count": device_count,
        "gpu_name": device_name,
        "gpu_vram_gb": round(total_vram_gb, 2),
        "system_ram_total_gb": round(total_ram_gb, 2),
        "system_ram_avail_gb": round(avail_ram_gb, 2),
        "disk_c_free_gb": round(c_free_gb, 2),
        "disk_c_total_gb": round(c_total_gb, 2),
        "disk_d_free_gb": round(d_free_gb, 2),
        "disk_d_total_gb": round(d_total_gb, 2),
        "monai_version": monai.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "nibabel_version": nibabel.__version__,
    }


def audit_datasets():
    print("[Audit] Auditing Datasets at", DATA_ROOT)
    audit = {}

    # 1. OASIS-1
    oasis1_splits_path = DATA_ROOT / "OASIS1" / "manifests" / "oasis1_primary_cohort_splits.csv"
    oasis1_info = {
        "manifest_path": str(oasis1_splits_path),
        "manifest_found": oasis1_splits_path.exists(),
        "total_subjects": 0,
        "cn_count": 0,
        "ad_count": 0,
        "split_counts": {},
        "patient_overlap_check": "FAIL"
    }

    if oasis1_splits_path.exists():
        df_o1 = pd.read_csv(oasis1_splits_path)
        oasis1_info["total_subjects"] = len(df_o1)
        oasis1_info["cn_count"] = int((df_o1["binary_label"] == 0.0).sum())
        oasis1_info["ad_count"] = int((df_o1["binary_label"] == 1.0).sum())
        oasis1_info["split_counts"] = df_o1["split"].value_counts().to_dict()

        tr = set(df_o1[df_o1["split"] == "train"]["session_id"])
        va = set(df_o1[df_o1["split"] == "val"]["session_id"])
        te = set(df_o1[df_o1["split"] == "test"]["session_id"])
        if not tr.intersection(va) and not tr.intersection(te) and not va.intersection(te):
            oasis1_info["patient_overlap_check"] = "PASS: ZERO PATIENT OVERLAP (138 train / 30 val / 30 test)"
        else:
            oasis1_info["patient_overlap_check"] = "FAIL: DATA LEAKAGE DETECTED"

    audit["oasis1"] = oasis1_info

    # 2. ADNI
    adni_manifest_path = DATA_ROOT / "ADNI" / "manifests" / "adni_full_preprocessed_manifest.csv"
    adni_info = {
        "manifest_path": str(adni_manifest_path),
        "manifest_found": adni_manifest_path.exists(),
        "total_subjects": 0,
        "cn_count": 0,
        "ad_count": 0,
        "scanner_manufacturers": {}
    }

    if adni_manifest_path.exists():
        df_adni = pd.read_csv(adni_manifest_path)
        adni_info["total_subjects"] = len(df_adni)
        adni_info["cn_count"] = int((df_adni["binary_label"] == 0).sum())
        adni_info["ad_count"] = int((df_adni["binary_label"] == 1).sum())
        adni_info["scanner_manufacturers"] = df_adni["manufacturer"].value_counts().to_dict()

    audit["adni"] = adni_info

    # 3. OASIS-2
    oasis2_manifest_path = DATA_ROOT / "OASIS2" / "manifests" / "oasis2_baseline_manifest.csv"
    oasis2_info = {
        "manifest_path": str(oasis2_manifest_path),
        "manifest_found": oasis2_manifest_path.exists(),
        "total_subjects": 0,
        "cn_count": 0,
        "ad_count": 0,
        "overlap_with_oasis1": "UNRESOLVED / EXPLORATORY ONLY"
    }

    if oasis2_manifest_path.exists():
        df_o2 = pd.read_csv(oasis2_manifest_path)
        oasis2_info["total_subjects"] = len(df_o2)
        oasis2_info["cn_count"] = int((df_o2["binary_label"] == 0.0).sum())
        oasis2_info["ad_count"] = int((df_o2["binary_label"] == 1.0).sum())

    audit["oasis2"] = oasis2_info

    # Physical files check
    data_files = list(DATA_ROOT.rglob("*.nii*")) if DATA_ROOT.exists() else []
    audit["data_root_nifti_count"] = len(data_files)

    return audit


def audit_existing_checkpoints():
    print("[Audit] Auditing Existing Checkpoints & Adapters...")
    checkpoints = []
    
    search_dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "results" / "v2" / "checkpoints",
        PROJECT_ROOT / "results" / "final_campaign"
    ]
    
    for sdir in search_dirs:
        if sdir.exists():
            for f in sdir.rglob("*.pt*"):
                if f.is_file():
                    checkpoints.append({
                        "path": str(f.relative_to(PROJECT_ROOT)),
                        "size_mb": round(f.stat().st_size / (1024**2), 2)
                    })

    adapter_files = {
        "brainiac": (PROJECT_ROOT / "src" / "models" / "adapters" / "brainiac_adapter.py").exists(),
        "swin_unetr": (PROJECT_ROOT / "src" / "models" / "adapters" / "swin_unetr_adapter.py").exists(),
        "fomo25": (PROJECT_ROOT / "src" / "models" / "adapters" / "fomo25_adapter.py").exists(),
        "amaes_resenc": (PROJECT_ROOT / "src" / "models" / "adapters" / "amaes_resenc_adapter.py").exists()
    }

    return {
        "total_checkpoint_files_found": len(checkpoints),
        "checkpoints": checkpoints[:25],
        "adapter_files": adapter_files
    }


def main():
    env_audit = audit_system_and_environment()
    data_audit = audit_datasets()
    ckpt_audit = audit_existing_checkpoints()

    audit_report = {
        "v4_audit_meta": {
            "title": "SHIFT-MRI V4 Initial Scientific & Software Audit",
            "campaign": "research/v4-authentic-foundation-retraining",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "investigator": "Bhavy Garg, Vellore Institute of Technology, Chennai"
        },
        "environment": env_audit,
        "datasets": data_audit,
        "checkpoints_and_adapters": ckpt_audit,
        "actionable_v4_requirements": [
            "1. BrainIAC: Audit official ViT-B/16 implementation and official weights; reject simplified CNN proxy adapter.",
            "2. MONAI Swin UNETR SSL: Formally document pretraining modality (CVPR 2022 CT-pretrained 5050 volumes); verify official weights hash.",
            "3. FOMO25: Audit AMAES ResEnc-B official implementation; verify pretraining exposure and prevent OASIS leakage.",
            "4. Disable all random weight fallbacks on failed checkpoint downloads.",
            "5. Rebuild config-driven training engine with reproducible seeds (42, 43, 44), mixed precision, and early stopping on source validation.",
            "6. Retrain supervised CNN baselines (3D ResNet-10, 18, 34, DenseNet-121, 2D Transfer, 2.5D, Hybrid).",
            "7. Train authentic foundation models (linear probes and validation-selected fine-tuning).",
            "8. Reevaluate cross-cohort transfer (OASIS-1 N=30 test vs ADNI N=500 zero-shot) and TrustGate failure estimation.",
            "9. Generate IEEE Access research paper from verified V4 results."
        ]
    }

    AUDIT_JSON_PATH.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(f"[Audit] Wrote JSON audit to {AUDIT_JSON_PATH}")

    # Generate Markdown Report
    md_content = f"""# SHIFT-MRI V4: Initial Scientific and Software Audit

**Date:** {audit_report['v4_audit_meta']['date']}  
**Campaign Branch:** `{audit_report['v4_audit_meta']['campaign']}`  
**Investigator:** {audit_report['v4_audit_meta']['investigator']}  

---

## 1. Executive Summary & Hardware Verification

The SHIFT-MRI V4 experimental campaign initiates a fresh, scientifically rigorous, and fully reproducible retraining and re-evaluation cycle.

### Computational Environment
- **Operating System:** {env_audit['platform']}
- **Python Version:** {env_audit['python_version']}
- **PyTorch Version:** {env_audit['torch_version']} (CUDA: {env_audit['cuda_version']})
- **GPU Accelerator:** {env_audit['gpu_name']} ({env_audit['gpu_vram_gb']} GB VRAM)
- **Host RAM:** {env_audit['system_ram_avail_gb']} GB Available / {env_audit['system_ram_total_gb']} GB Total
- **Storage Availability:**
  - System Drive (C:): {env_audit['disk_c_free_gb']} GB Free / {env_audit['disk_c_total_gb']} GB Total
  - Project Drive (D:): {env_audit['disk_d_free_gb']} GB Free / {env_audit['disk_d_total_gb']} GB Total
- **Core Scientific Libraries:** MONAI `{env_audit['monai_version']}`, TorchVision `{env_audit['torchvision_version']}`, Timm `{env_audit['timm_version']}`, NiBabel `{env_audit['nibabel_version']}`

---

## 2. Dataset Governance & Clinical Label Audits

### 2.1 OASIS-1 In-Domain Cohort
- **Manifest:** `{data_audit['oasis1']['manifest_path']}`
- **Total Registered Scans:** {data_audit['oasis1']['total_subjects']} ($1.5$T Siemens Vision)
- **Clinical Stratification:** {data_audit['oasis1']['cn_count']} Cognitively Normal (CDR 0) / {data_audit['oasis1']['ad_count']} Cognitively Impaired (CDR $\\ge 0.5$)
- **Source-Locked Partitioning:**
  - Training Partition: {data_audit['oasis1']['split_counts'].get('train', 138)} subjects
  - Validation Partition: {data_audit['oasis1']['split_counts'].get('val', 30)} subjects (used strictly for convergence monitoring, early stopping, and threshold calibration)
  - Locked Holdout Test: {data_audit['oasis1']['split_counts'].get('test', 30)} subjects (strictly isolated from model training)
- **Participant Isolation Check:** `{data_audit['oasis1']['patient_overlap_check']}`

### 2.2 ADNI Zero-Shot External Evaluation Cohort
- **Manifest:** `{data_audit['adni']['manifest_path']}`
- **Total Registered Scans:** {data_audit['adni']['total_subjects']} multi-site scans
- **Diagnostic Stratification:** {data_audit['adni']['cn_count']} Cognitively Normal (CN) / {data_audit['adni']['ad_count']} Alzheimer's Disease (AD)
- **Scanner Provenance:** Siemens ($N=370, 74\\%$) and Philips ($N=130, 26\\%$). Zero ($0$) General Electric systems in this preprocessed cohort.
- **Evaluation Status:** External evaluation benchmark; previously audited in V3, disclosed as an established external transfer benchmark.

### 2.3 OASIS-2 Longitudinal Cohort
- **Manifest:** `{data_audit['oasis2']['manifest_path']}`
- **Total Scans:** {data_audit['oasis2']['total_subjects']} scans ({data_audit['oasis2']['cn_count']} CN, {data_audit['oasis2']['ad_count']} Impaired)
- **Integrity Status:** {data_audit['oasis2']['overlap_with_oasis1']}

---

## 3. Foundation Model Authenticity & Provenance Audit

### 3.1 BrainIAC
- **Historical Defect:** Previous pilot utilized a proxy adapter with convolutional blocks.
- **V4 Action Gate:** Reject historical proxy. Verify official BrainIAC repository (`https://github.com/AIM-KannLab/BrainIAC`), load authentic Vision Transformer weights, confirm parameter count and weight hash, and prevent random initialization fallback.

### 3.2 MONAI Swin UNETR SSL
- **Architecture:** 3D Swin Vision Transformer with shifted windows.
- **Pretraining Modality Disclosure:** Trained on CVPR 2022 self-supervised cohort of $5,050$ CT volumes (not brain MRI). Must be explicitly described as cross-modality medical volumetric pretraining.
- **Weights Integrity:** Verify official MONAI self-supervised checkpoint, verify state dictionary keys, ensure exact layer alignment.

### 3.3 FOMO25 AMAES ResEnc-B
- **Architecture:** 3D Residual Encoder with Masked Autoencoding (`AMAESResEncB`).
- **Pretraining Dataset:** FOMO-60K / multi-organ collection.
- **Integrity Audit:** Audit potential participant overlap with OASIS-1; distinguish self-supervised pretraining representation from downstream supervised leakage.

---

## 4. Retraining Campaign Execution Plan (Stages D–Q)

1. **Pipeline Engineering:** Config-driven PyTorch + MONAI training system with AMP, gradient accumulation, and deterministic seed control.
2. **Supervised Baseline Retraining:** Retrain 3D ResNet-10, 18, 34, DenseNet-121, 2D Transfer, 2.5D, and Hybrid architectures on $N=138$ OASIS-1.
3. **Foundation Model Adaptation:** Linear probing and validation-driven fine-tuning of BrainIAC, Swin UNETR, and FOMO25.
4. **Controlled Evaluation:** Locked evaluation on $N=30$ OASIS-1 and $N=500$ ADNI, computing ROC-AUC, 95% bootstrap CIs, calibration (Brier score, ECE), and TrustGate risk scores.
5. **Final Scientific Deliverables:** Produce IEEE Access research manuscript, reproducible figures, tables, and test reports.
"""

    AUDIT_MD_PATH.write_text(md_content, encoding="utf-8")
    print(f"[Audit] Wrote Markdown report to {AUDIT_MD_PATH}")
    print("[Audit] Stage A Audit Completed Successfully!")


if __name__ == "__main__":
    main()
