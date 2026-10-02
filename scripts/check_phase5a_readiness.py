"""
Phase 5A Real Data Readiness & Environment Audit Script.
Checks dataset paths, GPU availability, disk space, checkpoints, and task feasibility.
"""

import os
import sys
import shutil
import json
import torch
import psutil

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_parser import load_config
from preprocessing.adni_parser import ADNIMetadataParser
from preprocessing.oasis_parser import OASIS3MetadataParser


def audit_phase5a():
    audit_results = {}

    # 1. Environment & Hardware Audit
    cuda_avail = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution Only)"
    gpu_vram = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0

    ram = psutil.virtual_memory()
    total_ram_gb = round(ram.total / (1024**3), 2)
    avail_ram_gb = round(ram.available / (1024**3), 2)

    # Disk Space
    d_usage = shutil.disk_usage("D:\\")
    d_free_gb = round(d_usage.free / (1024**3), 2)
    d_total_gb = round(d_usage.total / (1024**3), 2)

    c_usage = shutil.disk_usage("C:\\")
    c_free_gb = round(c_usage.free / (1024**3), 2)
    c_total_gb = round(c_usage.total / (1024**3), 2)

    audit_results["hardware"] = {
        "cuda_available": cuda_avail,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(gpu_vram, 2),
        "total_host_ram_gb": total_ram_gb,
        "available_host_ram_gb": avail_ram_gb,
        "drive_d_free_gb": d_free_gb,
        "drive_d_total_gb": d_total_gb,
        "drive_c_free_gb": c_free_gb,
        "drive_c_total_gb": c_total_gb,
    }

    # 2. Configured Dataset Paths
    config = load_config("configs")
    adni_cfg = config.get("datasets", {}).get("adni", {})
    oasis_cfg = config.get("datasets", {}).get("oasis3", {})

    adni_data_dir = os.environ.get("ADNI_DATA_DIR", adni_cfg.get("data_dir", "data/ADNI/scans"))
    adni_meta_path = os.environ.get("ADNI_METADATA_PATH", adni_cfg.get("metadata_path", "data/ADNI/metadata/ADNIMERGE.csv"))

    oasis_data_dir = os.environ.get("OASIS3_DATA_DIR", oasis_cfg.get("data_dir", "data/OASIS3/scans"))
    oasis_meta_path = os.environ.get("OASIS3_METADATA_PATH", oasis_cfg.get("metadata_path", "data/OASIS3/metadata/oasis3_clinical.csv"))

    # Check ADNI
    adni_data_exists = os.path.exists(adni_data_dir) and len(os.listdir(adni_data_dir)) > 0 if os.path.exists(adni_data_dir) else False
    adni_meta_exists = os.path.exists(adni_meta_path)

    adni_counts = {"scans": 0, "subjects": 0, "status": "NOT_MOUNTED"}
    if adni_meta_exists and adni_data_exists:
        try:
            parser = ADNIMetadataParser(metadata_path=adni_meta_path, data_dir=adni_data_dir)
            df = parser.parse()
            adni_counts = {
                "scans": len(df),
                "subjects": df["subject_id"].nunique(),
                "status": "MOUNTED_AND_VERIFIED",
            }
        except Exception as e:
            adni_counts = {"error": str(e), "status": "PARSING_ERROR"}

    audit_results["adni"] = {
        "data_dir": adni_data_dir,
        "metadata_path": adni_meta_path,
        "data_dir_exists": adni_data_exists,
        "metadata_path_exists": adni_meta_exists,
        "audit": adni_counts,
    }

    # Check OASIS-3
    oasis_data_exists = os.path.exists(oasis_data_dir) and len(os.listdir(oasis_data_dir)) > 0 if os.path.exists(oasis_data_dir) else False
    oasis_meta_exists = os.path.exists(oasis_meta_path)

    oasis_counts = {"scans": 0, "subjects": 0, "status": "NOT_MOUNTED"}
    if oasis_meta_exists and oasis_data_exists:
        try:
            parser = OASIS3MetadataParser(metadata_path=oasis_meta_path, data_dir=oasis_data_dir)
            df = parser.parse()
            oasis_counts = {
                "scans": len(df),
                "subjects": df["subject_id"].nunique(),
                "status": "MOUNTED_AND_VERIFIED",
            }
        except Exception as e:
            oasis_counts = {"error": str(e), "status": "PARSING_ERROR"}

    audit_results["oasis3"] = {
        "data_dir": oasis_data_dir,
        "metadata_path": oasis_meta_path,
        "data_dir_exists": oasis_data_exists,
        "metadata_path_exists": oasis_meta_exists,
        "audit": oasis_counts,
    }

    # 3. Foundation Model Checkpoints Audit
    checkpoints = {
        "BrainIAC": config.get("models", {}).get("foundation_brainiac", {}).get("weights_path", "checkpoints/brainiac_vit_b16.pt"),
        "FOMO25_Winner": config.get("models", {}).get("foundation_fomo25", {}).get("weights_path", "checkpoints/fomo25_ssl_encoder.pt"),
        "MONAI_SwinUNETR_SSL": config.get("models", {}).get("foundation_swin_unetr", {}).get("weights_path", "checkpoints/monai_swin_unetr_ssl.pt"),
    }

    chk_audit = {}
    for name, path in checkpoints.items():
        exists = os.path.exists(path)
        size_mb = round(os.path.getsize(path) / (1024**2), 2) if exists else 0.0
        chk_audit[name] = {
            "configured_path": path,
            "exists_locally": exists,
            "file_size_mb": size_mb,
            "status": "READY_LOCALLY" if exists else "NOT_FOUND_LOCALLY"
        }

    audit_results["foundation_checkpoints"] = chk_audit

    # 4. Task Feasibility Summary
    tasks = config.get("tasks", {})
    audit_results["task_definitions"] = {
        "ad_vs_cn": {
            "type": tasks.get("ad_vs_cn", {}).get("type"),
            "target": tasks.get("ad_vs_cn", {}).get("target_column"),
            "classes": tasks.get("ad_vs_cn", {}).get("classes"),
        },
        "brain_age": {
            "type": tasks.get("brain_age", {}).get("type"),
            "target": tasks.get("brain_age", {}).get("target_column"),
        },
        "mci_vs_cn": {
            "type": tasks.get("mci_vs_cn", {}).get("type"),
            "target": tasks.get("mci_vs_cn", {}).get("target_column"),
            "classes": tasks.get("mci_vs_cn", {}).get("classes"),
        }
    }

    print(json.dumps(audit_results, indent=2))
    with open("results/phase5a_readiness_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    return audit_results


if __name__ == "__main__":
    audit_phase5a()
