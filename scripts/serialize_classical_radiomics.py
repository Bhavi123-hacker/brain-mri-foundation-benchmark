import os
import sys
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import stats

DATA_ROOT = Path(r"D:\brain_mri_data")
RESULTS_DIR = Path(r"D:\ml project\results\v4")
CKPTS_DIR = RESULTS_DIR / "checkpoints"

def extract_classical_features_from_volume(arr_3d: np.ndarray) -> np.ndarray:
    if arr_3d.ndim == 4:
        arr_3d = arr_3d[0]
    brain_mask = np.abs(arr_3d) > 1e-4
    tissue_voxels = arr_3d[brain_mask]
    total_voxels = arr_3d.size
    brain_voxels = len(tissue_voxels)
    if brain_voxels < 100:
        return np.zeros(24, dtype=np.float32)

    brain_volume_fraction = brain_voxels / total_voxels
    p20 = np.percentile(tissue_voxels, 20)
    p80 = np.percentile(tissue_voxels, 80)
    csf_proxy_fraction = (tissue_voxels < p20).sum() / brain_voxels
    wm_proxy_fraction = (tissue_voxels > p80).sum() / brain_voxels
    csf_to_wm_ratio = (csf_proxy_fraction + 1e-6) / (wm_proxy_fraction + 1e-6)

    mean_val = float(np.mean(tissue_voxels))
    std_val = float(np.std(tissue_voxels))
    var_val = float(np.var(tissue_voxels))
    skew_val = float(stats.skew(tissue_voxels))
    kurt_val = float(stats.kurtosis(tissue_voxels))
    p10 = float(np.percentile(tissue_voxels, 10))
    p25 = float(np.percentile(tissue_voxels, 25))
    p50 = float(np.percentile(tissue_voxels, 50))
    p75 = float(np.percentile(tissue_voxels, 75))
    p90 = float(np.percentile(tissue_voxels, 90))
    iqr_val = p75 - p25
    entropy_val = float(stats.entropy(np.histogram(tissue_voxels, bins=32)[0] + 1e-5))

    mid_z = arr_3d.shape[0] // 2
    left_hem = arr_3d[:mid_z]
    right_hem = arr_3d[mid_z:]
    left_mean = float(np.mean(left_hem[np.abs(left_hem) > 1e-4])) if (np.abs(left_hem) > 1e-4).sum() > 0 else 0.0
    right_mean = float(np.mean(right_hem[np.abs(right_hem) > 1e-4])) if (np.abs(right_hem) > 1e-4).sum() > 0 else 0.0
    asymmetry_index = abs(left_mean - right_mean) / (abs(left_mean) + abs(right_mean) + 1e-6)

    grad_z, grad_y, grad_x = np.gradient(arr_3d)
    grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
    mean_grad = float(np.mean(grad_mag[brain_mask])) if brain_voxels > 0 else 0.0
    std_grad = float(np.std(grad_mag[brain_mask])) if brain_voxels > 0 else 0.0

    z_prof = np.mean(arr_3d, axis=(1, 2))
    y_prof = np.mean(arr_3d, axis=(0, 2))
    x_prof = np.mean(arr_3d, axis=(0, 1))
    z_peak_ratio = float(np.max(z_prof) / (np.mean(z_prof) + 1e-6))
    y_peak_ratio = float(np.max(y_prof) / (np.mean(y_prof) + 1e-6))
    x_peak_ratio = float(np.max(x_prof) / (np.mean(x_prof) + 1e-6))
    edge_contrast = float(p90 - p10)

    return np.array([
        brain_volume_fraction, csf_proxy_fraction, wm_proxy_fraction, csf_to_wm_ratio,
        mean_val, std_val, var_val, skew_val, kurt_val,
        p10, p25, p50, p75, p90, iqr_val, entropy_val,
        asymmetry_index, mean_grad, std_grad,
        z_peak_ratio, y_peak_ratio, x_peak_ratio, edge_contrast,
        float(brain_voxels)
    ], dtype=np.float32)

print("Loading dataset splits...")
oasis1_splits = pd.read_csv(DATA_ROOT / "OASIS1" / "manifests" / "oasis1_primary_cohort_splits.csv")
train_df = oasis1_splits[oasis1_splits["split"] == "train"].reset_index(drop=True)
val_df = oasis1_splits[oasis1_splits["split"] == "val"].reset_index(drop=True)
test_df = oasis1_splits[oasis1_splits["split"] == "test"].reset_index(drop=True)

train_tensors = [np.load(DATA_ROOT / "OASIS1" / "processed" / "tensors" / f"{sid}.npy") for sid in train_df["session_id"]]
train_y = train_df["binary_label"].values

print("Extracting training radiomics...")
train_rads = np.stack([extract_classical_features_from_volume(x) for x in train_tensors])
scaler = StandardScaler()
train_rads_s = scaler.fit_transform(train_rads)

print("Fitting Linear SVM...")
svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm.fit(train_rads_s, train_y)

artifact = {
    "model": svm,
    "scaler": scaler,
    "feature_names": [
        "brain_volume_fraction", "csf_proxy_fraction", "wm_proxy_fraction", "csf_to_wm_ratio",
        "mean_val", "std_val", "var_val", "skew_val", "kurt_val",
        "p10", "p25", "p50", "p75", "p90", "iqr_val", "entropy_val",
        "asymmetry_index", "mean_grad", "std_grad",
        "z_peak_ratio", "y_peak_ratio", "x_peak_ratio", "edge_contrast",
        "brain_voxels"
    ],
    "input_dim": 24,
    "model_family": "classical_radiomics",
    "kernel": "linear",
    "C": 1.0,
    "random_state": 42,
    "training_samples": len(train_df),
}

joblib_path = CKPTS_DIR / "classical_radiomics_model.joblib"
joblib.dump(artifact, joblib_path)
print(f"Saved classical radiomics model to: {joblib_path} ({joblib_path.stat().st_size} bytes)")

# Compute sha256
with open(joblib_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"Classical Radiomics SHA256: {sha}")
