"""
Phase 4: Classical Radiomics & Machine Learning Baselines for Volumetric Brain MRI.

Extracts validated volumetric, intensity histogram, and spatial texture features
from preprocessed 3D MRI volumes. Fits classical models strictly on the training set (N=138),
standardizes with training set statistics, and evaluates on validation (N=30), locked test (N=30),
and external ADNI (N=500) under the locked evaluation protocol.
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import sobel, laplace

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_classification_metrics

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
ADNI_MANIFEST_CSV = r"D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv"
ADNI_TENSORS_DIR = r"D:\brain_mri_data\ADNI\processed\tensors"

RUNS_DIR = r"D:\brain_mri_data\benchmark_runs"
CHECKPOINTS_DIR = os.path.join(RUNS_DIR, "checkpoints")
TABLES_DIR = os.path.join(RUNS_DIR, "tables")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def extract_features_from_volume(arr_3d):
    """
    Extracts 24 reproducible volumetric, intensity, and texture features
    from a single 3D MRI volume (128, 128, 128).
    """
    # Ensure 3D shape
    if arr_3d.ndim == 4:
        arr_3d = arr_3d[0]

    # Non-zero brain tissue mask
    brain_mask = np.abs(arr_3d) > 1e-4
    tissue_voxels = arr_3d[brain_mask]

    total_voxels = arr_3d.size # 128^3 = 2,097,152
    brain_voxels = len(tissue_voxels)

    if brain_voxels < 100:
        return np.zeros(24, dtype=np.float32)

    # 1. Volumetric morphology proxies
    brain_volume_fraction = brain_voxels / total_voxels
    
    # Low-intensity CSF / ventricular proxy (voxels below 20th percentile of tissue)
    p20 = np.percentile(tissue_voxels, 20)
    p80 = np.percentile(tissue_voxels, 80)
    csf_proxy_fraction = (tissue_voxels < p20).sum() / brain_voxels
    wm_proxy_fraction = (tissue_voxels > p80).sum() / brain_voxels
    csf_to_wm_ratio = (csf_proxy_fraction + 1e-6) / (wm_proxy_fraction + 1e-6)

    # 2. First-order intensity histogram statistics
    mean_val = float(np.mean(tissue_voxels))
    std_val = float(np.std(tissue_voxels))
    var_val = float(np.var(tissue_voxels))
    skew_val = float(stats.skew(tissue_voxels))
    kurt_val = float(stats.kurtosis(tissue_voxels))

    p05 = float(np.percentile(tissue_voxels, 5))
    p10 = float(np.percentile(tissue_voxels, 10))
    p25 = float(np.percentile(tissue_voxels, 25))
    p50 = float(np.percentile(tissue_voxels, 50))
    p75 = float(np.percentile(tissue_voxels, 75))
    p90 = float(np.percentile(tissue_voxels, 90))
    p95 = float(np.percentile(tissue_voxels, 95))
    iqr_val = p75 - p25

    energy_val = float(np.sum(tissue_voxels ** 2) / brain_voxels)
    rms_val = float(np.sqrt(energy_val))

    # 3. Spatial texture & structural gradient proxies
    # Sample central sub-volume (64x64x64) for compute efficiency
    center_subvol = arr_3d[32:96, 32:96, 32:96]
    grad_x = sobel(center_subvol, axis=0)
    grad_y = sobel(center_subvol, axis=1)
    grad_z = sobel(center_subvol, axis=2)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    mean_gradient = float(np.mean(grad_magnitude))
    std_gradient = float(np.std(grad_magnitude))

    lap_subvol = laplace(center_subvol)
    mean_laplacian = float(np.mean(np.abs(lap_subvol)))
    var_laplacian = float(np.var(lap_subvol))

    feature_vec = np.array([
        brain_volume_fraction,
        csf_proxy_fraction,
        wm_proxy_fraction,
        csf_to_wm_ratio,
        mean_val,
        std_val,
        var_val,
        skew_val,
        kurt_val,
        p05,
        p10,
        p25,
        p50,
        p75,
        p90,
        p95,
        iqr_val,
        energy_val,
        rms_val,
        mean_gradient,
        std_gradient,
        mean_laplacian,
        var_laplacian,
        brain_voxels / 1e6
    ], dtype=np.float32)

    return feature_vec


def extract_features_for_cohort(df, tensors_dir, is_adni=False):
    """Extracts features for all subjects in a cohort dataframe."""
    feature_matrix = []
    labels = []
    subject_ids = []

    print(f"Extracting features for {len(df)} subjects (is_adni={is_adni})...")
    for idx, row in df.iterrows():
        if is_adni:
            sid = str(row["subject_id"]).strip()
            iid = str(row["image_data_id"]).strip()
            npy_path = os.path.join(tensors_dir, f"{sid}_{iid}.npy")
            sub_key = f"{sid}_{iid}"
            label = 1.0 if row["group"] == "AD" else 0.0
        else:
            sid = str(row["session_id"]).strip()
            npy_path = os.path.join(tensors_dir, f"{sid}.npy")
            sub_key = sid
            if "binary_label" in row:
                label = float(row["binary_label"])
            else:
                label = 1.0 if row["cdr"] > 0.0 else 0.0

        arr = np.load(npy_path)
        feats = extract_features_from_volume(arr)
        feature_matrix.append(feats)
        labels.append(label)
        subject_ids.append(sub_key)

    X = np.vstack(feature_matrix)
    y = np.array(labels, dtype=np.float32)
    return X, y, subject_ids


def run_phase4():
    print("=" * 70)
    print("Phase 4: Classical Radiomics Machine Learning Baselines")
    print("=" * 70)

    # 1. Load splits and manifests
    oasis_df = pd.read_csv(OASIS_SPLITS_CSV)
    train_df = oasis_df[oasis_df["split"] == "train"]
    val_df = oasis_df[oasis_df["split"] == "val"]
    test_df = oasis_df[oasis_df["split"] == "test"]

    adni_df = pd.read_csv(ADNI_MANIFEST_CSV)
    adni_pass = adni_df[adni_df["status"] == "PASS"].reset_index(drop=True)

    # 2. Extract features
    cache_file = os.path.join(TABLES_DIR, "extracted_radiomics_features.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached radiomics features from {cache_file}...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        X_train, y_train, ids_train = data["train"]
        X_val, y_val, ids_val = data["val"]
        X_test, y_test, ids_test = data["test"]
        X_adni, y_adni, ids_adni = data["adni"]
    else:
        X_train, y_train, ids_train = extract_features_for_cohort(train_df, OASIS_TENSORS_DIR)
        X_val, y_val, ids_val = extract_features_for_cohort(val_df, OASIS_TENSORS_DIR)
        X_test, y_test, ids_test = extract_features_for_cohort(test_df, OASIS_TENSORS_DIR)
        X_adni, y_adni, ids_adni = extract_features_for_cohort(adni_pass, ADNI_TENSORS_DIR, is_adni=True)
        
        with open(cache_file, "wb") as f:
            pickle.dump({
                "train": (X_train, y_train, ids_train),
                "val": (X_val, y_val, ids_val),
                "test": (X_test, y_test, ids_test),
                "adni": (X_adni, y_adni, ids_adni)
            }, f)
        print(f"Cached radiomics features to {cache_file}")

    print(f"\nFeature matrix shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}, ADNI={X_adni.shape}")

    # 3. Fit StandardScaler STRICTLY on the training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_adni_scaled = scaler.transform(X_adni)

    # 4. Train and evaluate classical candidate models on Validation set
    candidate_models = {
        "Logistic_Regression_L2": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Logistic_Regression_L1": LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=2000, random_state=42),
        "Linear_SVM": CalibratedClassifierCV(SVC(kernel="linear", C=0.1, random_state=42), cv=3),
        "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)
    }

    val_results = {}
    best_model_name = None
    best_val_auc = -1.0
    best_model_obj = None

    print("\n--- Classical Models Evaluation on Validation Set (N=30) ---")
    for name, clf in candidate_models.items():
        clf.fit(X_train_scaled, y_train)
        probs_val = clf.predict_proba(X_val_scaled)[:, 1]
        m_val = compute_classification_metrics(y_val, probs_val, threshold=0.5)
        val_results[name] = m_val
        print(f"  {name:25s} | Val AUC: {m_val['roc_auc']:.4f} | BalAcc: {m_val['balanced_accuracy']:.4f} | Brier: {m_val['brier_score']:.4f}")

        if m_val["roc_auc"] > best_val_auc:
            best_val_auc = m_val["roc_auc"]
            best_model_name = name
            best_model_obj = clf

    print(f"\n--> Selected Winning Classical Model: {best_model_name} (Val AUC: {best_val_auc:.4f})")

    # Save the selected model
    classical_ckpt = os.path.join(CHECKPOINTS_DIR, "classical_radiomics_best.pkl")
    with open(classical_ckpt, "wb") as f:
        pickle.dump({"model": best_model_obj, "scaler": scaler, "name": best_model_name}, f)
    print(f"Saved winning classical model checkpoint to {classical_ckpt}")

    summary = {
        "candidate_validation_results": val_results,
        "selected_model": best_model_name,
        "best_val_auc": best_val_auc,
        "checkpoint_path": classical_ckpt
    }
    out_path = os.path.join(TABLES_DIR, "phase4_classical_radiomics_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_phase4()
