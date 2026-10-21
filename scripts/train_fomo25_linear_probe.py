"""
Phase B & C: FOMO25 AMAES ResEnc-B Linear Probe Benchmark.

Extracts frozen 320-dimensional features from OASIS-1 (198 subjects) and ADNI (500 subjects),
trains linear probes strictly on OASIS train (N=138),
selects hyperparameter C strictly on OASIS val (N=30),
and evaluates once on OASIS locked test (N=30) and ADNI external (N=500).
Computes 1,000-iteration bootstrap 95% confidence intervals.
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.foundation_fomo25.amaes_adapter import AMAESResEncBAdapter
from utils.logging_utils import get_logger

logger = get_logger("FOMO25LinearProbe")

OASIS_SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
OASIS_TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
ADNI_MANIFEST_CSV = r"D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv"
ADNI_TENSORS_DIR = r"D:\brain_mri_data\ADNI\processed\tensors"
OUTPUT_DIR = r"D:\ml project\results\fomo_linear_probe"
RUNS_TABLES_DIR = r"D:\brain_mri_data\benchmark_runs\tables"

RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_bootstrap_auc_ci(y_true, y_prob, n_bootstraps=1000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        return 0.5, 0.0, 1.0

    sorted_scores = np.sort(bootstrapped_scores)
    median = float(np.median(sorted_scores))
    ci_lower = float(np.percentile(sorted_scores, 2.5))
    ci_upper = float(np.percentile(sorted_scores, 97.5))
    return median, ci_lower, ci_upper


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def extract_features_for_manifest(df, tensors_dir, adapter, device, id_col, label_col):
    adapter.eval()
    features = []
    manifest_rows = []

    for idx, row in df.iterrows():
        subj_id = row[id_col]
        label = float(row[label_col])
        if "pt_path" in row and os.path.exists(str(row["pt_path"])):
            pt_path = str(row["pt_path"])
        else:
            pt_path = os.path.join(tensors_dir, f"{subj_id}.pt")
            
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Missing tensor for {subj_id} at {pt_path}")

        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        tensor = payload["tensor"] if isinstance(payload, dict) and "tensor" in payload else payload
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.float().to(device)
        with torch.no_grad():
            feat = adapter.extract_features(tensor)
            feat_vec = feat.cpu().squeeze(0).numpy()

        features.append(feat_vec)
        manifest_rows.append({
            "participant_id": subj_id,
            "label": label,
            "feature_dim": len(feat_vec),
            "feature_norm": float(np.linalg.norm(feat_vec)),
            "tensor_path": pt_path
        })

    features_matrix = np.array(features, dtype=np.float32)
    manifest_df = pd.DataFrame(manifest_rows)
    return features_matrix, manifest_df


def run_fomo_experiment():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RUNS_TABLES_DIR, exist_ok=True)

    logger.info("=================================================================")
    logger.info("Starting FOMO25 AMAES ResEnc-B Linear Probe Benchmark")
    logger.info("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # 1. Load adapter
    logger.info("Initializing AMAESResEncBAdapter...")
    adapter = AMAESResEncBAdapter(device=device)
    adapter.eval()
    logger.info(f"Adapter parameters: {adapter.get_parameter_count()}")

    # 2. Extract OASIS Features
    oasis_df = pd.read_csv(OASIS_SPLITS_CSV)
    oasis_features_pt = os.path.join(OUTPUT_DIR, "oasis_fomo_features.pt")
    oasis_manifest_csv = os.path.join(OUTPUT_DIR, "oasis_fomo_features_manifest.csv")

    if os.path.exists(oasis_features_pt) and os.path.exists(oasis_manifest_csv):
        logger.info(f"Loading cached OASIS FOMO features from {oasis_features_pt}")
        oasis_saved = torch.load(oasis_features_pt, map_location="cpu", weights_only=False)
        oasis_feats = oasis_saved["features"]
        oasis_manifest = pd.read_csv(oasis_manifest_csv)
    else:
        logger.info(f"Extracting FOMO features for {len(oasis_df)} OASIS participants...")
        t0 = time.time()
        oasis_feats, oasis_manifest = extract_features_for_manifest(
            oasis_df, OASIS_TENSORS_DIR, adapter, device, id_col="session_id", label_col="binary_label"
        )
        oasis_manifest["participant_id"] = oasis_df["base_subject_id"].values
        oasis_manifest["session_id"] = oasis_df["session_id"].values
        oasis_manifest["split"] = oasis_df["split"].values
        torch.save({
            "features": oasis_feats,
            "participant_ids": oasis_manifest["participant_id"].values,
            "labels": oasis_manifest["label"].values,
            "splits": oasis_manifest["split"].values
        }, oasis_features_pt)
        oasis_manifest.to_csv(oasis_manifest_csv, index=False)
        logger.info(f"OASIS feature extraction finished in {time.time()-t0:.2f}s. Saved to {oasis_features_pt}")

    # 3. Extract ADNI Features
    adni_df = pd.read_csv(ADNI_MANIFEST_CSV)
    adni_pass = adni_df[adni_df["status"] == "PASS"].reset_index(drop=True)
    # Ensure binary label (CN=0, AD=1)
    if "binary_label" not in adni_pass.columns:
        adni_pass["binary_label"] = (adni_pass["research_group"] == "AD").astype(float)

    adni_features_pt = os.path.join(OUTPUT_DIR, "adni_fomo_features.pt")
    adni_manifest_csv = os.path.join(OUTPUT_DIR, "adni_fomo_features_manifest.csv")

    if os.path.exists(adni_features_pt) and os.path.exists(adni_manifest_csv):
        logger.info(f"Loading cached ADNI FOMO features from {adni_features_pt}")
        adni_saved = torch.load(adni_features_pt, map_location="cpu", weights_only=False)
        adni_feats = adni_saved["features"]
        adni_manifest = pd.read_csv(adni_manifest_csv)
    else:
        logger.info(f"Extracting FOMO features for {len(adni_pass)} ADNI participants...")
        t0 = time.time()
        adni_feats, adni_manifest = extract_features_for_manifest(
            adni_pass, ADNI_TENSORS_DIR, adapter, device, id_col="subject_id", label_col="binary_label"
        )
        adni_manifest["manufacturer"] = adni_pass["manufacturer"].values
        adni_manifest["field_strength"] = adni_pass["field_strength"].values
        torch.save({
            "features": adni_feats,
            "participant_ids": adni_manifest["participant_id"].values,
            "labels": adni_manifest["label"].values,
            "manufacturers": adni_manifest["manufacturer"].values,
            "field_strengths": adni_manifest["field_strength"].values
        }, adni_features_pt)
        adni_manifest.to_csv(adni_manifest_csv, index=False)
        logger.info(f"ADNI feature extraction finished in {time.time()-t0:.2f}s. Saved to {adni_features_pt}")

    # 4. Split OASIS into Train, Val, Test
    train_mask = (oasis_manifest["split"] == "train").values
    val_mask = (oasis_manifest["split"] == "val").values
    test_mask = (oasis_manifest["split"] == "test").values

    X_train = oasis_feats[train_mask]
    y_train = oasis_manifest["label"].values[train_mask]

    X_val = oasis_feats[val_mask]
    y_val = oasis_manifest["label"].values[val_mask]

    X_test = oasis_feats[test_mask]
    y_test = oasis_manifest["label"].values[test_mask]

    X_adni = adni_feats
    y_adni = adni_manifest["label"].values

    logger.info(f"Dataset Partitions: Train N={len(X_train)} (pos={int(y_train.sum())}), "
                f"Val N={len(X_val)} (pos={int(y_val.sum())}), "
                f"Test N={len(X_test)} (pos={int(y_test.sum())}), "
                f"ADNI N={len(X_adni)} (pos={int(y_adni.sum())})")

    # Fit scaler strictly on Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_adni_scaled = scaler.transform(X_adni)

    # 5. Hyperparameter tuning & model selection strictly on Validation set
    candidate_c_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    tuning_records = []

    best_val_auc = -1.0
    best_c = None
    best_clf = None

    for c in candidate_c_values:
        clf = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED)
        clf.fit(X_train_scaled, y_train)

        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_probs))
        val_brier = float(brier_score_loss(y_val, val_probs))
        val_preds = (val_probs >= 0.5).astype(int)
        val_bal_acc = float(balanced_accuracy_score(y_val, val_preds))

        record = {
            "C": c,
            "val_auc": val_auc,
            "val_brier": val_brier,
            "val_balanced_acc": val_bal_acc
        }
        tuning_records.append(record)
        logger.info(f"Tuning C={c:8.4f} -> Val AUC: {val_auc:.4f}, Val Brier: {val_brier:.4f}, Val BalAcc: {val_bal_acc:.4f}")

        # Strict selection criterion: highest validation AUC (tie-breaker lowest Brier)
        if val_auc > best_val_auc or (val_auc == best_val_auc and val_brier < tuning_records[-1]["val_brier"]):
            best_val_auc = val_auc
            best_c = c
            best_clf = clf

    logger.info(f"Selected Best Hyperparameter strictly via Validation: C={best_c} (Val AUC={best_val_auc:.4f})")

    # Save fitted model
    model_save_path = os.path.join(OUTPUT_DIR, "fomo25_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump({"scaler": scaler, "classifier": best_clf, "best_C": best_c}, f)

    # 6. Evaluate Selected Model on Validation, Locked Test, and ADNI
    def evaluate_split(X_s, y_s, split_name):
        probs = best_clf.predict_proba(X_s)[:, 1]
        preds_05 = (probs >= 0.5).astype(int)

        auc = float(roc_auc_score(y_s, probs))
        auc_median, ci_low, ci_high = compute_bootstrap_auc_ci(y_s, probs, n_bootstraps=1000, rng_seed=RANDOM_SEED)
        acc = float(accuracy_score(y_s, preds_05))
        bal_acc = float(balanced_accuracy_score(y_s, preds_05))
        sens = float(recall_score(y_s, preds_05, zero_division=0))
        spec = float(confusion_matrix(y_s, preds_05, labels=[0, 1])[0, 0] / max(1, (y_s == 0).sum()))
        prec = float(precision_score(y_s, preds_05, zero_division=0))
        f1 = float(f1_score(y_s, preds_05, zero_division=0))
        brier = float(brier_score_loss(y_s, probs))
        ece = compute_ece(y_s, probs)

        metrics = {
            "split": split_name,
            "N": len(y_s),
            "N_pos": int((y_s == 1).sum()),
            "N_neg": int((y_s == 0).sum()),
            "roc_auc": round(auc, 4),
            "roc_auc_ci_lower": round(ci_low, 4),
            "roc_auc_ci_upper": round(ci_high, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "accuracy": round(acc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4)
        }
        return probs, metrics

    val_probs, val_metrics = evaluate_split(X_val_scaled, y_val, "OASIS_Validation")
    test_probs, test_metrics = evaluate_split(X_test_scaled, y_test, "OASIS_Locked_Test")
    adni_probs, adni_metrics = evaluate_split(X_adni_scaled, y_adni, "ADNI_External")

    # ADNI Subgroups
    adni_mfg = adni_manifest["manufacturer"].values
    adni_fs = adni_manifest["field_strength"].values

    siemens_mask = np.array(["SIEMENS" in str(m).upper() for m in adni_mfg])
    philips_mask = np.array(["PHILIPS" in str(m).upper() for m in adni_mfg])
    fs_15t_mask = (adni_fs < 2.0)
    fs_30t_mask = (adni_fs >= 2.0)

    _, adni_siemens_metrics = evaluate_split(X_adni_scaled[siemens_mask], y_adni[siemens_mask], "ADNI_Siemens")
    _, adni_philips_metrics = evaluate_split(X_adni_scaled[philips_mask], y_adni[philips_mask], "ADNI_Philips")
    _, adni_15t_metrics = evaluate_split(X_adni_scaled[fs_15t_mask], y_adni[fs_15t_mask], "ADNI_1.5T")
    _, adni_30t_metrics = evaluate_split(X_adni_scaled[fs_30t_mask], y_adni[fs_30t_mask], "ADNI_3.0T")

    # 7. Save Prediction CSVs
    val_pred_df = pd.DataFrame({
        "participant_id": oasis_manifest["participant_id"].values[val_mask],
        "true_label": y_val,
        "fomo_amaes_prob": val_probs
    })
    val_pred_df.to_csv(os.path.join(OUTPUT_DIR, "oasis_val_predictions.csv"), index=False)

    test_pred_df = pd.DataFrame({
        "participant_id": oasis_manifest["participant_id"].values[test_mask],
        "true_label": y_test,
        "fomo_amaes_prob": test_probs
    })
    test_pred_df.to_csv(os.path.join(OUTPUT_DIR, "oasis_test_predictions.csv"), index=False)

    adni_pred_df = pd.DataFrame({
        "participant_id": adni_manifest["participant_id"].values,
        "true_label": y_adni,
        "manufacturer": adni_mfg,
        "field_strength": adni_fs,
        "fomo_amaes_prob": adni_probs
    })
    adni_pred_df.to_csv(os.path.join(OUTPUT_DIR, "adni_external_predictions.csv"), index=False)

    # 8. Compile Master Summary JSON
    summary_report = {
        "benchmark_experiment": "FOMO25 AMAES ResEnc-B Frozen Linear Probe",
        "date": "2026-10-21",
        "random_seed": RANDOM_SEED,
        "model_spec": {
            "name": "FOMO25 AMAES ResEnc-B",
            "backbone_parameters": 90296576,
            "feature_dimension": 320,
            "trainable_parameters": 321,
            "pretraining": "AMAES on FOMO260K (260,000 multi-sequence 3D MRIs)",
            "license": "CC-BY-NC-SA 4.0",
            "weights_source": "FOMO-MRI/AMAES_resenc_b_fomo260k (resenc_unet_b.ckpt)"
        },
        "training_protocol": {
            "fit_cohort": "OASIS-1 Primary Train (N=138)",
            "selection_cohort": "OASIS-1 Primary Validation (N=30)",
            "selected_C": best_c,
            "hyperparameter_sweep": tuning_records,
            "feature_scaling": "StandardScaler fit strictly on Train"
        },
        "metrics": {
            "validation": val_metrics,
            "locked_test": test_metrics,
            "adni_external": adni_metrics,
            "adni_subgroups": {
                "siemens": adni_siemens_metrics,
                "philips": adni_philips_metrics,
                "field_1_5t": adni_15t_metrics,
                "field_3_0t": adni_30t_metrics
            }
        },
        "scientific_integrity_attestation": {
            "test_set_used_for_tuning": False,
            "adni_used_for_tuning": False,
            "selection_criterion": "Validation ROC-AUC"
        }
    }

    report_json_path = os.path.join(OUTPUT_DIR, "fomo25_linear_probe_results.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info("=================================================================")
    logger.info(f"OASIS Validation Metrics : AUC={val_metrics['roc_auc']:.4f} [{val_metrics['roc_auc_ci_lower']:.4f}, {val_metrics['roc_auc_ci_upper']:.4f}], BalAcc={val_metrics['balanced_accuracy']:.4f}, Sens={val_metrics['sensitivity']:.4f}, Spec={val_metrics['specificity']:.4f}, Brier={val_metrics['brier_score']:.4f}")
    logger.info(f"OASIS Locked Test Metrics: AUC={test_metrics['roc_auc']:.4f} [{test_metrics['roc_auc_ci_lower']:.4f}, {test_metrics['roc_auc_ci_upper']:.4f}], BalAcc={test_metrics['balanced_accuracy']:.4f}, Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f}, Brier={test_metrics['brier_score']:.4f}")
    logger.info(f"ADNI External Metrics    : AUC={adni_metrics['roc_auc']:.4f} [{adni_metrics['roc_auc_ci_lower']:.4f}, {adni_metrics['roc_auc_ci_upper']:.4f}], BalAcc={adni_metrics['balanced_accuracy']:.4f}, Sens={adni_metrics['sensitivity']:.4f}, Spec={adni_metrics['specificity']:.4f}, Brier={adni_metrics['brier_score']:.4f}")
    logger.info("=================================================================")
    logger.info(f"Summary JSON saved to {report_json_path}")
    return summary_report


if __name__ == "__main__":
    run_fomo_experiment()
