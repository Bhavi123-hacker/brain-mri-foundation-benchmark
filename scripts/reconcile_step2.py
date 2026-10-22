import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score
from pathlib import Path

PROJECT_ROOT = Path("D:/ml project")

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.sort(bootstrapped_scores)
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return round(float(ci_lower), 4), round(float(ci_upper), 4)

def reconcile_full():
    print("=== RECONCILING ALL 10 BENCHMARK MODELS ===")
    
    # -------------------------------------------------------------
    # 1. OASIS-1 VALIDATION (Locked Threshold Derivation)
    # -------------------------------------------------------------
    df_val = pd.read_csv(PROJECT_ROOT / "results/v3/v3_oasis_validation_predictions.csv")
    y_val = df_val["ground_truth"].values
    
    # Thresholds using Youden's J on validation
    val_model_cols = {
        "MONAI Swin UNETR SSL (Linear Probe)": "swin_unetr_ssl_prob",
        "2D Transfer ResNet-18 (Slice-Aggregated)": "resnet2d_transfer_prob",
        "Dual-Branch Hybrid (ResNet-10 + Radiomics)": "dual_branch_hybrid_prob",
        "FOMO25 AMAES ResEnc-B (Linear Probe)": "fomo_amaes_prob",
        "3D ResNet-10 (Pipeline B)": "3d_resnet_10_prob",
        "Classical Radiomics (Linear SVM)": "classical_radiomics_prob",
        "3D ResNet-34 (Scratch Baseline)": "resnet34_prob",
        "3D ResNet-18 (Scratch Baseline)": "resnet18_prob",
        "3D DenseNet-121 (Pipeline B)": "3d_densenet_121_prob",
        "2.5D Multi-View ResNet-18 (Pipeline B)": "2.5d_multi_view_resnet_18_prob",
    }
    
    thresholds = {}
    from sklearn.metrics import roc_curve
    for mname, col in val_model_cols.items():
        fpr, tpr, ths = roc_curve(y_val, df_val[col].values)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_th = ths[best_idx]
        # In case threshold > 1.0 (scikit-learn convention for ths[0])
        if best_th > 1.0:
            best_th = ths[1] if len(ths) > 1 else 0.5
        thresholds[mname] = round(float(best_th), 4)
        print(f"Validation threshold for {mname}: {thresholds[mname]}")

    # -------------------------------------------------------------
    # 2. OASIS-1 LOCKED TEST (N=30)
    # -------------------------------------------------------------
    df_test_v2 = pd.read_csv(PROJECT_ROOT / "results/v2/tables/v2_oasis_locked_test_predictions.csv")
    df_test_v1 = pd.read_csv(PROJECT_ROOT / "results/report/oasis_locked_test_predictions.csv")
    df_test_fomo = pd.read_csv(PROJECT_ROOT / "results/fomo_linear_probe/oasis_test_predictions.csv")

    # Join on session_id
    # Check session_id in each
    print("\nOASIS-1 Test session_id matching:")
    print("V2:", len(df_test_v2), "V1:", len(df_test_v1), "FOMO:", len(df_test_fomo))
    
    # Merge
    merged_test = df_test_v2[['session_id', 'ground_truth', '3d_resnet_10_prob', '3d_densenet_121_prob', '2.5d_multi_view_resnet_18_prob', 'dual_branch_hybrid_prob']].copy()
    
    # Check V1 columns
    if 'session_id' in df_test_v1.columns:
        v1_merge = df_test_v1[['session_id', 'resnet18_prob', 'resnet34_prob', 'resnet2d_transfer_prob', 'classical_radiomics_prob', 'swin_unetr_ssl_prob']]
        merged_test = pd.merge(merged_test, v1_merge, on='session_id', how='inner')
    else:
        # maybe participant_id or index?
        print("V1 columns:", df_test_v1.columns)
        merged_test = pd.concat([merged_test, df_test_v1[['resnet18_prob', 'resnet34_prob', 'resnet2d_transfer_prob', 'classical_radiomics_prob', 'swin_unetr_ssl_prob']]], axis=1)

    if 'session_id' in df_test_fomo.columns:
        fomo_merge = df_test_fomo[['session_id', 'fomo_amaes_prob']]
        merged_test = pd.merge(merged_test, fomo_merge, on='session_id', how='inner')
    else:
        print("FOMO columns:", df_test_fomo.columns)
        merged_test = pd.concat([merged_test, df_test_fomo[['fomo_amaes_prob']]], axis=1)

    print("Merged OASIS-1 Test shape:", merged_test.shape)

    # -------------------------------------------------------------
    # 3. ADNI ZERO-SHOT EXTERNAL (N=500)
    # -------------------------------------------------------------
    df_adni_v2 = pd.read_csv(PROJECT_ROOT / "results/v2/tables/v2_adni_external_predictions.csv")
    df_adni_v1 = pd.read_csv(PROJECT_ROOT / "results/report/adni_external_predictions.csv")
    df_adni_fomo = pd.read_csv(PROJECT_ROOT / "results/fomo_linear_probe/adni_external_predictions.csv")

    # Check join keys
    # V2 has subject_id, image_data_id, ground_truth
    # V1 has subject_id, image_data_id, binary_label
    # FOMO has participant_id (which is subject_id)
    # Let's inspect matching
    merged_adni = df_adni_v2[['subject_id', 'image_data_id', 'ground_truth', '3d_resnet_10_prob', '3d_densenet_121_prob', '2.5d_multi_view_resnet_18_prob', 'dual_branch_hybrid_prob']].copy()
    
    # Merge V1
    v1_sub = df_adni_v1[['subject_id', 'image_data_id', 'resnet18_prob', 'resnet34_prob', 'resnet2d_transfer_prob', 'classical_radiomics_prob', 'swin_unetr_ssl_prob']]
    merged_adni = pd.merge(merged_adni, v1_sub, on=['subject_id', 'image_data_id'], how='inner')
    print("Merged ADNI with V1 shape:", merged_adni.shape)

    # FOMO has participant_id. Let's see if participant_id matches subject_id
    df_adni_fomo_renamed = df_adni_fomo.rename(columns={'participant_id': 'subject_id'})
    # Is subject_id 1-to-1 in FOMO and V2?
    print("Unique subjects in V2:", merged_adni['subject_id'].nunique(), "FOMO:", df_adni_fomo_renamed['subject_id'].nunique())
    # Merge on subject_id
    merged_adni = pd.merge(merged_adni, df_adni_fomo_renamed[['subject_id', 'fomo_amaes_prob']], on='subject_id', how='inner')
    print("Merged ADNI with FOMO shape:", merged_adni.shape)

    # Compute metrics for all 10 models on OASIS-1 Test and ADNI
    results_list = []
    for mname, col in val_model_cols.items():
        th = thresholds[mname]
        
        # OASIS-1 Test
        y_test = merged_test['ground_truth'].values
        p_test = merged_test[col].values
        auc_test = round(roc_auc_score(y_test, p_test), 4)
        ci_test = bootstrap_auc_ci(y_test, p_test)
        brier_test = round(brier_score_loss(y_test, p_test), 4)
        bacc_test = round(balanced_accuracy_score(y_test, (p_test >= th).astype(int)), 4)

        # ADNI
        y_adni = merged_adni['ground_truth'].values
        p_adni = merged_adni[col].values
        auc_adni = round(roc_auc_score(y_adni, p_adni), 4)
        ci_adni = bootstrap_auc_ci(y_adni, p_adni)
        brier_adni = round(brier_score_loss(y_adni, p_adni), 4)
        bacc_adni = round(balanced_accuracy_score(y_adni, (p_adni >= th).astype(int)), 4)

        delta_auc = round(auc_adni - auc_test, 4)

        results_list.append({
            "model_name": mname,
            "col": col,
            "threshold_val": th,
            "oasis_test_auc": auc_test,
            "oasis_test_ci": ci_test,
            "oasis_test_brier": brier_test,
            "oasis_test_bacc": bacc_test,
            "adni_auc": auc_adni,
            "adni_ci": ci_adni,
            "adni_brier": brier_adni,
            "adni_bacc": bacc_adni,
            "delta_auc": delta_auc
        })

    df_res = pd.DataFrame(results_list)
    df_res = df_res.sort_values(by="adni_auc", ascending=False).reset_index(drop=True)
    df_res["rank_adni"] = range(1, len(df_res) + 1)
    
    print("\n=== RECONCILED MODEL LEADERBOARD (SORTED BY ADNI ZERO-SHOT AUC) ===")
    for idx, r in df_res.iterrows():
        print(f"Rank {r['rank_adni']}: {r['model_name']}")
        print(f"   OASIS-1 Test AUC: {r['oasis_test_auc']} {r['oasis_test_ci']} | Brier: {r['oasis_test_brier']}")
        print(f"   ADNI AUC:         {r['adni_auc']} {r['adni_ci']} | Brier: {r['adni_brier']}")
        print(f"   Delta AUC:        {r['delta_auc']} | Val Threshold: {r['threshold_val']}")

    # Save reconciled table
    df_res.to_csv(PROJECT_ROOT / "results/final/reconciled_model_leaderboard.csv", index=False)
    print("\nSaved to results/final/reconciled_model_leaderboard.csv")

if __name__ == "__main__":
    reconcile_full()
