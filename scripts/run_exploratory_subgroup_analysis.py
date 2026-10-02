"""
Priority 7 / Experiment 7: Exploratory Subgroup Analysis.
Evaluates model performance across demographic and acquisition subgroups:
- Sex (M vs F)
- Age group (< 75 vs >= 75)
- Scanner Manufacturer (Philips Medical Systems, GE Medical Systems, Siemens)
- Field Strength (1.5T vs 3.0T)

Outputs:
- results/extensions/statistics/subgroup_results.csv
- results/extensions/statistics/subgroup_results.json
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger


def extract_adni_metadata():
    # 1. Load diagnosis audit for age and sex
    diag_df = pd.read_csv("results/adni_diagnosis_audit.csv")
    subject_meta = {}
    for _, row in diag_df.iterrows():
        sid = row["subject_id"]
        notes = str(row.get("notes", ""))
        sex = "Unknown"
        age = np.nan
        m_sex = re.search(r"Sex\s+([MF])", notes)
        if m_sex:
            sex = m_sex.group(1)
        m_age = re.search(r"Age\s+([0-9\.]+)", notes)
        if m_age:
            age_str = m_age.group(1).rstrip(".")
            age = float(age_str)

        subject_meta[sid] = {
            "subject_id": sid,
            "diagnosis": row["baseline_diagnosis"],
            "sex": sex,
            "age": age,
            "age_group": "<75" if age < 75.0 else ">=75" if not np.isnan(age) else "Unknown",
        }

    # 2. Load series audit for manufacturer and field strength
    series_df = pd.read_csv("results/adni_series_audit.csv")
    for _, row in series_df.iterrows():
        sid = row["subject_id"]
        if sid in subject_meta:
            mfg = str(row.get("manufacturer", "Unknown")).strip()
            fs = str(row.get("field_strength", "Unknown")).strip()
            subject_meta[sid]["manufacturer"] = mfg
            subject_meta[sid]["field_strength"] = f"{fs}T" if "T" not in fs else fs

    return subject_meta


def run_subgroup_analysis():
    logger = get_logger("SubgroupAnalysis")
    logger.info("=================================================================")
    logger.info("Starting Priority 7: Exploratory Subgroup Analysis")
    logger.info("=================================================================")

    output_dir = "results/extensions/statistics"
    os.makedirs(output_dir, exist_ok=True)

    meta = extract_adni_metadata()

    # Load test results
    with open("results/adni_locked_test_results.json", "r", encoding="utf-8") as f:
        test_res = json.load(f)

    # In ADNI test set: 8 subjects
    test_split_df = pd.read_csv("results/adni_cn_ad_split.csv")
    test_sids = test_split_df[test_split_df["split"] == "test"]["subject_id"].tolist()
    y_true = np.array(test_res["ground_truth_labels"])

    # Map each test subject to metadata
    test_records = []
    for idx, sid in enumerate(test_sids):
        s_meta = meta.get(sid, {})
        test_records.append({
            "subject_id": sid,
            "index": idx,
            "ground_truth": y_true[idx],
            "sex": s_meta.get("sex", "Unknown"),
            "age": s_meta.get("age", np.nan),
            "age_group": s_meta.get("age_group", "Unknown"),
            "manufacturer": s_meta.get("manufacturer", "Unknown"),
            "field_strength": s_meta.get("field_strength", "Unknown"),
        })
    test_df = pd.DataFrame(test_records)

    # Models to analyze
    models = test_res["models"]

    subgroup_keys = ["sex", "age_group", "manufacturer", "field_strength"]
    results = []

    for sg_var in subgroup_keys:
        unique_vals = test_df[sg_var].dropna().unique()
        for val in sorted(unique_vals):
            mask = (test_df[sg_var] == val).values
            n_sub = int(np.sum(mask))
            if n_sub == 0:
                continue

            sub_y = y_true[mask]
            cn_count = int(np.sum(sub_y == 0))
            ad_count = int(np.sum(sub_y == 1))

            for model_name, m_data in models.items():
                probs = np.array(m_data["ensemble_evaluation"]["probabilities"])[mask]
                preds = (probs >= 0.5).astype(int)

                acc = float(accuracy_score(sub_y, preds))
                bal_acc = float(balanced_accuracy_score(sub_y, preds)) if (cn_count > 0 and ad_count > 0) else np.nan
                auc = float(roc_auc_score(sub_y, probs)) if (cn_count > 0 and ad_count > 0) else np.nan

                rec = {
                    "cohort": "ADNI MPRAGE (Test N=8)",
                    "subgroup_variable": sg_var,
                    "subgroup_value": str(val),
                    "subgroup_n": n_sub,
                    "cn_count": cn_count,
                    "ad_count": ad_count,
                    "model": model_name,
                    "accuracy": round(acc, 4),
                    "balanced_accuracy": round(bal_acc, 4) if not np.isnan(bal_acc) else None,
                    "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                    "mean_predicted_prob": round(float(np.mean(probs)), 4),
                    "caveat": "Exploratory pilot subgroup analysis with very small sample size (N <= 8). High statistical variance expected.",
                }
                results.append(rec)
                logger.info(f"[{sg_var}={val} | N={n_sub} (CN={cn_count}, AD={ad_count})] {model_name}: Acc={acc:.2f}, AUC={auc if not np.isnan(auc) else 'N/A'}")

    # Save outputs
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "subgroup_results.csv")
    res_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved subgroup results CSV to {csv_path}")

    json_path = os.path.join(output_dir, "subgroup_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved subgroup results JSON to {json_path}")

    return res_df


if __name__ == "__main__":
    run_subgroup_analysis()
