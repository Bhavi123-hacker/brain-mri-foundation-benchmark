"""
Creates deterministic, reproducible data-efficiency fraction splits on the OASIS-1 pilot training pool.
Fractions:
  - 10%:  4 subjects  (2 CN, 2 Dementia, ~11.4% actual)
  - 25%:  9 subjects  (4 CN, 5 Dementia, ~25.7% actual)
  - 50%: 18 subjects  (9 CN, 9 Dementia, ~51.4% actual)
  - 100%: 35 subjects (17 CN, 18 Dementia, 100.0% actual)
Seeds: 42, 43, 44, 45, 46
Fixed Evaluation Pools:
  - Validation: 7 subjects (4 CN, 3 Dementia) - NEVER REDUCED
  - Locked Test: 8 subjects (4 CN, 4 Dementia) - NEVER REDUCED, EVALUATED ONLY ONCE
Outputs:
  - results/extensions/data_efficiency/oasis_data_fraction_splits.csv
  - results/extensions/data_efficiency/oasis_data_fraction_splits.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

def create_oasis_data_efficiency_splits(
    splits_csv: str = "D:/OASIS1_PILOT/pilot_splits.csv",
    output_dir: str = "results/extensions/data_efficiency",
    seeds: list = [42, 43, 44, 45, 46],
):
    os.makedirs(output_dir, exist_ok=True)
    assert os.path.exists(splits_csv), f"Missing splits: {splits_csv}"
    df = pd.read_csv(splits_csv)

    # Exclude forbidden file if present
    df = df[~df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)

    train_pool = df[df["split"] == "train"].sort_values("ID").reset_index(drop=True)
    val_pool = df[df["split"] == "val"].sort_values("ID").reset_index(drop=True)
    test_pool = df[df["split"] == "test"].sort_values("ID").reset_index(drop=True)

    assert len(train_pool) == 35, f"Expected 35 train, got {len(train_pool)}"
    assert len(val_pool) == 7, f"Expected 7 val, got {len(val_pool)}"
    assert len(test_pool) == 8, f"Expected 8 test, got {len(test_pool)}"

    cn_train = train_pool[train_pool["label"] == 0.0].reset_index(drop=True)
    dem_train = train_pool[train_pool["label"] == 1.0].reset_index(drop=True)

    fractions_spec = {
        "10%": {"fraction_requested": 0.10, "n_cn": 2, "n_dem": 2, "n_train": 4},
        "25%": {"fraction_requested": 0.25, "n_cn": 4, "n_dem": 5, "n_train": 9},
        "50%": {"fraction_requested": 0.50, "n_cn": 9, "n_dem": 9, "n_train": 18},
        "100%": {"fraction_requested": 1.00, "n_cn": 17, "n_dem": 18, "n_train": 35},
    }

    all_records = []
    json_summary = {
        "cohort": "OASIS-1 Pilot 50 Subjects",
        "training_pool_total": len(train_pool),
        "validation_pool_fixed": len(val_pool),
        "test_pool_fixed": len(test_pool),
        "fractions": {},
    }

    for frac_key, spec in fractions_spec.items():
        json_summary["fractions"][frac_key] = {
            "fraction_requested": spec["fraction_requested"],
            "fraction_actual": round(spec["n_train"] / len(train_pool), 4),
            "n_train": spec["n_train"],
            "n_CN": spec["n_cn"],
            "n_DEMENTIA": spec["n_dem"],
            "seeds": {},
        }

        for s in seeds:
            rng = np.random.RandomState(s)
            if frac_key == "100%":
                selected_cn = cn_train
                selected_dem = dem_train
            else:
                idx_cn = rng.choice(len(cn_train), size=spec["n_cn"], replace=False)
                idx_dem = rng.choice(len(dem_train), size=spec["n_dem"], replace=False)
                selected_cn = cn_train.iloc[idx_cn]
                selected_dem = dem_train.iloc[idx_dem]

            selected_train = pd.concat([selected_cn, selected_dem]).sort_values("ID").reset_index(drop=True)
            sampled_ids = selected_train["ID"].tolist()

            json_summary["fractions"][frac_key]["seeds"][str(s)] = {
                "subjects": sampled_ids,
                "n_actual": len(sampled_ids),
                "cn_count": int((selected_train["label"] == 0.0).sum()),
                "dementia_count": int((selected_train["label"] == 1.0).sum()),
            }

            for _, row in selected_train.iterrows():
                all_records.append({
                    "fraction": frac_key,
                    "fraction_requested": spec["fraction_requested"],
                    "fraction_actual": round(spec["n_train"] / len(train_pool), 4),
                    "seed": s,
                    "subject_id": row["ID"],
                    "mri_file": row["mri_file"],
                    "label": float(row["label"]),
                    "label_name": row["label_name"],
                    "Age": row["Age"],
                    "CDR": row["CDR"],
                    "split": "train",
                })

    splits_manifest_df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "oasis_data_fraction_splits.csv")
    splits_manifest_df.to_csv(csv_path, index=False)
    print(f"Saved fraction splits CSV to {csv_path} (Total Rows: {len(splits_manifest_df)})")

    json_path = os.path.join(output_dir, "oasis_data_fraction_splits.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved fraction splits JSON to {json_path}")

    return splits_manifest_df, json_summary

if __name__ == "__main__":
    create_oasis_data_efficiency_splits()
