"""
Experiment Registry for Final Research Optimization Campaign.
Manages:
- results/final_campaign/experiment_registry.csv
- results/final_campaign/experiment_registry.json
"""

import os
import sys
import json
import threading
import pandas as pd
from typing import Dict, Any, List

REGISTRY_CSV = "results/final_campaign/experiment_registry.csv"
REGISTRY_JSON = "results/final_campaign/experiment_registry.json"

REGISTRY_COLUMNS = [
    "experiment_id",
    "dataset",
    "model",
    "training_fraction",
    "seed",
    "preprocessing_version",
    "augmentation_version",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "epochs",
    "early_stopping",
    "trainable_parameters",
    "total_parameters",
    "training_time",
    "inference_time",
    "best_epoch",
    "validation_auc",
    "validation_accuracy",
    "validation_f1",
    "test_auc",
    "test_accuracy",
    "test_f1",
    "status",
    "notes",
]

_lock = threading.Lock()


class ExperimentRegistry:
    def __init__(self, csv_path: str = REGISTRY_CSV, json_path: str = REGISTRY_JSON):
        self.csv_path = csv_path
        self.json_path = json_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._init_files()

    def _init_files(self):
        with _lock:
            if not os.path.exists(self.csv_path):
                df = pd.DataFrame(columns=REGISTRY_COLUMNS)
                df.to_csv(self.csv_path, index=False)
            if not os.path.exists(self.json_path):
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)

    def register_run(self, run_data: Dict[str, Any]) -> str:
        with _lock:
            # Load existing
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []

            # Ensure all columns present
            clean_entry = {}
            for col in REGISTRY_COLUMNS:
                clean_entry[col] = run_data.get(col, None)

            # Auto-generate experiment_id if not present
            if not clean_entry.get("experiment_id"):
                exp_id = f"EXP_{len(data)+1:04d}_{clean_entry.get('model', 'MODEL')}_{clean_entry.get('seed', 0)}"
                clean_entry["experiment_id"] = exp_id

            data.append(clean_entry)

            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            df = pd.DataFrame(data, columns=REGISTRY_COLUMNS)
            df.to_csv(self.csv_path, index=False)

            return clean_entry["experiment_id"]

    def get_all_runs(self) -> List[Dict[str, Any]]:
        with _lock:
            if os.path.exists(self.json_path):
                with open(self.json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []


# Global singleton
registry = ExperimentRegistry()
