"""
Leaderboard Generation and Multi-Model Comparison Engine.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional


class BenchmarkLeaderboard:
    """
    Maintains and renders the unified leaderboard comparing Supervised Baselines
    against Foundation Models across label budgets, compute, and tasks.
    """
    def __init__(self, output_csv: str = "results/leaderboard.csv"):
        self.output_csv = output_csv
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def add_entry(
        self,
        model_name: str,
        dataset_name: str,
        task_name: str,
        label_budget: float,
        seed: int,
        metrics: Dict[str, Any],
        compute: Dict[str, Any],
    ) -> None:
        """Add a completed experimental run record."""
        entry = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Task": task_name,
            "Label_Budget": f"{int(label_budget * 100)}%",
            "Seed": seed,
            "ROC_AUC": metrics.get("roc_auc", None),
            "Accuracy": metrics.get("accuracy", None),
            "F1_Score": metrics.get("f1", None),
            "Sensitivity": metrics.get("sensitivity", None),
            "Specificity": metrics.get("specificity", None),
            "MAE": metrics.get("mae", None),
            "RMSE": metrics.get("rmse", None),
            "R2": metrics.get("r2", None),
            "GPU_Hours": compute.get("gpu_hours", 0.0),
            "Training_Sec": compute.get("elapsed_seconds", 0.0),
            "Peak_VRAM_MB": compute.get("peak_vram_mb", 0.0),
            "Params_Total": compute.get("total_parameters", None),
            "Latency_ms": metrics.get("latency_ms_per_sample", None),
        }
        self.records.append(entry)

    def save(self) -> pd.DataFrame:
        """Save leaderboard records to CSV."""
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_csv, index=False)
        return df

    def to_markdown(self) -> str:
        """Generate a formatted markdown table."""
        df = pd.DataFrame(self.records)
        if df.empty:
            return "_No experiments recorded yet._"
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)
