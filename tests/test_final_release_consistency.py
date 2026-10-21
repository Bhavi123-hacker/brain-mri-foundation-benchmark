"""
tests/test_final_release_consistency.py
Automated release-gate consistency tests verifying perfect synchronization between
authoritative CSV tables, canonical registry JSON, research reports, and documentation.
"""

import os
import re
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v3")
FINAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "final")
DOCS_FINAL_DIR = os.path.join(PROJECT_ROOT, "docs", "final")


def test_canonical_registry_matches_authoritative_csvs():
    """Verify that results/final/canonical_registry.json exactly matches current V3 CSVs."""
    reg_path = os.path.join(FINAL_RESULTS_DIR, "canonical_registry.json")
    assert os.path.exists(reg_path), "canonical_registry.json missing"

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # 1. Check ADNI rankings match cross_cohort_reliability_matrix.csv and demographic_baselines_results.csv
    rel_path = os.path.join(V3_RESULTS_DIR, "reliability", "cross_cohort_reliability_matrix.csv")
    df_rel = pd.read_csv(rel_path)
    adni_rel = df_rel[df_rel["Cohort"].str.contains("ADNI")].copy()

    demo_path = os.path.join(V3_RESULTS_DIR, "demographic_baselines", "demographic_baselines_results.csv")
    df_demo = pd.read_csv(demo_path)
    adni_demo = df_demo[df_demo["Cohort"].str.contains("ADNI")].copy()

    for item in registry["model_rankings_adni"]:
        if isinstance(item.get("rank"), int):
            model_name = item["model"]
            if model_name.startswith("Demographic"):
                # Match in demographic baselines
                match = adni_demo[adni_demo["Model"].str.contains(model_name.replace("Demographic: ", ""))]
                assert len(match) >= 1, f"Demographic model {model_name} missing from demographic table"
                csv_auc = float(match["ROC_AUC"].values[0])
                assert abs(item["auc"] - csv_auc) < 2e-3, f"AUC mismatch for {model_name}: reg={item['auc']}, csv={csv_auc}"
            else:
                # Match in reliability matrix (allowing suffix like ' (Linear Probe)')
                match = adni_rel[adni_rel["Model"].str.contains(re.escape(model_name.split(" (")[0]))]
                assert len(match) >= 1, f"Model {model_name} missing from reliability matrix"
                csv_auc = float(match["ROC_AUC"].values[0])
                assert abs(item["auc"] - csv_auc) < 2e-3, f"AUC mismatch for {model_name}: reg={item['auc']}, csv={csv_auc}"

    # 2. Check Demographic Baselines match demographic_baselines_results.csv
    adni_age_sex = df_demo[(df_demo["Cohort"].str.contains("ADNI")) & (df_demo["Model"] == "Demographic: Age + Sex")]
    assert len(adni_age_sex) == 1
    assert int(adni_age_sex["N"].values[0]) == 500, "ADNI demographic N must be 500"
    assert abs(float(adni_age_sex["ROC_AUC"].values[0]) - 0.5123) < 1e-3, "ADNI Age+Sex AUC must be ~0.5123"

    # 3. Check TrustGate Benchmark matches trustgate_benchmark_results.csv
    tg_path = os.path.join(V3_RESULTS_DIR, "trustgate", "trustgate_benchmark_results.csv")
    df_tg = pd.read_csv(tg_path)
    margin_row = df_tg[df_tg["Method"].str.contains("Margin")]
    assert len(margin_row) == 1
    assert abs(float(margin_row["ADNI_Error_Detection_AUC"].values[0]) - 0.5829) < 1e-3


def test_no_obsolete_pilot_numbers_in_primary_documents():
    """Verify that current reports and README do not present the old 0.5671 pilot number as authoritative current performance."""
    primary_docs = [
        os.path.join(PROJECT_ROOT, "README.md"),
        os.path.join(DOCS_FINAL_DIR, "RESEARCH_SOURCE_OF_TRUTH.md"),
        os.path.join(V3_RESULTS_DIR, "SHIFT_MRI_EXECUTIVE_HANDOFF.md"),
        os.path.join(V3_RESULTS_DIR, "SHIFT_MRI_FINAL_RESEARCH_REPORT.md"),
    ]

    for p in primary_docs:
        assert os.path.exists(p), f"Document missing: {p}"
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines, 1):
            if "0.5671" in line:
                # If 0.5671 appears, it MUST be explicitly labeled as historical / pilot / artifact
                assert any(w in line.lower() for w in ["pilot", "historical", "artifact", "resolving", "earlier"]), (
                    f"Line {idx} in {p} contains 0.5671 without explicit qualification as historical pilot artifact: {line}"
                )
                # It must NOT be represented as current ADNI Age+Sex performance in any table row
                assert not line.strip().startswith("|") or "historical" in line.lower() or "pilot" in line.lower(), (
                    f"Line {idx} in {p} presents 0.5671 in a table as current performance: {line}"
                )


def test_provenance_manifest_complete():
    """Verify that results/final/experiment_provenance_manifest.json exists and has valid records."""
    man_path = os.path.join(FINAL_RESULTS_DIR, "experiment_provenance_manifest.json")
    assert os.path.exists(man_path), "experiment_provenance_manifest.json missing"
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["provenance_records"]) >= 6
