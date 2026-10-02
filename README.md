# Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines

A reproducible, fair benchmarking harness designed to evaluate public **3D Brain-MRI Foundation Models** against **well-tuned supervised 3D deep learning baselines** across controlled label budgets, matching compute constraints, and multi-seed statistical testing.

---

## Key Benchmarking Principles

1. **Fairness First:** Baselines and foundation models receive identical hyperparameter tuning budgets (search space, trials, validation protocol).
2. **Label Efficiency:** Models are evaluated across strictly controlled training label budgets: **10%, 25%, 50%, and 100%**.
3. **Leakage-Free Splitting:** Strict **patient-level partitioning** ensures that no longitudinal scans from the same subject cross train/val/test splits.
4. **Matched Compute Profiling:** Every experiment measures training wall-clock time, GPU hours, peak VRAM, active parameters, and inference latency per sample.
5. **Multi-Seed Statistical Rigor:** All core configurations are evaluated across 3 fixed random seeds (42, 123, 999) reporting mean and standard deviation.

---

## Project Directory Structure

```text
brain_mri_benchmark/
├── configs/                  # Modular YAML experiment specifications
│   ├── datasets.yaml         # ADNI, OASIS-3, and mock dataset configurations
│   ├── models.yaml           # Baseline 3D ResNet and Foundation Model hyperparams
│   ├── tasks.yaml            # AD_vs_CN, Brain_Age, MCI_vs_CN task definitions
│   └── benchmark.yaml        # Global seeds, budgets, stages A-E, fairness protocols
├── data/
│   ├── adni/                 # [Configured path] ADNI dataset root (git-ignored)
│   ├── oasis3/               # [Configured path] OASIS-3 dataset root (git-ignored)
│   ├── mock_data/            # Synthetic phantoms for testing and CI
│   └── splits/               # Patient-level Train/Val/Test split CSVs
├── preprocessing/            # MONAI & NiBabel 3D medical preprocessing
│   ├── transforms.py         # Resampling, Z-score, resizing, 3D augmentations
│   ├── pipeline.py           # Patient splitting & PyTorch BrainMRIDataset
│   └── mock_data.py          # Synthetic 3D NIfTI brain phantom generator
├── models/                   # Architecture implementations & adapters
│   ├── base.py               # Abstract BaseBenchModel interface
│   ├── baseline/             # 3D ResNet-18/34/50 supervised models
│   ├── foundation_brainiac/  # BrainIAC ViT foundation adapter
│   ├── foundation_fomo25/    # FOMO25 Winner 3D SSL adapter
│   └── foundation_swin/      # MONAI Swin UNETR SSL adapter
├── tasks/                    # Downstream loss & metric handlers
│   ├── classification.py     # Binary BCE & probability extraction
│   └── regression.py         # Continuous MSE & prediction extraction
├── training/                 # Optimization & training engine
│   ├── budget_sampler.py     # Deterministic 10%, 25%, 50%, 100% patient sampler
│   ├── optimizer.py          # Fair AdamW/Cosine optimizer & scheduler builder
│   └── trainer.py            # Benchmark trainer with early stopping & AMP
├── evaluation/               # Metrics & compute profiling
│   ├── metrics.py            # ROC-AUC, F1, Sensitivity, Specificity, MAE, R2
│   ├── profiler.py           # GPU hours, peak VRAM, latency tracker
│   └── evaluator.py          # Standardized test set evaluation runner
├── experiments/              # Orchestration & validation
│   ├── validator.py          # Configuration matrix validation
│   └── runner.py             # Staged execution runner
├── visualization/            # Visuals & leaderboard
│   ├── leaderboard.py        # Leaderboard CSV & Markdown compiler
│   └── plots.py              # ROC curves, label efficiency, scatter plots
├── utils/                    # Core utilities
│   ├── config_parser.py      # YAML parser with env var overrides
│   ├── logging_utils.py      # Structured file and console logging
│   └── seed.py               # Multi-seed reproducibility
├── tests/                    # Pytest unit & integration test suite
├── results/                  # Checkpoints, logs, figures, and leaderboard
├── requirements.txt          # Python environment dependencies
└── README.md                 # Project documentation
```

---

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

---

## Synthetic Mock Data Generation

For local CPU development, automated tests, and CI, generate synthetic 3D MRI phantoms:

```bash
python preprocessing/mock_data.py
```

*Note: Synthetic data contains geometric phantoms with simulated anatomical structures (skull, parenchyma, CSF cavities) and simulated longitudinal metadata for pipeline validation only.*

---

## Connecting Real ADNI & OASIS-3 Datasets

Access-controlled medical datasets are never committed to version control and are mounted externally:

### Option A: Environment Variables (Recommended)
```bash
# ADNI Dataset
export ADNI_DATA_DIR="/path/to/adni/scans"
export ADNI_METADATA_PATH="/path/to/adni/metadata.csv"

# OASIS-3 Dataset
export OASIS3_DATA_DIR="/path/to/oasis3/scans"
export OASIS3_METADATA_PATH="/path/to/oasis3/metadata.csv"
```

### Option B: Modify `configs/datasets.yaml`
Update `root_dir` and `metadata_path` under `adni` or `oasis3` sections directly.

---

## Staged Execution Workflow

| Stage | Name | Target Environment | Purpose |
| :--- | :--- | :--- | :--- |
| **Stage A** | **Smoke Test** | Local CPU | 1-epoch verification on miniature synthetic phantoms |
| **Stage B** | **Pilot Experiment** | Single GPU (Colab/Kaggle/VM) | Convergence and VRAM profiling on real subset |
| **Stage C** | **Fair Tuning** | Remote GPU | 8-trial hyperparameter sweep per model/task on Val set |
| **Stage D** | **Reduced Benchmark** | Remote GPU | Preliminary label-efficiency check (10% and 100% budget) |
| **Stage E** | **Full Benchmark** | Remote GPU Cluster | Complete 144-run benchmark across all tasks, budgets, and seeds |

To execute the Stage A local CPU smoke test:
```bash
python experiments/runner.py
```
