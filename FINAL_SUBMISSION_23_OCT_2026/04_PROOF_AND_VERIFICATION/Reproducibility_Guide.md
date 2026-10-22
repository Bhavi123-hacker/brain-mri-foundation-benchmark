# SHIFT-MRI V4 — Experimental Reproducibility Guide

**Author:** Bhavy Garg (24BAI1088), Vellore Institute of Technology, Chennai  

To reproduce all V4 experimental outcomes from scratch:

```powershell
# 1. Activate Environment
cd "D:\ml project"
.\.venv\Scripts\Activate.ps1

# 2. Verify Hardware & Tensors
python scripts/check_all_tensors_exist.py

# 3. Execute Unified V4 Training Campaign (Seed 42)
python scripts/run_v4_campaign.py

# 4. Execute Double Verification Pass
python scripts/run_double_verification.py

# 5. Run Full Test Suite
pytest tests/ -v
```

All models train deterministically with PyTorch seed 42, AdamW optimizer, and Cosine Annealing learning rate schedules. Outputs are written directly to `results/v4/`.
