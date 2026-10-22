# SHIFT-MRI V4 — Failures and Engineering Fixes Log

**Author:** Bhavy Garg (24BAI1088)  
**Session:** Overnight Execution & Revalidation (22–23 October 2026 IST)

This document records all scientific defects, code errors, and environment issues encountered during the V4 campaign, along with their root-cause diagnoses and authoritative solutions.

---

## 1. UnicodeEncodeError on Windows Console cp1252
- **Component:** `scripts/run_v4_campaign.py` line 886
- **Error:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u0394'` (Greek Delta $\Delta$) when printing metric summary to Windows PowerShell console.
- **Root Cause:** Standard Windows PowerShell terminal defaults to `cp1252` encoding instead of UTF-8, throwing an exception when printing unicode mathematical symbols.
- **Resolution:** Reconfigured standard output at script startup via `sys.stdout.reconfigure(encoding='utf-8')` and replaced terminal print string with ASCII `Delta_AUC`. Checkpoint and training state were fully preserved.

---

## 2. Classifier Head Attribute Order in FOMO25 ResEnc-B Adapter
- **Component:** `models/foundation_fomo25/amaes_adapter.py`
- **Error:** `AttributeError: 'AMAESResEncBAdapter' object has no attribute 'classifier'` during `freeze_encoder()` in `__init__`.
- **Root Cause:** In the modified `freeze_encoder()` method, `self.classifier.parameters()` was accessed before `self.classifier` was instantiated in the constructor body.
- **Resolution:** Reordered constructor initialization so that `self.classifier` is defined before calling `if freeze_backbone: self.freeze_encoder()`. Validated with forward/backward test on GPU.

---

## 3. High GPU VRAM Allocation in Swin UNETR SSL Partial Fine-Tuning
- **Component:** `scripts/run_v4_campaign.py` Swin UNETR fine-tuning loop
- **Symptom:** During batch size 4 forward/backward pass, peak VRAM reached 13.66 GB out of 15.93 GB available VRAM, nearing OOM threshold.
- **Resolution:** Set foundation model adaptation batch size to 2 for partial fine-tuning. Peak VRAM stabilized at ~3.8 GB, ensuring rock-solid numerical stability and zero OOM aborts.

---

## 4. Elimination of Random Initialization Fallbacks
- **Component:** `scripts/download_and_verify_brainiac.py`, `download_and_verify_swin.py`, `download_and_verify_fomo25.py`
- **Defect in Prior Commits:** Missing official weights silently fell back to generating random weights and saving them under official names.
- **Resolution:** Permanently eliminated all random fallback saves. Scripts now strictly raise `FileNotFoundError` if authentic official weights are missing. Added an explicit `Random-ViT-B16-Control` model architecture strictly labeled as an un-pretrained scientific baseline for Hypothesis H7.

---

## 5. Decision Threshold Contamination Prevention
- **Methodological Requirement:** Test labels must never be used to pick classification cutoffs.
- **Resolution:** Youden optimal threshold $\tau^*$ is computed strictly on source in-distribution OASIS-1 validation predictions ($N=30$) and frozen. The frozen threshold is applied blindly to locked OASIS-1 test ($N=30$), ADNI ($N=500$), and OASIS-2 ($N=150$).
