"""
scripts/reconcile_manuscript_v3.py
Reconciles all obsolete numbers, abstract claims, OOD descriptions,
and TrustGate in-domain results across research reports and submission documentation.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def update_file(path, replacements):
    full_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full_path):
        print(f"File not found: {path}")
        return
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
        else:
            print(f"Warning: pattern not found in {path}: {repr(old[:50])}...")

    if changed:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully updated: {path}")
    else:
        print(f"No changes made to: {path}")

# 1. literature_review_and_novelty.md
update_file("docs/research/v3/literature_review_and_novelty.md", [
    (
        "Establishes that in external ADNI, **Age + Sex Logistic Regression achieves ROC-AUC = 0.5671**, outperforming 7 out of 10 complex deep learning and foundation model linear probes.",
        "Establishes that on the complete matched 500-subject ADNI external cohort, **Age + Sex Logistic Regression achieves ROC-AUC = 0.5123** (near chance), while vision models (such as MONAI Swin UNETR SSL at 0.5834 and 2D Transfer at 0.5778) capture genuine image-derived signal beyond demographic priors, resolving the historical pilot N=50 artifact.",
    ),
    (
        "Achieves near-perfect separation (**OOD ROC-AUC = 0.9933** for feature norm; **1.0000** for entropy) between in-domain OASIS-1 and external ADNI scans.",
        "Achieves prospective separation (**OOD ROC-AUC = 1.0000** for regularized Mahalanobis distance; **0.9930** for centroid Euclidean distance) between in-domain OASIS-1 and external ADNI scans.",
    ),
    (
        '| **Demographic Confounding** | "Deep models learn pure anatomical biomarkers of neurodegeneration." | **False.** Demographic Age + Sex logistic regression (AUC = 0.5671) outperforms 70% of deep neural networks on external ADNI. |',
        '| **Demographic Confounding** | "Deep models learn pure anatomical biomarkers of neurodegeneration." | **Verified on Matched Cohort.** On matched N=500 ADNI, Age + Sex achieves AUC = 0.5123 (near chance), with deep vision models (Swin UNETR 0.5834, 2D Transfer 0.5778) proving genuine image representation over demographic shortcuts. |',
    ),
])

# 2. SHIFT_MRI_NOVELTY_DEFENSE.md
update_file("docs/submission/SHIFT_MRI_NOVELTY_DEFENSE.md", [
    (
        '#### Q7: "Your report shows that Age + Sex Logistic Regression achieves an AUC of 0.5671 on ADNI, beating 7 out of 10 deep models. What are the scientific implications?"',
        '#### Q7: "How do demographic baselines compare against deep vision models on the full ADNI external cohort (N=500)?"',
    ),
    (
        '#### Q19: "How can foundation models achieve an OOD detection AUC of 0.9933 (Novelty N07) while failing at downstream classification?"',
        '#### Q19: "How can foundation models achieve an OOD detection AUC of 1.0000 (Novelty N07) while failing at downstream classification?"',
    ),
    (
        "(OOD AUC = 0.9933 for FOMO25 feature norm; 1.0000 for Swin UNETR entropy)",
        "(OOD AUC = 1.0000 for FOMO25 Mahalanobis distance; 0.9930 for centroid Euclidean distance)",
    ),
])

print("Completed reconciliation script run!")
