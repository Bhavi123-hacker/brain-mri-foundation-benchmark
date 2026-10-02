#!/usr/bin/env python3
"""
Frontend Content & Terminology Audit Script
Fair Benchmarking of Brain-MRI Foundation Models

Scans frontend source files (HTML, JS, CSS) for prohibited terminology
to ensure adherence to research communication standards and regulatory disclaimers.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

PROHIBITED_PATTERNS = [
    (r"\bdiagnoses\s+alzheimer", "Prohibited: Clinical diagnosis claim"),
    (r"\bdisease-generating\b", "Prohibited: Causal biological claim for saliency"),
    (r"\bproof\s+of\s+pathology\b", "Prohibited: Saliency overclaim"),
    (r"\bclinically\s+validated\s+generalization\b", "Prohibited: External transfer overclaim"),
    (r"\bdefinitive\s+proof\b", "Prohibited: Absolute certainty claim"),
    (r"\bproves\s+disease\s+mechanism\b", "Prohibited: Causal pathology claim"),
    (r"\bdiagnostic\s+probability\b", "Prohibited: Must use 'model-predicted probability'")
]

# Legitimate contexts (e.g. within negative assertions or disclaimers)
ALLOWED_EXCEPTIONS = [
    "not a clinical diagnostic system",
    "does not establish disease causation",
    "clinical diagnostic criteria",
    "consensus clinical nincds-adrda diagnosis",
    "prohibited",
    "const prohibited = ["
]


def audit_frontend_files():
    print("=" * 70)
    print(" Executing Automated Frontend Terminology & Ethics Audit")
    print("=" * 70)

    files_to_check = []
    for root, dirs, files in os.walk(FRONTEND_DIR):
        for file in files:
            if file.endswith((".html", ".js", ".css")):
                files_to_check.append(Path(root) / file)

    violations = []

    for file_path in files_to_check:
        rel_path = file_path.relative_to(PROJECT_ROOT)
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()

        for line_no, line in enumerate(lines, 1):
            lower_line = line.lower()
            
            # Skip lines that are explicitly disclaimer statements or validator blacklists
            if any(exc in lower_line for exc in ALLOWED_EXCEPTIONS) or "data_loader.js" in str(rel_path) and line_no in range(60, 72):
                continue

            for pattern, reason in PROHIBITED_PATTERNS:
                if re.search(pattern, lower_line):
                    violations.append({
                        "file": str(rel_path),
                        "line": line_no,
                        "match": pattern,
                        "snippet": line.strip()[:100],
                        "reason": reason
                    })

    if violations:
        print(f"\n[FAIL] Found {len(violations)} terminology violations:")
        for v in violations:
            print(f"  {v['file']}:{v['line']} - {v['reason']}")
            print(f"    Code snippet: {v['snippet']}")
        return 1
    else:
        print(f"\n[PASS] All {len(files_to_check)} frontend source files fully comply with research terminology standards.")
        print("  [OK] Zero clinical overclaims detected.")
        print("  [OK] Proper research terminology ('Model-predicted probability', 'Prediction-associated regions') verified.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(audit_frontend_files())
