#!/usr/bin/env python3
"""
Research Benchmark Frontend & API Server
Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines
SHIFT-MRI: Source-Locked, Label-Aware Cross-Cohort Reliability and Failure Analysis

Serves the interactive research dashboard, orthogonal MRI slices, multi-model predictions,
TrustGate failure risk analysis, and downloadable research deliverables.
Supports standard CLI execution and automated --test smoke checks.
"""

import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.server import ThreadingBenchmarkServer, SHIFTMRIRequestHandler, FRONTEND_DIR, CANONICAL_DATA_DIR


def run_smoke_test(port: int = 8088):
    """Start server in thread, test all critical endpoints, verify response status and payload."""
    print(f"Starting test server on port {port}...")
    server = ThreadingBenchmarkServer(("127.0.0.1", port), SHIFTMRIRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    endpoints = [
        ("/", 200, "text/html"),
        ("/data/study_metadata.json", 200, "application/json"),
        ("/data/models.json", 200, "application/json"),
        ("/data/leaderboard.json", 200, "application/json"),
        ("/data/audit_status.json", 200, "application/json"),
        ("/data/interpretability_index.json", 200, "application/json"),
        ("/data/training_diagnostics.json", 200, "application/json"),
        ("/data/foundation_adaptation.json", 200, "application/json"),
        ("/data/statistical_uncertainty.json", 200, "application/json"),
        ("/data/adaptation_comparison.json", 200, "application/json"),
        ("/results/final_campaign/figures/01_model_auc_comparison.png", 200, "image/png"),
        # SHIFT-MRI V3 endpoints
        ("/api/registry", 200, "application/json"),
        ("/api/mri/scans", 200, "application/json"),
        ("/downloads/SHIFT_MRI_Complete_Research_Paper.pdf", 200, "application/pdf"),
    ]

    all_passed = True
    print("\nExecuting HTTP Smoke Tests:")
    for url_path, expected_status, expected_mime in endpoints:
        full_url = f"http://127.0.0.1:{port}{url_path}"
        try:
            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                if status == expected_status and expected_mime in content_type:
                    print(f"  [PASS] {url_path} -> HTTP {status} ({content_type})")
                else:
                    print(f"  [FAIL] {url_path} -> HTTP {status} (Expected {expected_status}, got {content_type})")
                    all_passed = False
        except Exception as e:
            print(f"  [ERROR] {url_path} -> {e}")
            all_passed = False

    server.shutdown()
    server.server_close()

    if all_passed:
        print("\nAll Smoke Tests Passed Successfully!\n")
        return 0
    else:
        print("\nSmoke Tests Failed!\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="SHIFT-MRI Research Benchmark Interactive Application Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--test", action="store_true", help="Run automated HTTP smoke tests and exit")
    args = parser.parse_args()

    if args.test:
        sys.exit(run_smoke_test(port=8089))

    print("=" * 75)
    print(" SHIFT-MRI: Research Web Platform & Real Inference Server")
    print(" Fair Benchmarking of Brain-MRI Foundation Models")
    print("=" * 75)
    print(f" Web Application URL : http://{args.host}:{args.port}")
    print(f" Frontend Root       : {FRONTEND_DIR}")
    print(f" Canonical Data Root : {CANONICAL_DATA_DIR}")
    print(" Status              : FROZEN AUDITED BENCHMARK (Version 3.0.0-final)")
    print(" Disclaimer          : Research Use Only. Not for Clinical Diagnostic Use.")
    print("=" * 75)
    print("Press Ctrl+C to terminate the server.\n")

    with ThreadingBenchmarkServer((args.host, args.port), SHIFTMRIRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer shutting down gracefully.")
            httpd.shutdown()


if __name__ == "__main__":
    main()
