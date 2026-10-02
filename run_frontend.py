#!/usr/bin/env python3
"""
Research Benchmark Frontend Server
Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines

Serves the interactive research dashboard, canonical JSON data, and visual assets.
Supports standard CLI running and an automated --test smoke check.
"""

import argparse
import http.server
import json
import mimetypes
import os
import socketserver
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CANONICAL_DATA_DIR = PROJECT_ROOT / "results" / "final_campaign" / "canonical_frontend_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "final_campaign"

# Ensure MIME types are recognized correctly
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/svg+xml", ".svg")


class ThreadingBenchmarkServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server to handle high-concurrency browser requests without socket blocking."""
    daemon_threads = True
    request_queue_size = 64
    allow_reuse_address = True


class BenchmarkRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler routing frontend assets, canonical data, and results."""

    def end_headers(self):
        # Enable CORS and caching headers appropriate for local research dashboard
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def translate_path(self, path):
        """Map URL paths to disk locations:
        - / or /index.html -> frontend/index.html
        - /css/..., /js/..., /vendor/... -> frontend/...
        - /data/... or data/... -> results/final_campaign/canonical_frontend_data/...
        - /results/final_campaign/... -> results/final_campaign/...
        - Everything else -> frontend/... or PROJECT_ROOT/...
        """
        # Clean query strings and fragments
        path = path.split("?", 1)[0].split("#", 1)[0]
        norm_path = urllib.parse.unquote(path)
        clean_path = norm_path.lstrip("/")

        if clean_path == "" or clean_path == "index.html":
            return str(FRONTEND_DIR / "index.html")

        if clean_path.startswith("data/") or clean_path.startswith("canonical_data/"):
            subpath = clean_path.split("/", 1)[1]
            return str(CANONICAL_DATA_DIR / subpath)

        if clean_path.startswith("results/"):
            return str(PROJECT_ROOT / clean_path)

        # Try inside frontend directory first
        frontend_target = FRONTEND_DIR / clean_path
        if frontend_target.exists():
            return str(frontend_target)

        # Fallback to project root
        return str(PROJECT_ROOT / clean_path)

    def log_message(self, format, *args):
        # Suppress routine log messages unless in debug mode
        if os.environ.get("BENCHMARK_DEBUG") == "1":
            super().log_message(format, *args)


def run_smoke_test(port: int = 8088):
    """Start server in thread, test all critical endpoints, verify response status and payload."""
    print(f"Starting test server on port {port}...")
    server = ThreadingBenchmarkServer(("127.0.0.1", port), BenchmarkRequestHandler)
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
    parser = argparse.ArgumentParser(description="Research Benchmark Frontend Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--test", action="store_true", help="Run automated HTTP smoke tests and exit")
    args = parser.parse_args()

    if args.test:
        sys.exit(run_smoke_test(port=8089))

    print("=" * 70)
    print(" Research Benchmark Interactive Web Application Server")
    print(" Fair Benchmarking of Brain-MRI Foundation Models")
    print("=" * 70)
    print(f" Web Application URL : http://{args.host}:{args.port}")
    print(f" Frontend Root       : {FRONTEND_DIR}")
    print(f" Canonical Data Root : {CANONICAL_DATA_DIR}")
    print(" Status              : FROZEN AUDITED BENCHMARK")
    print(" Disclaimer          : Research Use Only. Not for Clinical Diagnostic Use.")
    print("=" * 70)
    print("Press Ctrl+C to terminate the server.\n")

    with ThreadingBenchmarkServer((args.host, args.port), BenchmarkRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer shutting down gracefully.")
            httpd.shutdown()


if __name__ == "__main__":
    main()
