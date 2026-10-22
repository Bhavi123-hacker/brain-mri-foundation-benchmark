import subprocess
from pathlib import Path

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
sections = [
    "overview",
    "leaderboard",
    "domain-shift",
    "interpretability",
    "training-dynamics",
    "cnn-benchmark",
    "trustgate-studio",
    "mri-explorer"
]

print("=== VERIFYING REAL BROWSER DOM RENDERING ===")
for sec in sections:
    url = f"http://127.0.0.1:8000/#{sec}"
    res = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--virtual-time-budget=4000", "--dump-dom", url],
        capture_output=True,
        timeout=15
    )
    dom = res.stdout.decode("utf-8", errors="ignore")
    assert len(dom) > 1000, f"DOM too short for {sec} ({len(dom)} chars)"
    assert "fatal-error" not in dom, f"Fatal error found in {sec}"
    
    if sec == "domain-shift":
        assert "Ten-Architecture" in dom, "Missing Ten-Architecture table in domain-shift"
        assert "MONAI Swin UNETR SSL" in dom, "Missing Swin UNETR in domain-shift"
        assert "Historical Feasibility Pilot Archive" in dom, "Missing pilot archive in domain-shift"
    elif sec == "interpretability":
        assert 'id="axial-img"' in dom, "Missing axial-img in interpretability"
        assert 'id="composite-img"' in dom, "Missing composite-img in interpretability"
        assert "results/final_campaign/interpretability/" in dom, "Missing composite image src in interpretability"
    elif sec == "training-dynamics":
        assert "Historical Development Diagnostics Archive" in dom, "Missing archive notice in training-dynamics"
        assert "train_vs_val.png" in dom, "Missing training curve images in training-dynamics"
    elif sec == "cnn-benchmark":
        assert "Authoritative V3 Benchmark Evaluation" in dom, "Missing V3 notice in cnn-benchmark"

    print(f"PASS: #{sec} rendered correctly ({len(dom)} chars, 0 errors)")

print("=== ALL BROWSER DOM CHECKS PASSED SUCCESSFULLY ===")
