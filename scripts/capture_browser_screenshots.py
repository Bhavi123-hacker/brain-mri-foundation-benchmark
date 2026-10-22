import os
import subprocess
from pathlib import Path

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_path):
    chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

out_dir = Path("D:/ml project/docs/screenshots")
out_dir.mkdir(parents=True, exist_ok=True)

sections = [
    "overview",
    "leaderboard",
    "domain-shift",
    "interpretability",
    "training-dynamics",
    "trustgate-studio",
    "mri-explorer",
    "cnn-benchmark"
]

print(f"Using browser: {chrome_path}")

for sec in sections:
    url = f"http://127.0.0.1:8000/#{sec}"
    shot_path = out_dir / f"screenshot_{sec}.png"
    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1920,1080",
        "--virtual-time-budget=3000",
        f"--screenshot={str(shot_path)}",
        url
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        if shot_path.exists() and shot_path.stat().st_size > 0:
            print(f"SUCCESS: Captured {sec} -> {shot_path.name} ({shot_path.stat().st_size // 1024} KB)")
        else:
            print(f"WARNING: Screenshot file empty or not created for {sec}")
    except Exception as e:
        print(f"ERROR capturing {sec}: {e}")

print("Browser screenshot verification completed!")
