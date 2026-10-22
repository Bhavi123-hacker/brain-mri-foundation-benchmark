import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("D:/ml project")
sys.path.insert(0, str(PROJECT_ROOT))

from backend.server import LIVE_RUNNER, get_volume_array, SCAN_REPO

def test_live_models():
    print("=== LIVE CHECKPOINT INFERENCE AUDIT ===")
    print("Runner Device:", LIVE_RUNNER.device)
    print("Preloaded Models:", list(LIVE_RUNNER.models.keys()))

    # Find a scan with valid local npy
    test_scan = None
    for s in SCAN_REPO.scans:
        if s.get("has_npy") and os.path.exists(s.get("npy_path", "")):
            test_scan = s
            break
            
    assert test_scan is not None, "No local MRI scan found with .npy file!"
    print(f"\nUsing Test Scan: {test_scan['scan_id']} ({test_scan['cohort']})")
    print(f"Path: {test_scan['npy_path']}")

    vol = get_volume_array(test_scan['npy_path'])
    print(f"Loaded Volume Shape: {vol.shape}, dtype: {vol.dtype}, min: {vol.min():.2f}, max: {vol.max():.2f}")

    # Test each model individually
    for model_name, model in LIVE_RUNNER.models.items():
        print(f"\n--- Testing Model: {model_name} ---")
        ckpt_path = LIVE_RUNNER.checkpoints.get(model_name)
        print(f"Checkpoint: {ckpt_path} (Exists: {os.path.exists(ckpt_path)})")
        
        t0 = time.time()
        with torch.no_grad():
            x = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(LIVE_RUNNER.device)
            # Verify input shape
            print(f"Input Tensor Shape: {tuple(x.shape)}")
            
            if "Dual-Branch" in model_name:
                from preprocessing.enhanced_pipeline import AnatomicalROIExtractor
                extractor = AnatomicalROIExtractor()
                tabular = extractor.extract(vol).to(LIVE_RUNNER.device).unsqueeze(0)
                print(f"Extracted Tabular Morphometry Shape: {tuple(tabular.shape)}")
                out = model(x, tabular)
            else:
                out = model(x)
                
            prob = float(torch.sigmoid(out.view(-1))[0].item())
            runtime_ms = round((time.time() - t0) * 1000, 2)
            
        print(f"Predicted Probability: {prob:.4f}")
        print(f"Runtime: {runtime_ms} ms")
        print(f"Device: {LIVE_RUNNER.device}")
        assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"

    print("\n--- Testing Full LIVE_RUNNER.run_live_forward_pass ---")
    summary = LIVE_RUNNER.run_live_forward_pass(vol)
    print(json.dumps(summary, indent=2))
    assert len(summary["models_evaluated_live"]) == 4

if __name__ == "__main__":
    test_live_models()
