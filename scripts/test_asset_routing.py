import urllib.request
import urllib.error
import io
from PIL import Image

BASE = "http://127.0.0.1:8000"

test_images = [
    "/results/final_campaign/figures/01_model_auc_comparison.png",
    "/results/final_campaign/training_diagnostics/train_vs_validation_curves/resnet18_oasis_train_vs_val.png",
    "/results/final_campaign/frontend_data/interpretability/adni/resnet18/002_S_0816/axial_overlay.png",
    "/results/final_campaign/frontend_data/interpretability/adni/resnet18/002_S_0816/axial_original.png",
    "/results/final_campaign/frontend_data/interpretability/adni/resnet18/002_S_0816/axial_cam.png",
    "/results/final_campaign/interpretability/adni_resnet18_gradcam/002_S_0816_false_negative.png",
    "/results/final_campaign/foundation_adaptation/adaptation_curves/swinunetr_linear_probe_train_vs_val.png"
]

print("=== 1. VERIFYING AUTHORIZED IMAGE ASSETS ===")
for path in test_images:
    url = BASE + path
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            ctype = resp.headers.get("Content-Type")
            data = resp.read()
            img = Image.open(io.BytesIO(data))
            assert img.format == "PNG", f"Expected PNG format, got {img.format}"
            print(f"PASS: {path} -> Status {status}, Content-Type: {ctype}, Size: {len(data)} bytes, Dimensions: {img.size}")
    except Exception as e:
        print(f"FAIL: {path} -> {e}")
        raise

print("\n=== 2. VERIFYING 404 ON MISSING / UNAUTHORIZED ASSETS ===")
invalid_paths = [
    "/results/final_campaign/figures/non_existent_figure.png",
    "/results/final_campaign/training_diagnostics/train_vs_validation_curves/missing.png",
    "/results/final_campaign/training_diagnostics/train_vs_validation_curves/script.py",
    "/frontend/js/app.py",
    "/results/final_campaign/secret_data.csv"
]

for path in invalid_paths:
    url = BASE + path
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            print(f"FAIL: {path} -> returned {resp.status} instead of 404! Content-Type: {resp.headers.get('Content-Type')}")
            raise AssertionError(f"Expected 404 for {path}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"PASS: {path} -> correctly returned HTTP 404 Not Found")
        else:
            print(f"PASS: {path} -> returned HTTP {e.code}")

print("\n=== ALL ASSET ROUTING TESTS PASSED! ===")
