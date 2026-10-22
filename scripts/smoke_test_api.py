import urllib.request
import json
import sys

def main():
    print("=== SHIFT-MRI API SMOKE VERIFICATION ===")
    
    # 1. Health
    req = urllib.request.urlopen('http://127.0.0.1:8000/api/health')
    health = json.loads(req.read())
    print("[1] /api/health:", health["status"], "| Models:", health["benchmarked_models"], "| ADNI:", health["adni_cohort_n"])
    assert health["status"] == "healthy"
    assert health["benchmarked_models"] == 10

    # 2. Registry
    req = urllib.request.urlopen('http://127.0.0.1:8000/api/registry')
    reg = json.loads(req.read())
    print("[2] /api/registry:", reg["benchmark_title"][:50], "... | Author:", reg["investigator"]["name"])
    assert "Bhavy Garg" in reg["investigator"]["name"]
    assert len(reg["validation_derived_thresholds"]) == 10

    # 3. Scans
    req = urllib.request.urlopen('http://127.0.0.1:8000/api/mri/scans?page=1&limit=5')
    scans = json.loads(req.read())
    print("[3] /api/mri/scans: Total =", scans["total"], "| Page =", scans["page"])
    assert scans["total"] == 848

    # 4. Launch Experiment
    exp_payload = {
        'name': 'API Smoke Test Run',
        'dataset': 'ADNI',
        'model_name': 'MONAI Swin UNETR SSL (Linear Probe)',
        'threshold': 0.5088,
        'seed': 42
    }
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/experiments/launch',
        data=json.dumps(exp_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    exp_res = json.loads(urllib.request.urlopen(req).read())
    print("[4] /api/experiments/launch:", exp_res["status"], "| Run ID:", exp_res["run"]["run_id"], "| ROC-AUC:", exp_res["run"]["metrics"]["roc_auc"])
    assert exp_res["status"] == "success"
    assert exp_res["run"]["metrics"]["sample_size"] == 500

    # 5. Live Inference (GET /api/inference/predict?scan_id=...&mode=live)
    req = urllib.request.urlopen('http://127.0.0.1:8000/api/inference/predict?scan_id=002_S_0295_I238627&mode=live')
    inf_res = json.loads(req.read())
    print("[5] /api/inference/predict (Live): mode =", inf_res.get("inference_mode"), "| TrustGate available =", ("trustgate" in inf_res))
    assert inf_res.get("inference_mode") == "LIVE_CHECKPOINT_INFERENCE"

    # 6. Cached Inference (GET /api/inference/predict?scan_id=...&mode=cached)
    req = urllib.request.urlopen('http://127.0.0.1:8000/api/inference/predict?scan_id=002_S_0295_I238627&mode=cached')
    inf_res_cached = json.loads(req.read())
    preds = inf_res_cached.get("predictions", {})
    print("[6] /api/inference/predict (Cached): mode =", inf_res_cached.get("inference_mode"), "| predictions count =", len(preds))
    print("    Sample prediction:", list(preds.items())[0] if preds else "None")
    assert inf_res_cached.get("inference_mode") == "CACHED_VERIFIED_PREDICTION"
    assert len(preds) == 10

    # 7. Explainability / Grad-CAM (GET /api/inference/explain?scan_id=...&plane=axial&slice_idx=64)
    cam_res = urllib.request.urlopen('http://127.0.0.1:8000/api/inference/explain?scan_id=002_S_0295_I238627&plane=axial&slice_idx=64')
    cam_bytes = cam_res.read()
    print("[7] /api/inference/explain: HTTP", cam_res.status, "| bytes =", len(cam_bytes), "| Source:", cam_res.headers.get("X-Attribution-Source"))
    assert cam_res.status == 200
    assert len(cam_bytes) > 1000

    # 8. Downloads
    paper_res = urllib.request.urlopen('http://127.0.0.1:8000/downloads/SHIFT_MRI_Complete_Research_Paper.pdf')
    paper_bytes = paper_res.read()
    print("[8] /downloads/SHIFT_MRI_Complete_Research_Paper.pdf: HTTP", paper_res.status, "| bytes =", len(paper_bytes))
    assert paper_res.status == 200
    assert len(paper_bytes) > 500000

    docx_res = urllib.request.urlopen('http://127.0.0.1:8000/downloads/SHIFT_MRI_Complete_Research_Paper.docx')
    print("[9] /downloads/SHIFT_MRI_Complete_Research_Paper.docx: HTTP", docx_res.status, "| bytes =", len(docx_res.read()))
    assert docx_res.status == 200

    pptx_res = urllib.request.urlopen('http://127.0.0.1:8000/downloads/SHIFT_MRI_Final_Presentation.pptx')
    print("[10] /downloads/SHIFT_MRI_Final_Presentation.pptx: HTTP", pptx_res.status, "| bytes =", len(pptx_res.read()))
    assert pptx_res.status == 200

    zip_res = urllib.request.urlopen('http://127.0.0.1:8000/downloads/SHIFT_MRI_Submission_Package.zip')
    print("[11] /downloads/SHIFT_MRI_Submission_Package.zip: HTTP", zip_res.status, "| bytes =", len(zip_res.read()))
    assert zip_res.status == 200

    print("\n>>> ALL 11 VERIFICATION SMOKE TESTS PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    main()
