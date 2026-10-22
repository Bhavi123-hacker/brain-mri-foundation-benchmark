import urllib.request
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def test_full_platform():
    print("=== SHIFT-MRI FULL PLATFORM INTEGRATION AUDIT ===")

    # 1. Multiple pages of scan registry
    for page in [1, 2, 5, 10]:
        req = urllib.request.urlopen(f"http://127.0.0.1:8000/api/mri/scans?page={page}&limit=20")
        data = json.loads(req.read())
        print(f"Scan Registry Page {page}: Total={data['total']}, Page={data['page']}, Returned={len(data['scans'])}")
        assert data['page'] == page
        assert len(data['scans']) == 20

    # 2. Cohort, scanner, and diagnosis filters
    # Cohort: OASIS-1
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/mri/scans?cohort=OASIS-1&limit=5")
    d_o1 = json.loads(req.read())
    print(f"Filter Cohort=OASIS-1: Total={d_o1['total']}")
    assert d_o1['total'] == 198
    for s in d_o1['scans']:
        assert s['cohort'] == 'OASIS-1'

    # Cohort: ADNI
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/mri/scans?cohort=ADNI&limit=5")
    d_adni = json.loads(req.read())
    print(f"Filter Cohort=ADNI: Total={d_adni['total']}")
    assert d_adni['total'] == 500
    for s in d_adni['scans']:
        assert s['cohort'] == 'ADNI'

    # Scanner: Philips
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/mri/scans?scanner=Philips&limit=5")
    d_phil = json.loads(req.read())
    print(f"Filter Scanner=Philips: Total={d_phil['total']}")
    assert d_phil['total'] == 130
    for s in d_phil['scans']:
        assert 'philips' in s['manufacturer'].lower()

    # Scanner: Siemens
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/mri/scans?cohort=ADNI&scanner=Siemens&limit=5")
    d_siem = json.loads(req.read())
    print(f"Filter ADNI Scanner=Siemens: Total={d_siem['total']}")
    assert d_siem['total'] == 370

    # Diagnosis: AD
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/mri/scans?cohort=ADNI&diagnosis=AD&limit=5")
    d_ad = json.loads(req.read())
    print(f"Filter ADNI Diagnosis=AD: Total={d_ad['total']}")
    assert d_ad['total'] == 197

    # 3. Cached predictions matching source artifacts
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/inference/predict?scan_id=002_S_0295_I238627&mode=cached")
    pred_data = json.loads(req.read())
    print(f"Cached Scan ID: {pred_data['scan_id']} | Cohort: {pred_data['cohort']}")
    print(f"Predictions Count: {len(pred_data['predictions'])}")
    assert len(pred_data['predictions']) == 10
    assert pred_data['inference_mode'] == "CACHED_VERIFIED_PREDICTION"
    # Verify exact prediction values match CSV
    assert round(pred_data['predictions']['MONAI Swin UNETR SSL (Linear Probe)'], 4) == 0.5293

    # 4. Correct model-specific thresholds in registry
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/registry")
    reg = json.loads(req.read())
    expected_thresholds = {
        "MONAI Swin UNETR SSL (Linear Probe)": 0.5088,
        "3D ResNet-10 (Pipeline B)": 0.6860,
        "3D DenseNet-121 (Pipeline B)": 0.4651,
        "2.5D Multi-View ResNet-18 (Pipeline B)": 0.9741,
        "Dual-Branch Hybrid (ResNet-10 + Radiomics)": 0.4763,
        "FOMO25 AMAES ResEnc-B (Linear Probe)": 0.5457,
        "3D ResNet-18 (Scratch Baseline)": 0.5725,
        "3D ResNet-34 (Scratch Baseline)": 0.5512,
        "2D Transfer ResNet-18 (Slice-Aggregated)": 0.6120,
        "Classical Radiomics (Linear SVM)": 0.5693
    }
    for m, th in expected_thresholds.items():
        assert reg["validation_derived_thresholds"][m] == th, f"Threshold mismatch for {m}"
    print("Model-specific validation thresholds verified for all 10 models.")

    # 5. Authentic attribution vs unavailable explanations
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/inference/explain?scan_id=002_S_0295_I238627&plane=axial&slice_idx=64")
    assert req.status == 200
    source_hdr = req.headers.get("X-Attribution-Source")
    print(f"Authentic Grad-CAM Attribution Source Header: {source_hdr}")
    assert source_hdr is not None

    # Unavailable explanation check
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/api/inference/explain?scan_id=NON_EXISTENT_SCAN_ID")
        assert False, "Should have returned 404 for missing scan!"
    except urllib.error.HTTPError as e:
        print(f"Missing scan correctly rejected with HTTP {e.code}: {e.reason}")
        assert e.code == 404

    # 6. Run Experiment Lab evaluation & verify independent calculation
    exp_payload = {
        'name': 'Verification Lab Run',
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
    exp_resp = json.loads(urllib.request.urlopen(req).read())
    run_info = exp_resp['run']
    run_id = run_info['run_id']
    metrics = run_info['metrics']
    print(f"\nExperiment Lab Launched Run: {run_id}")
    print(f"ROC-AUC: {metrics['roc_auc']} | 95% CI: {metrics['ci_95']} | Brier: {metrics['brier_score']}")

    # Independent direct calculation verification
    df_adni = pd.read_csv("D:/ml project/results/report/adni_external_predictions.csv")
    y_true = df_adni['binary_label'].values
    y_pred = df_adni['swin_unetr_ssl_prob'].values
    calc_auc = round(float(roc_auc_score(y_true, y_pred)), 4)
    calc_brier = round(float(brier_score_loss(y_true, y_pred)), 4)
    print(f"Direct Independent Calculation: ROC-AUC={calc_auc}, Brier={calc_brier}")
    assert metrics['roc_auc'] == calc_auc
    assert metrics['brier_score'] == calc_brier

    # Verify SQLite persistence
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/experiments/history")
    history = json.loads(req.read())
    run_ids = [r['run_id'] for r in history['runs']]
    print(f"SQLite History Run Count: {history['total_runs']} | Contains {run_id}: {run_id in run_ids}")
    assert run_id in run_ids

    # 7. Experiment comparison and cross-cohort label discordance warning
    # Note: query param is `ids=id1,id2`
    first_hist_id = history['runs'][-1]['run_id'] if history['runs'] else run_id
    req = urllib.request.urlopen(f"http://127.0.0.1:8000/api/experiments/compare?ids={run_id},{first_hist_id}")
    comp_res = json.loads(req.read())
    print(f"Comparison Status: {comp_res['status']}")
    if comp_res.get('comparison', {}).get('cross_cohort_warning'):
        print(f"Cross-Cohort Warning: {comp_res['comparison']['cross_cohort_warning']}")
    assert comp_res['status'] == "success"

    print("\n>>> ALL PLATFORM INTEGRATION AUDITS PASSED CLEANLY! <<<")

if __name__ == "__main__":
    test_full_platform()
