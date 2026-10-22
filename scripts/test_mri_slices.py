import urllib.request
from PIL import Image
import io

def test_mri_slices():
    print("=== TESTING MRI SLICE EXTRACTION IN ALL 3 PLANES ===")
    base_url = "http://127.0.0.1:8000/api/mri/slice?scan_id=002_S_0295_I238627"
    
    planes = ["axial", "coronal", "sagittal"]
    for plane in planes:
        url = f"{base_url}&plane={plane}&slice_idx=64"
        req = urllib.request.urlopen(url)
        assert req.status == 200
        img_bytes = req.read()
        img = Image.open(io.BytesIO(img_bytes))
        print(f"Plane: {plane:8s} | HTTP: {req.status} | Size: {img.size} | Format: {img.format} | Bytes: {len(img_bytes)}")
        assert img.size == (128, 128)
        assert img.format == "JPEG"

    print("ALL THREE ORTHOGONAL SLICE PLANES VERIFIED!")

if __name__ == "__main__":
    test_mri_slices()
