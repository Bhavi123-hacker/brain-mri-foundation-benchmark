import os
import zipfile
from pathlib import Path

sub_dir = Path("D:/ml project/FINAL_SUBMISSION_23_OCT_2026")
zip_path = sub_dir / "SHIFT_MRI_V4_Submission_Package.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(sub_dir):
        for f in files:
            p = Path(root) / f
            if p == zip_path or p.suffix in [".pt", ".ckpt", ".safetensors", ".npy", ".nii", ".gz"]:
                continue
            arcname = p.relative_to(sub_dir)
            zipf.write(p, arcname)

print("Updated ZIP size MB:", round(zip_path.stat().st_size / (1024 * 1024), 2))
