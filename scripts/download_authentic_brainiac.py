from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import hashlib
import os
import shutil

dest_dir = "D:/ml project/weights"
os.makedirs(dest_dir, exist_ok=True)
dest_path = os.path.join(dest_dir, "brainiac_vit_backbone.safetensors")

print("Downloading authentic BrainIAC backbone.safetensors from eugenehp/brainiac...")
downloaded_path = hf_hub_download(repo_id="eugenehp/brainiac", filename="backbone.safetensors")
shutil.copyfile(downloaded_path, dest_path)

size_mb = os.path.getsize(dest_path) / (1024**2)
h = hashlib.sha256(open(dest_path, "rb").read()).hexdigest()
print(f"Downloaded successfully: {dest_path}")
print(f"Size: {size_mb:.2f} MB, SHA256: {h}")

state_dict = load_file(dest_path)
print(f"Loaded {len(state_dict)} tensors from safetensors.")
total_params = sum(v.numel() for v in state_dict.values())
print(f"Total parameter count in authentic BrainIAC backbone: {total_params:,}")
sample_keys = list(state_dict.keys())[:10]
print("Sample keys:")
for k in sample_keys:
    print(f"  {k}: {state_dict[k].shape}")
