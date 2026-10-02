"""
Preprocessed 3D Volume Caching and Disk-Indexed Storage Manager.
"""

import os
import hashlib
import json
import torch
from typing import Optional, Dict, Any, Tuple


class PreprocessedCacheManager:
    """
    Manages preprocessed 3D tensor caching on disk to eliminate repeated
    resampling and spatial transforms during iterative training epochs.
    """
    def __init__(self, cache_dir: str = "data/cache/preprocessed"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, "cache_index.json")
        self.index: Dict[str, str] = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def compute_cache_key(self, file_path: str, transform_id: str = "default_128") -> str:
        """Generate a deterministic hash key for a scan and preprocessing configuration."""
        raw_str = f"{os.path.abspath(file_path)}_{transform_id}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

    def has(self, file_path: str, transform_id: str = "default_128") -> bool:
        """Check if preprocessed tensor is available in cache."""
        key = self.compute_cache_key(file_path, transform_id)
        if key in self.index:
            cached_path = self.index[key]
            return os.path.exists(cached_path)
        return False

    def get(self, file_path: str, transform_id: str = "default_128") -> Optional[torch.Tensor]:
        """Retrieve preprocessed tensor from cache."""
        key = self.compute_cache_key(file_path, transform_id)
        if key in self.index:
            cached_path = self.index[key]
            if os.path.exists(cached_path):
                try:
                    tensor_dict = torch.load(cached_path, map_location="cpu", weights_only=True)
                except Exception:
                    tensor_dict = torch.load(cached_path, map_location="cpu", weights_only=False)
                return tensor_dict["tensor"]
        return None

    def put(self, file_path: str, tensor: torch.Tensor, transform_id: str = "default_128", dtype: torch.dtype = torch.float16) -> str:
        """Save preprocessed tensor to disk cache in compact float16 format."""
        key = self.compute_cache_key(file_path, transform_id)
        save_filename = f"{key}.pt"
        save_path = os.path.join(self.cache_dir, save_filename)

        clean_tensor = torch.as_tensor(tensor.detach().cpu(), dtype=dtype)
        torch.save({"tensor": clean_tensor, "source_path": file_path, "shape": list(clean_tensor.shape)}, save_path)

        self.index[key] = os.path.abspath(save_path)
        self._save_index()
        return save_path

    def get_cache_size_mb(self) -> float:
        """Calculate total disk space occupied by cache."""
        total_bytes = 0
        for root, _, files in os.walk(self.cache_dir):
            for f in files:
                fp = os.path.join(root, f)
                total_bytes += os.path.getsize(fp)
        return round(total_bytes / (1024.0 * 1024.0), 2)
