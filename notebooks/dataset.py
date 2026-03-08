"""
PHLOP dataset with lazy shard extraction — reusable for any model (vision, VLM, etc.).

This module is model-agnostic: it only handles loading the HF dataset and
resolving paths by downloading/extracting shard zips on demand. Model-specific
logic (video decoding, prompts, taxonomy, etc.) belongs in wrappers such as
eval_dataset.py (for SmolVLM/vision evals) or other model-specific modules.

Repo layout:
  repo/
   ├── train_shard_0000.parquet
   ├── val_shard_0000.parquet
   ├── data/
   │    ├── train/train_shard_0000.zip
   │    ├── val/val_shard_0000.zip

Parquet rows include `shard_file` and paths like `videos`, `qa`, `metadata`.
__getitem__(idx) returns the row with those paths resolved to absolute paths
on disk (after extracting the shard zip if needed).

Usage (from notebooks/ or with this path on PYTHONPATH):
    from datasets import load_dataset
    from dataset import PHLOPDataset

    ds = load_dataset("your-username/phlop", split="train")
    loader = PHLOPDataset(ds, repo_id="your-username/phlop")
    sample = loader[0]  # downloads shard zip on first access, then resolves paths
    video_path = sample["videos"]["static"]  # or sample["videos"]["moving"]
    # Use sample with any model: load video/qa/metadata in your own code.
"""

from pathlib import Path
from typing import Any, Optional, Union

import zipfile

from huggingface_hub import hf_hub_download


class PHLOPDataset:
    """
    PyTorch-style dataset over a PHLOP Hugging Face dataset.
    Loads metadata from parquet only; extracts shard zips on demand.
    """

    def __init__(
        self,
        hf_dataset,
        repo_id: str,
        extract_root: Optional[Union[str, Path]] = None,
        repo_type: str = "dataset",
        token: Optional[Union[str, bool]] = None,
    ):
        """
        Args:
            hf_dataset: Result of load_dataset("username/phlop", split="train").
            repo_id: Hugging Face repo id (e.g. "username/phlop").
            extract_root: Directory for extracted shards. Defaults to ~/.cache/phlop_shards.
            repo_type: "dataset" or "model".
            token: Hugging Face token for private repos (True = use cached login).
        """
        self.ds = hf_dataset
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token
        self.extract_root = Path(extract_root or _default_extract_root())
        self.extract_root.mkdir(parents=True, exist_ok=True)
        self._loaded_shards: dict[str, Path] = {}

    def _ensure_shard_extracted(self, shard_file: str) -> Path:
        """Download zip if needed and extract; return path to extracted root."""
        if shard_file in self._loaded_shards:
            return self._loaded_shards[shard_file]

        # e.g. "data/val/val_shard_0000.zip" -> "val_shard_0000"
        shard_name = Path(shard_file).stem
        extract_dir = self.extract_root / shard_name

        if extract_dir.exists() and any(extract_dir.iterdir()):
            self._loaded_shards[shard_file] = extract_dir
            return extract_dir

        zip_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=shard_file,
            repo_type=self.repo_type,
            local_dir=None,
            token=self.token,
        )

        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        self._loaded_shards[shard_file] = extract_dir
        return extract_dir

    def _resolve_path(self, root: Path, path_str: str) -> Path:
        """Path in parquet is relative to zip root (e.g. data/val/shard_0000/...)."""
        return root / path_str

    def __len__(self) -> int:
        return len(self.ds)

    def _resolve_dict_paths(self, root: Path, d: Optional[dict]) -> Optional[dict]:
        if not d or not isinstance(d, dict):
            return d
        return {k: str(self._resolve_path(root, v)) if v else None for k, v in d.items()}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.ds[idx])
        shard_file = sample.get("shard_file")
        if not shard_file:
            return sample
        root = self._ensure_shard_extracted(shard_file)

        # Resolve paths that live inside the shard zip
        for key in ("videos", "metadata", "qa", "segmentated_file"):
            if key in sample:
                sample[key] = self._resolve_dict_paths(root, sample[key])

        return sample


def _default_extract_root() -> Path:
    return Path.home() / ".cache" / "phlop_shards"
