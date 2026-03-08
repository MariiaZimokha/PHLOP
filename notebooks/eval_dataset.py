"""
PHLOP dataset for model evals: load from Hugging Face (zip + parquet layout)
or from a local root. Same interface as the original PHLOPVideoDataset.
"""

import os
import json
from typing import Any, Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu

# Prefer local reusable dataset (notebooks/dataset.py); fallback to phlop package
try:
    from dataset import PHLOPDataset
except ImportError:
    from phlop.dataset import PHLOPDataset


# Taxonomy labels for one-hot targets (must match training annotation)
TAXONOMY_LABELS = [
    "Stationary",
    "Accelerating",
    "Decelerating",
    "Constant Velocity",
    "Pure Rotation",
    "Rolling Motion",
    "Rolling Motion with Slipping",
    "Moving to Stopping",
    "Stationary to Moving",
    "Elastic Collision",
    "Partially Inelastic Collision",
    "Highly Inelastic Collision",
    "Inelastic Collision",
    "Friction Stop",
    "Sliding with Friction",
]
TAXONOMY_LABELS = [label.lower() for label in TAXONOMY_LABELS]
labels_to_idx = {label: idx for idx, label in enumerate(TAXONOMY_LABELS)}
NUM_TAX = len(TAXONOMY_LABELS)


def format_physics_signals(signals: dict) -> str:
    if not signals:
        return "None"
    lines = [f"- {k}: {v}" for k, v in signals.items()]
    return "\n".join(lines)


def load_phlop_from_hf(
    repo_id: str,
    split: str = "train",
    num_frames: int = 8,
    camera_mode: str = "static",
    extract_root: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[Union[str, bool]] = None,
) -> "PHLOPVideoDataset":
    """
    Load PHLOP from Hugging Face and return a PHLOPVideoDataset ready for evals.
    For private repos, pass token=True (cached login) or token="hf_...".

    Usage:
        from phlop.eval_dataset import load_phlop_from_hf

        ds = load_phlop_from_hf("your-username/phlop", split="val", camera_mode="static", token=True)
        sample = ds[0]
        video, qa, metadata = sample["video"], sample["qa"], sample["metadata"]
    """
    import os
    from datasets import load_dataset

    hf_token = token if token is not None else os.environ.get("HF_TOKEN") or True
    hf_split = load_dataset(
        repo_id,
        split=split,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    return PHLOPVideoDataset(
        hf_dataset_split=hf_split,
        num_frames=num_frames,
        split=split,
        root_dir="",
        camera_mode=camera_mode,
        repo_id=repo_id,
        extract_root=extract_root,
        token=hf_token,
    )


class PHLOPVideoDataset(Dataset):
    """
    Dataset for PHLOP videos + QA + metadata. Supports:
    - Hugging Face: pass repo_id so shard zips are downloaded and extracted on demand.
    - Local: pass root_dir pointing to repo root (paths in parquet are relative to it).
    """

    def __init__(
        self,
        hf_dataset_split,
        num_frames: int = 8,
        split: str = "train",
        root_dir: str = "./",
        camera_mode: str = "static",
        repo_id: Optional[str] = None,
        extract_root: Optional[Union[str, os.PathLike]] = None,
        token: Optional[Union[str, bool]] = None,
    ):
        self.num_frames = num_frames
        self.split = split
        self.camera_mode = camera_mode
        self.root_dir = root_dir or ""

        if repo_id:
            self._backend = PHLOPDataset(
                hf_dataset_split,
                repo_id=repo_id,
                extract_root=extract_root,
                token=token,
            )
            self._from_hf = True
        else:
            self._backend = hf_dataset_split
            self._from_hf = False

    def _get_item_raw(self, idx: int) -> dict:
        """Return row with paths resolved (absolute when from HF, relative when local)."""
        item = self._backend[idx] if self._from_hf else dict(self._backend[idx])
        return item

    def _resolve_path(self, path: str) -> str:
        if self._from_hf or not path or os.path.isabs(path):
            return path
        return os.path.join(self.root_dir, path)

    def get_qa_list(self, scene_idx: int) -> list:
        """
        Retrieves the QA list for a specific scene index.
        Same logic as working smol-eval.py: item["qa"][cam] then load JSON.
        Uses _backend[idx] first (triggers shard extraction); if qa is missing or
        path not found, falls back to raw parquet row + manual path resolution.
        """
        # 1) Try resolved row from backend (shard extracted, paths absolute)
        try:
            item = self._backend[scene_idx]
            if not isinstance(item, dict):
                item = dict(item)
        except Exception:
            return []

        cam = item.get("camera_mode") if self.split == "train" else self.camera_mode
        qa_data = item.get("qa")
        if isinstance(qa_data, dict):
            qa_path = qa_data.get(cam)
            if qa_path:
                raw = self._load_json(qa_path)
                if isinstance(raw, list):
                    return raw
                if isinstance(raw, dict):
                    return raw.get("questions", raw.get("qa", []))
            # Try any other key (e.g. "static" when cam is "moving" and only static exists)
            for path in qa_data.values():
                if path:
                    raw = self._load_json(path)
                    if isinstance(raw, list) and raw:
                        return raw
                    if isinstance(raw, dict):
                        lst = raw.get("questions", raw.get("qa", []))
                        if lst:
                            return lst

        # 2) Fallback: raw parquet row (Hub may not expose "qa" on resolved item)
        if hasattr(self._backend, "ds"):
            try:
                raw_row = dict(self._backend.ds[scene_idx])
            except Exception:
                return []
            qa_raw = raw_row.get("qa")
            if not isinstance(qa_raw, dict):
                return []
            path = qa_raw.get(cam) or qa_raw.get("static") or qa_raw.get("moving")
            if not path or not isinstance(path, str):
                return []
            resolve_root = None
            if raw_row.get("shard_file") and hasattr(self._backend, "_ensure_shard_extracted"):
                try:
                    resolve_root = self._backend._ensure_shard_extracted(raw_row["shard_file"])
                except Exception:
                    pass
            path_to_load = str(resolve_root / path) if resolve_root else self._resolve_path(path)
            raw = self._load_json(path_to_load)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                return raw.get("questions", raw.get("qa", []))
        return []


    def __len__(self) -> int:
        return len(self._backend)

    def _load_video(self, video_path: str) -> np.ndarray:
        abs_path = self._resolve_path(video_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Video not found at: {abs_path}")
        vr = VideoReader(abs_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()
        return vr.get_batch(indices).asnumpy()

    def _load_json(self, path: str):
        """Load JSON from path; returns {} if path is None or file missing (avoids NoneType errors)."""
        if not path or not isinstance(path, str) or not path.strip():
            return {}
        abs_path = self._resolve_path(path)
        if not os.path.exists(abs_path):
            return {}
        try:
            with open(abs_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_json_safe(self, path: Optional[str]) -> list:
        """Safely load JSON; returns [] if path is None, file missing, or parse error. Use for QA lists."""
        if not path or not isinstance(path, str) or not path.strip():
            return []
        abs_path = self._resolve_path(path)
        if not os.path.exists(abs_path):
            return []
        try:
            with open(abs_path, "r") as f:
                raw = json.load(f)
        except Exception:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return raw.get("questions", raw.get("qa", []))
        return []

    def _get_physics_summary(self, metadata: dict) -> str:
        summary = []
        for obj in metadata.get("objects", []):
            summary.append(
                f"Object {obj['id']} ({obj['shape']}, {obj['material']}): "
                f"Mass={obj.get('mass', 'N/A')}, "
                f"Friction={obj.get('friction', 'N/A')}"
            )
        return "\n".join(summary)

    def _get_taxonomy_targets(self, metadata: dict) -> torch.Tensor:
        target = torch.zeros(NUM_TAX)
        for frame in metadata.get("frames", []):
            for obj_id, obj_data in frame.get("objects", {}).items():
                if obj_data.get("bbox", [[0, 0], [0, 0]]) != [[0, 0], [0, 0]]:
                    for entry in obj_data.get("taxonomy", []):
                        for label in entry.get("labels", []):
                            label = label.lower()
                            if label in labels_to_idx:
                                target[labels_to_idx[label]] = 1.0
        return target

    def __getitem__(self, idx: int) -> dict:
        item = self._get_item_raw(idx)
        camera_mode = (
            item.get("camera_mode") if self.split == "train" else self.camera_mode
        )

        video_path = (item.get("videos") or {}).get(camera_mode)
        qa_path = (item.get("qa") or {}).get(camera_mode)
        meta_path = (item.get("metadata") or {}).get(camera_mode)

        if not meta_path:
            raise KeyError(f"Missing metadata path for camera_mode={camera_mode} at idx={idx}")
        metadata = self._load_json(meta_path)

        return {
            "video": self._load_video(video_path) if video_path else None,
            "qa": self._load_json(qa_path) if qa_path else None,
            "metadata": metadata,
            "physics_summary": self._get_physics_summary(metadata),
            "taxonomy_target": self._get_taxonomy_targets(metadata),
            "camera_mode": camera_mode,
            "id": item.get("id", idx),
        }
