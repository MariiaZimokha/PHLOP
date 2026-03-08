import os
import torch
from decord import VideoReader, cpu
import json

def format_physics_signals(signals: dict) -> str:
    """Format physics signals dict as a string with one '- key: value' per line. Returns 'None' if empty."""
    if not signals:
        return "None"
    lines = []
    for k, v in signals.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def get_physics_summary(metadata: dict) -> str:
    """Build a short text summary of object mass/friction from scene metadata."""
    summary = []
    for obj in metadata.get("objects", []):
        summary.append(
            f"Object {obj['id']} ({obj['shape']}, {obj['material']}): "
            f"Mass={obj.get('mass', 'N/A')}, "
            f"Friction={obj.get('friction', 'N/A')}"
        )
    return "\n".join(summary)


def load_video(video_path: str, root_dir: str, num_frames: int):
    """Load a video from disk and sample num_frames uniformly. Returns numpy array (t, h, w, c)."""
    abs_path = os.path.join(root_dir, video_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Video not found at: {abs_path}")
    vr = VideoReader(abs_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
    return vr.get_batch(indices).asnumpy()

def load_json(path: str) -> dict:
    """Load JSON file and return as dictionary."""
    with open(path, "r") as f:
        return json.load(f)
