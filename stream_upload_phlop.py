import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

HF_REPO_ID = "zimmari-ai/phlop"
HF_PRIVATE = True
SPLIT = "train"

TOTAL_SAMPLES = 10_000
SAMPLES_PER_SHARD = 10


TMP_DIR = Path("tmp_hf_upload")
TMP_DIR.mkdir(exist_ok=True, parents=True)


def generate_example(idx: int) -> Dict[str, Any]:
    video_path = TMP_DIR / f"sample_{idx}.mp4"
    # MARIIA TODO: implement actual generation logic

    with open(video_path, "wb") as f:
        f.write(b"DUMMY")  

    return {
        "video_path": str(video_path),
        "id": idx,
    }

def ensure_repo_exists(repo_id: str, private: bool = True):
    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type="dataset")
        print(f"✅ Repo {repo_id} already exists.")
    except Exception:
        print(f"ℹ️ Repo {repo_id} not found. Creating...")


def create_shard(start_idx: int, end_idx: int) -> Dataset:
    rows: List[Dict[str, Any]] = []
    print(f"🎬 Generating examples {start_idx}..{end_idx-1}")

    for i in range(start_idx, end_idx):
        example = generate_example(i)
        rows.append(example)

    ds = Dataset.from_list(rows)
    return ds


def upload_shard_to_hub(
    ds: Dataset,
    repo_id: str,
    split: str,
    shard_id: int,
    num_shards: int,
):
    """
    Merge this shard with existing remote split and push.
    To avoid memory explosion, we:
      1. Download existing split (streaming)
      2. Concatenate with new shard
      3. Push back
    For large datasets, you may want a different strategy (e.g. store each shard
    in its own split, like `train-000`, etc.).
    """
    print(f"⬆️  Uploading shard {shard_id+1}/{num_shards} to {repo_id} [{split}]")

    # Try loading existing split if any
    try:
        # Streaming=True keeps memory down, but we need in-memory for concatenation.
        # For huge datasets, consider a different layout (multiple splits).
        remote = load_dataset(repo_id, split=split)
        print(f"🔗 Found existing remote split with {len(remote)} rows.")
        merged = Dataset.from_dict(remote.to_dict())  # materialize
        merged = merged.concatenate(ds)
    except Exception:
        print("ℹ️ No existing remote split found, creating new one.")
        merged = ds

    merged.push_to_hub(repo_id, split=split)
    print(f"✅ Shard {shard_id+1} uploaded. Total rows on remote: {len(merged)}")


def clean_temp_files():
    """
    Remove temporary directory to free disk.
    """
    if TMP_DIR.exists():
        print(f"🧹 Cleaning temp dir {TMP_DIR}")
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(exist_ok=True, parents=True)


# -----------------------------
# MAIN LOOP
# -----------------------------

def main():
    ensure_repo_exists(HF_REPO_ID, private=HF_PRIVATE)

    # num_shards = (TOTAL_SAMPLES + SAMPLES_PER_SHARD - 1) // SAMPLES_PER_SHARD
    # print(f"📦 Will generate {TOTAL_SAMPLES} samples in {num_shards} shard(s).")

    # for shard_id in range(num_shards):
    #     start_idx = shard_id * SAMPLES_PER_SHARD
    #     end_idx = min(TOTAL_SAMPLES, (shard_id + 1) * SAMPLES_PER_SHARD)
    #     if start_idx >= end_idx:
    #         break

    #     # Make sure temp dir is clean before each shard
    #     clean_temp_files()

    #     # 1. Generate shard locally
    #     ds = create_shard(start_idx, end_idx)

    #     # 2. Upload to Hub
    #     upload_shard_to_hub(
    #         ds,
    #         repo_id=HF_REPO_ID,
    #         split=SPLIT,
    #         shard_id=shard_id,
    #         num_shards=num_shards,
    #     )

    #     # 3. Free local space for this shard
    #     clean_temp_files()

    # print("🎉 All shards done!")


if __name__ == "__main__":
    main()