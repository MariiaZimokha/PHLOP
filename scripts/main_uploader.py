import json
import os
import shutil
import time
import pandas as pd
import numpy as np
import random
import tempfile
from pathlib import Path
from tqdm import tqdm

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from phlop.simulator import Simulation
from phlop.split_config import SPLIT_CONFIG
from phlop.video_annotation_visualizer import VideoAnnotationVisualizer
from phlop.world.object import Object
from phlop.annotator import Annotator
from phlop.question_answer import QuestionAnswers
from phlop.advanced_physics_questions import AdvancedPhysicsQuestions

from upload.config import (
    HF_REPO,
    HF_TOKEN,
    TRAIN_COUNT,
    VAL_COUNT,
    TEST_COUNT,
    SHARD_SIZE,
    OUTPUT_DIR,
    WIDTH,
    HEIGHT,
    FPS,
)
import tempfile
import shutil

HF_API = HfApi(token=HF_TOKEN)

def atomic_write_json(data, path):
    """Write JSON atomically to prevent empty/corrupted files."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        temp_name = tmp.name

    shutil.move(temp_name, path)

def upload_with_retry(fn, retries=8, base_sleep=2):
    for i in range(retries):
        try:
            return fn()
        except (HfHubHTTPError, RuntimeError) as e:
            # check for 429 (Too Many Requests) or 500+ server errors
            is_rate_limit = isinstance(e, HfHubHTTPError) and e.response is not None and e.response.status_code == 429
            is_server_error = isinstance(e, HfHubHTTPError) and e.response is not None and e.response.status_code >= 500
            
            if is_rate_limit or is_server_error:
                sleep_time = base_sleep * (2 ** i)  # Exponential backoff
                print(f"⚠️  Upload snag (Attempt {i+1}/{retries}). Sleeping {sleep_time}s... Error: {e}")
                time.sleep(sleep_time)
            else:
                raise e
    raise RuntimeError("❌ Upload failed after maximum retries")


def upload_shard_folder(shard_id: int, shard_dir: Path, repo_id: str, split: str):
    print(f"\n⬆️  Preparing Large Folder Upload for {split} Shard {shard_id}...")

    staging_root = Path("temp_staging")
    repo_structure_path = staging_root / "data" / split / f"shard_{shard_id:04d}"
    
    if staging_root.exists():
        shutil.rmtree(staging_root)
    repo_structure_path.mkdir(parents=True)

    for item in shard_dir.iterdir():
        shutil.move(str(item), str(repo_structure_path / item.name))

    try:
        print(f"🚀 Committing large folder to {repo_id}...")
        upload_with_retry(lambda: HF_API.upload_large_folder(
            folder_path=str(staging_root),
            repo_id=repo_id,
            repo_type="dataset",
        ))
        
        print(f"  ✅ Shard {shard_id} upload complete!")
        return True
    except Exception as e:
        print(f"  ❌ Failed to upload shard {shard_id}: {e}")
        return False
    finally:
        # Cleanup staging
        if staging_root.exists():
            shutil.rmtree(staging_root)


def calculate_adaptive_distance(elevation_angle, angle_range, distance_range):
    min_elev, max_elev = angle_range
    min_dist, max_dist = distance_range

    normalized = (elevation_angle - min_elev) / (max_elev - min_elev)
    normalized = np.clip(normalized, 0.0, 1.0)

    adaptive_dist = min_dist + normalized * (max_dist - min_dist)

    return float(adaptive_dist)


def sample_camera_for_split(split: str, mode: str = "static"):
    """
    Returns split-specific camera config with ADAPTIVE distance based on elevation.
    All camera settings are read from SPLIT_CONFIG.

    The distance automatically adjusts based on the elevation angle:
    - Steep angles (top-down): Camera moves closer for better framing
    - Shallow angles (near horizon): Camera moves back for wider view
    """
    camera_cfg = SPLIT_CONFIG[split]["camera"]
    
    # Get ranges from split config
    az_min, az_max = camera_cfg["azimuth_range"]
    elev_min, elev_max = camera_cfg["elevation_range"]
    distance_range = camera_cfg["distance_range"]
    lookat_z_min, lookat_z_max = camera_cfg["lookat_z_range"]
    limits = camera_cfg["limits"]
    
    # Sample random values within ranges
    az = random.uniform(az_min, az_max)
    elev = random.uniform(elev_min, elev_max)
    lookat_z = random.uniform(lookat_z_min, lookat_z_max)
    
    # Calculate adaptive distance based on elevation
    angle_range = (elev_min, elev_max)
    dist = calculate_adaptive_distance(elev, angle_range, distance_range)

    lookat = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), lookat_z]

    cam_mode_int = 1 if mode == "moving" else 0

    return {
        "mode": cam_mode_int,
        "init": {
            "lookat": lookat,
            "azimuth": az,
            "elevation": elev,
            "distance": dist,
        },
        "limits": limits,
        "follow": "none" if mode == "static" else "fastest",
    }


def build_object_specs(split_cfg, num_objects):
    shapes = split_cfg["shapes"]
    materials = split_cfg["materials"]
    comp = split_cfg["material_components"]

    specs = []
    for _ in range(num_objects):
        mat = random.choice(materials)

        specs.append(
            {
                "shape": random.choice(shapes),
                "material": mat,
                "density_idx": comp[mat]["density_idx"],
                "friction_idx": comp[mat]["friction_idx"],
                "elasticity_idx": comp[mat]["elasticity_idx"],
            }
        )
    return specs


def run_single_simulation(
    sim: Simulation,
    num_objects: int,
    duration: float,
    framerate: int,
    scene_dir: Path,
    camera_cfg: dict,
    objects=None,
    floor=None,
    lights=None,
    object_specs=None,
    split: str = "train",
):
    if scene_dir.exists():
        for p in scene_dir.glob("*"):
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)
    scene_dir.mkdir(parents=True, exist_ok=True)

    sim_out = sim.run_simulation(
        num_objects=num_objects,
        duration=duration,
        framerate=framerate,
        path=str(scene_dir) + "/",
        camera=camera_cfg,
        objects=objects,
        floor=floor,
        lights=lights,
        object_specs=object_specs,
    )

    video_file = sim_out["video_file"]
    file_path = sim_out["file_path"]

    annotator = VideoAnnotationVisualizer()
    annotated_video = scene_dir / "annotated_video.mp4"
    annotator.annotate(
        file_path=file_path,
        video_path=video_file,
        annotated_video_path=str(annotated_video),
    )


    qa_json_path = scene_dir / "qa.json"
    qa_pairs = QuestionAnswers(file_path).get_questions_answers()
    advanced_qa_pairs = AdvancedPhysicsQuestions(file_path, split=split).generate_all_advanced_questions()
    qa_pairs = qa_pairs + advanced_qa_pairs
    atomic_write_json(qa_pairs, qa_json_path)

    return {
        "video": video_file,
        "annotated": str(annotated_video),
        "metadata": file_path,
        "qa": str(qa_json_path),
        # "qa_pairs": qa_pairs,
    }


def generate_training_shard(shard_id, start_idx, count):
    obj = Object()
    annotator = Annotator()
    sim = Simulation(obj, annotator=annotator, width=WIDTH, height=HEIGHT)

    moving_count = count // 2
    static_count = count - moving_count
    cam_modes = ["moving"] * moving_count + ["static"] * static_count
    random.shuffle(cam_modes)

    rows = []
    for i in range(count):
        global_idx = start_idx + i
        cam_mode = cam_modes[i]

        scene_path = OUTPUT_DIR / f"train_{global_idx}"
        obj_count_min, obj_count_max = SPLIT_CONFIG["train"]["object_count"]
        num_objects = random.randint(obj_count_min, obj_count_max)
        object_specs = build_object_specs(SPLIT_CONFIG["train"], num_objects)
        camera_cfg = sample_camera_for_split(split="train", mode=cam_mode)

        out = run_single_simulation(
            sim=sim,
            num_objects=num_objects,
            duration=6,
            framerate=FPS,
            scene_dir=scene_path,
            camera_cfg=camera_cfg,
            object_specs=object_specs,
            split="train",
        )
        # print("out ", out)
        rows.append(
            {
                "id": global_idx,
                "split": "train",
                "camera_mode": cam_mode,
                "num_objects": num_objects,
                "videos": {
                    cam_mode: f"data/train/shard_{shard_id:04d}/train_{global_idx}/simulation_objects.mp4"
                },
                "metadata": {
                    cam_mode: f"data/train/shard_{shard_id:04d}/train_{global_idx}/meta.json"
                },
                "qa": {
                    cam_mode: f"data/train/shard_{shard_id:04d}/train_{global_idx}/qa.json"
                },
                "segmentated_file": {
                    cam_mode: f"data/train/shard_{shard_id:04d}/train_{global_idx}/simulation_objects_segmented.mp4"
                },
                # "video_file": f"data/train/shard_{shard_id:04d}/train_{global_idx}/simulation_objects.mp4",
                # "segmentated_file": f"data/train/shard_{shard_id:04d}/train_{global_idx}/simulation_objects_segmented.mp4",
                # "metadata_file": f"data/train/shard_{shard_id:04d}/train_{global_idx}/meta.json",
                # "qa_file": f"data/train/shard_{shard_id:04d}/train_{global_idx}/qa.json",
            }
        )
        print(' out["metadata"] ', out["metadata"])

    df = pd.DataFrame(rows)
    parquet_filename = f"train_shard_{shard_id:04d}.parquet"
    parquet_path = OUTPUT_DIR / parquet_filename
    df.to_parquet(parquet_path, index=False)

    print(f"✅ Saved parquet: {parquet_filename}")

    success = upload_shard_folder(
        shard_id=shard_id,
        shard_dir=OUTPUT_DIR,
        repo_id=HF_REPO,
        split="train"
    )

    # Safe cleanup
    if success:
        print("Cleaning up local data...")
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    else:
        print("⚠️ Upload failed. Keeping local data for inspection.")

def generate_validation_shard(shard_id, start_idx, count, split):
    obj = Object()
    annotator = Annotator()
    sim = Simulation(obj, annotator=annotator, width=WIDTH, height=HEIGHT)

    rows = []
    for i in range(count):
        global_idx = start_idx + i
        base_scene_path = OUTPUT_DIR / f"{split}_{global_idx}"
        base_scene_path.mkdir(parents=True, exist_ok=True)

        obj_count_min, obj_count_max = SPLIT_CONFIG[split]["object_count"]
        num_objects = random.randint(obj_count_min, obj_count_max)
        # camera_cfg = sample_camera(split, "static")
        camera_cfg = sample_camera_for_split(split=split, mode="static")

        print("camera_cfg ", camera_cfg)

        out_static = run_single_simulation(
            sim=sim,
            num_objects=num_objects,
            duration=6,
            framerate=FPS,
            scene_dir=base_scene_path / "static",
            camera_cfg=camera_cfg,
            split=split,
        )

        meta = json.load(open(base_scene_path / "static" / "meta.json"))
        objects = meta["objects"]

        floor = meta["world"]["floor"]
        lights = meta["world"]["lights"]

        camera_cfg["mode"] = 1  # moving

        out_moving = run_single_simulation(
            sim=sim,
            num_objects=num_objects,
            duration=6,
            framerate=FPS,
            scene_dir=base_scene_path / "moving",
            camera_cfg=camera_cfg,
            objects=objects,
            floor=floor,
            lights=lights,
            split=split,
        )

        rows.append(
            {
                "id": global_idx,
                "split": split,
                "num_objects": num_objects,
                # Static
                "videos": {
                    "static": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/static/simulation_objects.mp4",
                    "moving": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/moving/simulation_objects.mp4"
                },
                "metadata": {
                    "static": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/static/meta.json",
                    "moving": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/moving/meta.json"
                },
                "qa": {
                    "static": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/static/qa.json",
                    "moving": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/moving/qa.json"
                },
                "segmentated_file": {
                    "static": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/static/simulation_objects_segmented.mp4",
                    "moving": f"data/{split}/shard_{shard_id:04d}/{split}_{global_idx}/moving/simulation_objects_segmented.mp4"
                },
            }
        )

    df = pd.DataFrame(rows)
    parquet_filename = f"{split}_shard_{shard_id:04d}.parquet"
    parquet_path = OUTPUT_DIR / parquet_filename
    df.to_parquet(parquet_path, index=False)

    print(f"✅ Generated {split} shard {shard_id} locally.")

    # Single Upload Call (Folder)
    success = upload_shard_folder(
        shard_id=shard_id,
        shard_dir=OUTPUT_DIR,
        repo_id=HF_REPO,
        split=split,
    )

    if success:
        print("Cleaning up local data...")
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    else:
        print("⚠️ Upload failed. Keeping local data.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--shard-id", type=int, default=0)
    # parser.add_argument("--start-idx", type=int, default=0)
    # parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    print("Generating", args)

    # Ensure the HuggingFace dataset repo exists
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True, private=True)

    current_idx = 0
    shard_id = 0
    split = args.split

    # TRAIN_COUNT, VAL_COUNT, TEST_COUNT

    if split == "train":
        total_examples = TRAIN_COUNT
        total_shards = (total_examples + SHARD_SIZE - 1) // SHARD_SIZE
        for shard_id in tqdm(range(total_shards), desc="train shards"):
            print("shard_id", shard_id)
            remaining = total_examples - current_idx
            count = min(SHARD_SIZE, remaining)
            if count <= 0:
                break
            generate_training_shard(shard_id, current_idx, count)
            current_idx += count

    if split in ["val", "test"]:
        total_count = VAL_COUNT if split == "val" else TEST_COUNT
        while current_idx < total_count:
            remaining = total_count - current_idx
            count = min(SHARD_SIZE, remaining)
            if count <= 0:
                break
            generate_validation_shard(shard_id, current_idx, count, split=split)
            current_idx += count
            shard_id += 1


if __name__ == "__main__":
    main()



# to do
# start uploading some data
# in colab build dataloader and show some samples
#  zero shot inference on val set