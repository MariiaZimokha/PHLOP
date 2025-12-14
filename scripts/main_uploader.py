# main_uploader.py
import json
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm

from phlop.simulator import Simulation
from phlop.split_config import SPLIT_CONFIG
from phlop.video_annotation_visualizer import VideoAnnotationVisualizer
from phlop.world.object import Object
from phlop.annotator import Annotator
from phlop.question_answer import QuestionAnswers
from phlop.advanced_physics_questions import AdvancedPhysicsQuestions
import random

from upload.config import (
    HF_REPO,
    HF_TOKEN,
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


def atomic_write_json(data, path):
    """Write JSON atomically to prevent empty/corrupted files."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        temp_name = tmp.name

    shutil.move(temp_name, path)


def upload_file_to_hf(local_path: str, repo_id: str, path_in_repo: str):
    api = HfApi(token=HF_TOKEN)

    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✅ Uploaded: {path_in_repo}")
    except Exception as e:
        print(f"❌ Failed to upload {path_in_repo}: {e}")


def upload_shard_to_hf(
    shard_id: int, shard_dir: Path, repo_id: str, split: str, parquet_filename: str
):
    api = HfApi(token=HF_TOKEN)

    print(f"\n⬆️  Uploading {split} Shard {shard_id}...")

    parquet_path = shard_dir / parquet_filename
    if not parquet_path.exists():
        print(f"❌ Parquet file not found: {parquet_path}")
        return False

    df = pd.read_parquet(parquet_path)
    try:
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"data/{split}/shard_{shard_id:04d}.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  ✅ Uploaded parquet: data/{split}/shard_{shard_id:04d}.parquet")
    except Exception as e:
        print(f"  ❌ Failed to upload parquet: {e}")
        return False

    video_cols = [col for col in df.columns if "video" in col or "annotated" in col]
    metadata_cols = [col for col in df.columns if "metadata" in col or "qa" in col]

    total_files = len(df) * (len(video_cols) + len(metadata_cols))
    uploaded_count = 0

    for idx, row in df.iterrows():
        # Upload videos
        for col in video_cols:
            local_file = row[col]
            if pd.notna(local_file) and os.path.exists(local_file):
                filename = Path(local_file).name
                repo_path = f"data/{split}/shard_{shard_id:04d}/{idx:04d}/{filename}"

                try:
                    api.upload_file(
                        path_or_fileobj=local_file,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                    uploaded_count += 1
                    print(f"  ✅ [{uploaded_count}/{total_files}] {repo_path}")
                except Exception as e:
                    print(f"  ⚠️  Skipped {filename}: {str(e)[:50]}")

        # Upload metadata files
        for col in metadata_cols:
            local_file = row[col]
            if pd.notna(local_file) and os.path.exists(local_file):
                filename = Path(local_file).name
                repo_path = f"data/{split}/shard_{shard_id:04d}/{idx:04d}/{filename}"

                try:
                    api.upload_file(
                        path_or_fileobj=local_file,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                    uploaded_count += 1
                    print(f"  ✅ [{uploaded_count}/{total_files}] {repo_path}")
                except Exception as e:
                    print(f"  ⚠️  Skipped {filename}: {str(e)[:50]}")

    print(
        f"  ✅ Shard {shard_id} upload complete ({uploaded_count}/{total_files} files)"
    )
    return True


def calculate_adaptive_distance(elevation_angle, angle_range, distance_range):
    """
    Calculate distance based on elevation angle.
    Steeper angles (more negative) = closer distance.
    Shallow angles (less negative) = farther distance.

    Args:
        elevation_angle: Camera elevation in degrees (-90 to 0, where -90 is top-down)
        angle_range: Tuple (min_elev, max_elev) used in this split
        distance_range: Tuple (min_dist, max_dist) for this split

    Returns:
        Adaptive distance as float

    Example:
        elev = -60  # Steep downward
        dist = calculate_adaptive_distance(elev, (-80, -10), (1.5, 4.5))
        # Returns close distance because steep angle

        elev = -15  # Shallow angle
        dist = calculate_adaptive_distance(elev, (-80, -10), (1.5, 4.5))
        # Returns far distance because shallow angle
    """
    min_elev, max_elev = angle_range
    min_dist, max_dist = distance_range

    # Normalize elevation to [0, 1] where:
    # 0 = most top-down (min_elev, e.g., -80)
    # 1 = shallowest (max_elev, e.g., -10)
    normalized = (elevation_angle - min_elev) / (max_elev - min_elev)
    normalized = np.clip(normalized, 0.0, 1.0)

    # INVERSE relationship: steeper (lower normalized) = closer (lower distance)
    # normalized=0 (top-down) → min_dist
    # normalized=1 (shallow)  → max_dist
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

    # --- ATOMIC QA WRITE ---
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
        "qa_pairs": qa_pairs,
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
        # print(object_specs)
        # camera_cfg = sample_camera("train", cam_mode)
        camera_cfg = sample_camera_for_split(split="train", mode=cam_mode)
        # print("camera_cfg", camera_cfg)

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
                "video_file": out["video"],
                "annotated_file": out["annotated"],
                "metadata_file": out["metadata"],
                "qa_file": out["qa"],
                "qa_pairs": json.dumps(out["qa_pairs"]),
            }
        )
        print(' out["metadata"] ', out["metadata"])

    # df = pd.DataFrame(rows)
    # df.to_parquet(OUTPUT_DIR / f"train_shard_{shard_id:04d}.parquet", index=False)
    df = pd.DataFrame(rows)
    parquet_filename = f"train_shard_{shard_id:04d}.parquet"
    parquet_path = OUTPUT_DIR / parquet_filename
    df.to_parquet(parquet_path, index=False)

    print(f"✅ Saved parquet: {parquet_filename}")

    # # Upload to HuggingFace if requested
    # # if upload_to_hf and hf_repo_id:
    # upload_shard_to_hf(
    #         shard_id=shard_id,
    #         shard_dir=OUTPUT_DIR,
    #         repo_id=HF_REPO,
    #         split="train",
    #         parquet_filename=parquet_filename,
    #     )


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
        # print("objects ", objects)

        floor = meta["world"]["floor"]
        lights = meta["world"]["lights"]

        camera_cfg["mode"] = 1  # moving

        # static_video_path = scene_path / "simulation_objects_static.mp4"
        # os.rename(out_static["video"], static_video_path)
        # print("====================")
        # print("camera_cfg ", camera_cfg)

        # print(out_static["qa"])
        # print("====================")
        # print(out_static["qa_pairs"])
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

        # dynamic_video_path = scene_path / "simulation_objects_dynamic.mp4"
        # os.rename(out_dynamic["video"], dynamic_video_path)
        rows.append(
            {
                "id": global_idx,
                "split": split,
                "num_objects": num_objects,
                # Static
                "static_video": out_static["video"],
                "static_annotated": out_static["annotated"],
                "static_metadata": out_static["metadata"],
                "static_qa": out_static["qa"],
                "static_qa_pairs": json.dumps(out_static["qa_pairs"]),
                # Moving
                "moving_video": out_moving["video"],
                "moving_annotated": out_moving["annotated"],
                "moving_metadata": out_moving["metadata"],
                "moving_qa": out_moving["qa"],
                "moving_qa_pairs": json.dumps(out_moving["qa_pairs"]),
            }
        )
        # break

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_DIR / f"{split}_shard_{shard_id:04d}.parquet", index=False)

    print(f"✅ Finished {split} shard {shard_id}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--shard-id", type=int, default=0)
    # parser.add_argument("--start-idx", type=int, default=0)
    # parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    print("Generating", args)
    # api = HfApi(token=HF_TOKEN)
    # api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True, private=True)

    current_idx = 0
    shard_id = 0
    split = args.split

    # TRAIN_COUNT, VAL_COUNT, TEST_COUNT

    if split == "train":
        # while current_idx < 10:
        total_examples = (
            10  # keep test-size behavior; replace with TRAIN_COUNT if available
        )
        total_shards = (total_examples + SHARD_SIZE - 1) // SHARD_SIZE
        for shard_id in tqdm(range(total_shards), desc="train shards"):
            # while current_idx < TRAIN_COUNT:
            print("shard_id", shard_id)
            generate_training_shard(shard_id, current_idx, SHARD_SIZE)
            current_idx += SHARD_SIZE
            shard_id += 1

    if split in ["val", "test"]:
        total_count = VAL_COUNT if split == "val" else TEST_COUNT
        while current_idx < 20:
            # while current_idx < total_count:
            generate_validation_shard(shard_id, current_idx, SHARD_SIZE, split=split)
            current_idx += SHARD_SIZE
            shard_id += 1
            break


if __name__ == "__main__":
    main()
