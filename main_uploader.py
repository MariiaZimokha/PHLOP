import os
import json
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi

# from dataset.simulator import Simulation
from dataset.simulator import Simulation
from dataset.video_annotation_visualizer import VideoAnnotationVisualizer
from dataset.world.object import Object
from dataset.annotator import Annotator
from dataset.question_answer import QuestionAnswers
import random

from upload.config import (
    HF_TOKEN,
    VAL_COUNT,
    TEST_COUNT,
    SHARD_SIZE,
    OUTPUT_DIR,
    WIDTH,
    HEIGHT,
    FPS,
)

from upload.config import (
    TRAIN_CAM_AZIMUTH,
    TRAIN_CAM_ELEVATION,
    TRAIN_CAM_DISTANCE,
    VAL_CAM_AZIMUTH,
    VAL_CAM_ELEVATION,
    VAL_CAM_DISTANCE,
    TEST_CAM_AZIMUTH,
    TEST_CAM_ELEVATION,
    TEST_CAM_DISTANCE,
)


def upload_file_to_hf(local_path: str, hf_path: str, shard_id: int, split: str):
    api = HfApi(token=HF_TOKEN)

    print(f"⬆️ Uploading Shard {shard_id} content...")

    # # Upload Parquet (Metadata)
    # api.upload_file(
    #     path_or_fileobj=parquet_path,
    #     path_in_repo=f"data/{split}/shard_{shard_id:04d}.parquet",
    #     repo_id=HF_REPO,
    #     repo_type="dataset"
    # )

    # # Upload Videos (Files)
    # for row in shard_data:
    #     api.upload_file(
    #         path_or_fileobj=row["local_video_path"],
    #         path_in_repo=row["video_file"],  # matches the 'video_file' column in parquet
    #         repo_id=HF_REPO,
    #         repo_type="dataset"
    #     )

    # # 5. Cleanup
    # print(f"🧹 Cleaning up Shard {shard_id}")
    # shutil.rmtree(OUTPUT_DIR)  # Wipes the buffer


def sample_camera(split: str, mode: str = "static"):
    """
    Returns camera config dict based on the dataset split.
    """
    if split == "train":
        az = random.uniform(*TRAIN_CAM_AZIMUTH)
        elev = random.uniform(*TRAIN_CAM_ELEVATION)
        dist = random.uniform(*TRAIN_CAM_DISTANCE)
    elif split == "val":
        az = random.uniform(*VAL_CAM_AZIMUTH)
        elev = random.uniform(*VAL_CAM_ELEVATION)
        dist = random.uniform(*VAL_CAM_DISTANCE)
    else:  # test
        az = random.uniform(*TEST_CAM_AZIMUTH)
        elev = random.uniform(*TEST_CAM_ELEVATION)
        dist = random.uniform(*TEST_CAM_DISTANCE)

    cam_mode_int = 1 if mode == "moving" else 0

    return {
        "mode": cam_mode_int,
        "init": {"azimuth": az, "elevation": elev, "distance": dist},
    }


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
):
    """Runs simulation and returns file paths + QA JSON."""
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
    with open(qa_json_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

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
        num_objects = random.randint(2, 6)
        camera_cfg = sample_camera("train", cam_mode)

        out = run_single_simulation(
            sim=sim,
            num_objects=num_objects,
            duration=6,
            framerate=FPS,
            scene_dir=scene_path,
            camera_cfg=camera_cfg,
        )
        print("out ", out)
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

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_DIR / f"train_shard_{shard_id:04d}.parquet", index=False)


def generate_validation_shard(shard_id, start_idx, count, split):
    obj = Object()
    annotator = Annotator()
    sim = Simulation(obj, annotator=annotator, width=WIDTH, height=HEIGHT)

    rows = []
    for i in range(count):
        global_idx = start_idx + i
        base_scene_path = OUTPUT_DIR / f"{split}_{global_idx}"
        base_scene_path.mkdir(parents=True, exist_ok=True)

        num_objects = random.randint(2, 6)
        camera_cfg = sample_camera(split, "static")
        print('camera_cfg ',camera_cfg)

        out_static = run_single_simulation(
            sim=sim,
            num_objects=num_objects,
            duration=6,
            framerate=FPS,
            scene_dir=base_scene_path / "static",
            camera_cfg=camera_cfg,
        )

        meta = json.load(open(base_scene_path / "static" / "meta.json"))
        objects = meta["objects"]
        # print("objects ", objects)

        floor = meta["world"]["floor"]
        lights = meta["world"]["lights"]

        camera_cfg["mode"] = 1  # moving

        # static_video_path = scene_path / "simulation_objects_static.mp4"
        # os.rename(out_static["video"], static_video_path)
        print('====================')
        print('camera_cfg ',camera_cfg)

        print(out_static["qa"])
        print('====================')
        print(out_static["qa_pairs"])
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
        break

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
        while current_idx < 10:
            # while current_idx < TRAIN_COUNT:
            # process_shard(shard_id, current_idx, SHARD_SIZE, split=split)
            generate_training_shard(shard_id, current_idx, SHARD_SIZE)
            current_idx += SHARD_SIZE
            shard_id += 1
    if split in ["val", "test"]:
        total_count = VAL_COUNT if split == "val" else TEST_COUNT
        while current_idx < 10:
            # while current_idx < total_count:
            # process_shard(shard_id, current_idx, SHARD_SIZE, split=split)
            generate_validation_shard(shard_id, current_idx, SHARD_SIZE, split=split)
            current_idx += SHARD_SIZE
            shard_id += 1
            break


if __name__ == "__main__":
    main()
