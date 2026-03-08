"""
PHLOP eval common: model-agnostic dataset loading, metadata extraction,
evaluation helpers, and metrics.

All model notebooks share:
  1. load_phlop_splits() - load data from HF, wrap in PHLOPDataset
  2. load_json_file()    - load JSON from a resolved path
  3. get_physical_props() - extract physical properties from obj.json
  4. get_taxonomy()       - extract taxonomy labels from obj.json frames
  5. describe_obj()       - "red sphere" style description from props dict
  6. evaluate_response_quality() - compare model response to ground truth
  7. build_dynamic_prompt()      - build prompt with taxonomy/physics context
  8. save_and_score_results()    - save JSON, compute accuracy per option
  9. compute_metrics(), print_metrics() - detailed metrics

Usage (in any model notebook):
    from phlop_eval_common import (
        load_phlop_splits, load_json_file, get_physical_props, get_taxonomy,
        evaluate_response_quality, build_dynamic_prompt, save_and_score_results,
    )
    splits = load_phlop_splits("zimmari-ai/phlop", token=True)
    for idx in range(len(splits["test"])):
        sample = splits["test"][idx]
        ...
"""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Union

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

try:
    import matplotlib.colors as mcolors
except ImportError:
    mcolors = None

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import PHLOPDataset

PHLOPSplits = dict[str, PHLOPDataset]

DEFAULT_REPO_ID = "zimmari-ai/phlop"
DIFFICULTIES = ("easy", "medium", "hard", "very_hard")

EVAL_OPTIONS = [
    {"is_taxonomy": False, "is_physics": False, "name": "no_additional_info"},
    {"is_taxonomy": True, "is_physics": False, "name": "taxonomy_only"},
    {"is_taxonomy": False, "is_physics": True, "name": "physics_only"},
    {"is_taxonomy": True, "is_physics": True, "name": "taxonomy_and_physics"},
]

FINE_TUNE_CONFIGS = {
    "easy": {"train_difficulty": ["easy"], "val_on_rest": True},
    "easy_medium": {"train_difficulty": ["easy", "medium"], "val_on_rest": True},
    "hard": {"train_difficulty": ["hard"], "val_on_rest": True},
    "full": {"train_difficulty": None, "val_on_rest": False},
}


def get_val_difficulty_filter(train_difficulty: Optional[list[str]]) -> Optional[list[str]]:
    """Validation = difficulties NOT in the training set."""
    if train_difficulty is None:
        return None
    return [d for d in DIFFICULTIES if d not in set(train_difficulty)]


def filter_qa_by_difficulty(qa_list: list[dict], difficulties: Optional[list[str]]) -> list[dict]:
    """Filter a QA list to only include questions matching the given difficulties.
    If difficulties is None, returns all questions (no filtering)."""
    if difficulties is None:
        return qa_list
    allowed = {d.lower() for d in difficulties}
    return [qa for qa in qa_list if (qa.get("difficulty") or "unknown").lower() in allowed]


def build_training_index(
    ds: PHLOPDataset,
    difficulty_filter: Optional[list[str]] = None,
    camera_mode: str = "static",
) -> list[tuple[int, int]]:
    """
    Build an index of (scene_idx, qa_idx) pairs for training.
    Expands each scene into one sample per question, optionally filtered by difficulty.
    Returns list of (scene_idx, qa_idx) tuples.
    """
    index = []
    for scene_idx in tqdm(range(len(ds)), desc="Building training index"):
        try:
            sample = ds[scene_idx]
        except Exception:
            continue
        qa_path = (sample.get("qa") or {}).get(camera_mode)
        if not qa_path:
            continue
        qa_list = load_qa_from_path(qa_path)
        qa_list = filter_qa_by_difficulty(qa_list, difficulty_filter)
        for qa_idx in range(len(qa_list)):
            index.append((scene_idx, qa_idx))
    print(f"  Training index: {len(index)} (scene, question) pairs")
    return index


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_phlop_splits(
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[Union[str, bool]] = None,
    extract_root: Optional[str] = None,
    splits_to_load: Optional[list[str]] = None,
) -> PHLOPSplits:
    """
    Load PHLOP parquet metadata from HF and wrap each split in a PHLOPDataset.

    Args:
        repo_id: Hugging Face dataset repo.
        token: HF token (True = cached login, or "hf_..." string).
        extract_root: Directory for extracted shard zips.
        splits_to_load: Which splits to load. Defaults to ["train", "validation", "test"].

    Returns:
        dict mapping split name -> PHLOPDataset
    """
    hf_token = token if token is not None else os.environ.get("HF_TOKEN") or True
    wanted = splits_to_load or ["train", "validation", "test"]

    data_files = {}
    if "train" in wanted:
        data_files["train"] = "train_shard_*.parquet"
    if "validation" in wanted:
        data_files["validation"] = "*val_shard_*.parquet"
    if "test" in wanted:
        data_files["test"] = "*test_shard_*.parquet"

    print(f"Loading splits {list(data_files.keys())} from {repo_id} ...")
    raw = load_dataset(repo_id, data_files=data_files, token=hf_token)

    splits: PHLOPSplits = {}
    for name in raw:
        ds = PHLOPDataset(
            raw[name],
            repo_id=repo_id,
            extract_root=extract_root,
            token=hf_token,
        )
        splits[name] = ds
        print(f"  {name}: {len(ds)} scenes")

    return splits


# ---------------------------------------------------------------------------
# JSON / metadata helpers
# ---------------------------------------------------------------------------

def load_json_file(path: Optional[str]) -> Any:
    """Load a JSON file; returns {} on any error."""
    if not path or not isinstance(path, str) or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_qa_from_path(path: Optional[str]) -> list[dict]:
    """Load QA list from a JSON file. Handles both list and dict-with-questions formats."""
    raw = load_json_file(path)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("questions", raw.get("qa", []))
    return []


def rgba_to_name(rgba) -> str:
    """Convert RGBA list to closest CSS color name (needs matplotlib)."""
    if mcolors is None or not rgba or len(rgba) < 3:
        return "unknown color"
    rgb = tuple(rgba[:3])
    min_dist = float("inf")
    best = "unknown color"
    for name, hex_val in mcolors.CSS4_COLORS.items():
        named_rgb = mcolors.to_rgb(hex_val)
        dist = sum((a - b) ** 2 for a, b in zip(rgb, named_rgb))
        if dist < min_dist:
            min_dist = dist
            best = name
    return best.replace("grey", "gray")


MIN_VISIBLE_SIZE = 10

def _is_visible(bbox) -> bool:
    if not bbox or bbox == [[0, 0], [0, 0]]:
        return False
    (x0, y0), (x1, y1) = bbox
    return abs(x1 - x0) >= MIN_VISIBLE_SIZE and abs(y1 - y0) >= MIN_VISIBLE_SIZE


def get_physical_props(metadata: dict) -> dict:
    """
    Extract physical properties dict from obj.json metadata.
    Returns: {obj_id: {"mass", "friction", "elasticity", "velocity", "material", "shape", "color"}}
    """
    objects = metadata.get("objects", [])
    frames = metadata.get("frames", [])

    visible_ids = {
        obj_id
        for fr in frames
        for obj_id, obj_state in fr.get("objects", {}).items()
        if _is_visible(obj_state.get("bbox", [[0, 0], [0, 0]]))
    }

    props = {}
    for obj in objects:
        obj_id = obj.get("id")
        if not obj_id or obj_id not in visible_ids:
            continue
        fr = obj.get("friction", "").split()
        shape = obj.get("geom_type", "unknown")
        rgba_str = obj.get("visual", {}).get("rgba", "")
        color = [float(x) for x in rgba_str.split()] if rgba_str else []
        color_name = rgba_to_name(color)
        mass = obj.get("mass", None)
        try:
            mass = float(mass) if mass is not None else 1.0
        except (ValueError, TypeError):
            mass = 1.0

        props[obj_id] = {
            "mass": mass,
            "friction": [float(x) for x in fr] if fr else [0.4],
            "elasticity": obj.get("elasticity", 0.0),
            "velocity": obj.get("velocity", [0, 0, 0]),
            "material": obj.get("material", "unknown"),
            "shape": shape,
            "color": color_name,
        }
    return props


def get_taxonomy(metadata: dict) -> dict:
    """
    Extract taxonomy dict from obj.json metadata.
    Returns: {obj_id: [[labels_frame1], [labels_frame2], ...]}
    """
    frames = metadata.get("frames", [])
    taxonomy: dict[str, list] = {}
    for fr in frames:
        for obj_id, obj_state in fr.get("objects", {}).items():
            if not _is_visible(obj_state.get("bbox", [[0, 0], [0, 0]])):
                continue
            labels = []
            for tax_entry in obj_state.get("taxonomy", []):
                labels.extend(tax_entry.get("labels", []))
            taxonomy.setdefault(obj_id, []).append(labels)
    return taxonomy


def compute_has_collision(metadata: dict) -> bool:
    """Check if any collision occurred in the scene."""
    _coll_re = re.compile(r"collision", re.IGNORECASE)
    for fr in metadata.get("frames", []):
        if fr.get("interactions"):
            return True
        for obj_state in fr.get("objects", {}).values():
            for tax in obj_state.get("taxonomy", []):
                if any(_coll_re.search(lbl) for lbl in tax.get("labels", [])):
                    return True
    return False


def describe_obj(props: dict) -> str:
    """e.g. 'red sphere' from a single object's physical_props entry."""
    return f"{props.get('color', 'unknown')} {props.get('shape', 'object')}"


# ---------------------------------------------------------------------------
# Prompt building (shared across models)
# ---------------------------------------------------------------------------

def build_dynamic_prompt(
    taxonomy: dict = None,
    physical_props: dict = None,
    question: str = "",
    options: list = None,
    explanation: str = None,
    num_frames: int = 32,
    video_duration: float = 15.0,
    fps: int = 25,
) -> str:
    """
    Build a prompt string from taxonomy, physical properties, and the question.
    Works for all models; model-specific conversation wrapping is done in the notebook.
    """
    taxonomy = taxonomy or {}
    physical_props = physical_props or {}

    # Object behaviors from taxonomy
    object_behaviors = []
    behavior_dict = {}
    valid_objects = []
    for obj_id, frame_labels in taxonomy.items():
        all_labels = set()
        for labels in frame_labels:
            all_labels.update(lbl for lbl in labels if lbl)
        if all_labels:
            valid_objects.append((obj_id, all_labels))

    for i, (obj_id, all_labels) in enumerate(valid_objects, start=1):
        desc = f"{physical_props.get(obj_id, {}).get('color', 'unknown')} {physical_props.get(obj_id, {}).get('shape', 'object')}"
        object_behaviors.append(f"{i}. {desc}: {', '.join(sorted(all_labels))}")
        behavior_dict[obj_id] = {"description": desc, "behaviors": all_labels}

    behavior_context = "\n".join(object_behaviors) if object_behaviors else "No behavior labels available"

    # Object properties
    if physical_props:
        physical_props_str = "\n".join(
            f"{i+1}. {v['color']} {v['shape']}: mass={v['mass']}, friction={v['friction'][0]:.2f}"
            for i, (k, v) in enumerate(physical_props.items())
        )
    else:
        physical_props_str = "No physical properties available"

    # Optional sections
    additional_information = ""
    if options:
        additional_information = f"\nOptions: {options}\nPick exactly one option."

    explanation_info = ""
    if explanation:
        explanation_info = f"\nExplanation for calculation: {explanation}"

    # Frame timestamps
    frame_interval = video_duration / num_frames if num_frames > 0 else 0
    video_context = "".join(
        f"Frame {i+1} ({i * frame_interval:.2f}s): <image>\n"
        for i in range(num_frames)
    )

    return f"""Analyze the video and answer the physics question using all available evidence.

ANSWER FORMATTING:
   - yes/no: lowercase, no punctuation
   - numbers: digits only (e.g. "3")
   - objects: exact name (e.g. "darkcyan sphere")
   - percentages: "X%" (e.g. "42%")
   - time: "Xs" (e.g. "0.5s")
   - multi-choice: exact option text only

Provide ONLY the final answer with no explanation.

Video details:
 - Duration: {video_duration:.0f} seconds
 - FPS: {fps}
 - Frames sampled: {num_frames}

VIDEO FRAMES:
{video_context}
OBJECT PROPERTIES:
{physical_props_str}

OBJECT BEHAVIORS:
{behavior_context}

QUESTION: {question}{explanation_info}{additional_information}
ANSWER:
""".strip()


# ---------------------------------------------------------------------------
# Response evaluation (shared across all models)
# ---------------------------------------------------------------------------

def evaluate_response_quality(
    response: str,
    true_answer,
    taxonomy: dict = None,
    physical_props: dict = None,
) -> dict:
    """
    Compare model response to ground truth. Returns {"correct": 0|1, "error": str|None}.
    Identical logic used by all three model notebooks.
    """
    def normalize_answer(ans):
        if isinstance(ans, (list, tuple)):
            return [str(a).strip().lower() for a in ans]
        if isinstance(ans, (int, float)):
            return [str(float(ans))]
        ans = str(ans).strip().lower()
        if ans.startswith("no collision"):
            return ["0"]
        if ans in ["no collisions detected", "none"]:
            return ["0"]
        if ans.startswith("yes"):
            return ["yes"]
        if ans.startswith("no"):
            return ["no"]
        if "%" in ans:
            ans = ans.replace("%", "")
        if ans.endswith("s") and ans[:-1].replace(".", "").isdigit():
            ans = ans[:-1]
        ans = "".join(c for c in ans if c.isdigit() or c in [".", "-"])
        return [ans]

    norm_response = normalize_answer(response)
    norm_true = normalize_answer(true_answer)

    # Multiple correct answers
    if isinstance(true_answer, (list, tuple)) or len(norm_true) > 1:
        for resp_item in norm_response:
            if any(resp_item in true for true in norm_true):
                return {"correct": 1, "error": None}
        return {"correct": 0, "error": "value"}

    # Numeric comparison with tolerance
    try:
        resp_num = float(norm_response[0])
        true_num = float(norm_true[0])
        if abs(resp_num - true_num) <= 0.05 * abs(true_num):
            return {"correct": 1, "error": None}
        return {"correct": 0, "error": "value"}
    except (ValueError, TypeError, IndexError):
        pass

    # Exact string match
    if norm_response and norm_true and norm_response[0] == norm_true[0]:
        return {"correct": 1, "error": None}

    return {"correct": 0, "error": "value"}


# ---------------------------------------------------------------------------
# Results saving and accuracy computation
# ---------------------------------------------------------------------------

def save_and_score_results(results: list[dict], output_path: str = "results.json") -> dict:
    """
    Save results to JSON and compute accuracy per option variant.
    Returns: {"overall": float, "by_option": {option_name: float}}
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} results to {output_path}")

    if not results:
        return {"overall": 0.0, "by_option": {}}

    by_option: dict[str, list] = defaultdict(list)
    for r in results:
        by_option[r.get("option", "unknown")].append(r.get("correct", 0))

    scores = {}
    for opt, vals in sorted(by_option.items()):
        acc = sum(vals) / len(vals)
        scores[opt] = acc
        print(f"  {opt} accuracy: {acc:.2f} ({sum(vals)}/{len(vals)})")

    overall = sum(r.get("correct", 0) for r in results) / len(results)
    print(f"  overall accuracy: {overall:.2f}")
    return {"overall": overall, "by_option": scores}


# ---------------------------------------------------------------------------
# Data analysis (run once to inspect QA distributions)
# ---------------------------------------------------------------------------

def run_data_analysis(
    splits: PHLOPSplits,
    output_json: Optional[str] = "phlop_data_analysis.json",
    output_csv: Optional[str] = "question_counts_by_difficulty.csv",
    camera_mode: str = "static",
) -> dict:
    """
    Analyze QA distributions across splits. Uses PHLOPDataset to resolve paths
    and extract shards on demand.

    Returns: {split_name: {"n_scenes", "total_questions", "by_difficulty", ...}}
    """
    all_stats = {}

    for split_name, ds in splits.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {split_name.upper()} ({len(ds)} scenes)")
        print(f"{'='*60}")

        qt_counter = Counter()
        ans_type_counter = Counter()
        diff_counter = Counter()
        total_questions = 0

        for idx in tqdm(range(len(ds)), desc=split_name):
            try:
                sample = ds[idx]
            except Exception as e:
                print(f"  Skipping idx {idx}: {e}")
                continue

            qa_dict = sample.get("qa", {})
            if not isinstance(qa_dict, dict):
                continue

            for cam, qa_path in qa_dict.items():
                if not qa_path:
                    continue
                qas = load_qa_from_path(qa_path)
                total_questions += len(qas)

                for qa in qas:
                    q_type = qa.get("question_type") or qa.get("category") or "unknown"
                    qt_counter[q_type] += 1

                    difficulty = qa.get("difficulty", "unknown")
                    diff_counter[difficulty] += 1

                    ans = qa.get("answer", "")
                    if isinstance(ans, bool) or str(ans).lower() in ("yes", "no", "true", "false"):
                        ans_type_counter["Boolean / Yes-No"] += 1
                    elif isinstance(ans, (int, float)) or str(ans).isnumeric():
                        ans_type_counter["Numeric / Counting"] += 1
                    elif qa.get("options") or isinstance(ans, list):
                        ans_type_counter["Multiple Choice"] += 1
                    else:
                        ans_type_counter["Open-ended Text"] += 1

        stats = {
            "n_scenes": len(ds),
            "total_questions": total_questions,
            "by_difficulty": dict(diff_counter),
            "by_question_type": dict(qt_counter),
            "by_answer_type": dict(ans_type_counter),
        }
        all_stats[split_name] = stats

        if total_questions > 0:
            print(f"  Total questions: {total_questions}")
            print(f"  Avg per scene: {total_questions / len(ds):.1f}")
            for qt, count in qt_counter.most_common():
                print(f"    {qt}: {count} ({100*count/total_questions:.1f}%)")
        else:
            print("  No questions found.")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nWrote {output_json}")

    if output_csv:
        try:
            import csv
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["split", "difficulty", "count"])
                for split_name, stats in all_stats.items():
                    for diff, count in stats["by_difficulty"].items():
                        writer.writerow([split_name, diff, count])
            print(f"Wrote {output_csv}")
        except Exception as e:
            print(f"CSV write failed: {e}")

    return all_stats


# ---------------------------------------------------------------------------
# Metric helpers (generic: any model that produces results dicts)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    text = text.lower()
    if "answer:" in text:
        return text.split("answer:", 1)[1].strip()
    return text.strip()


def answer_token_f1(pred: str, gt: str) -> float:
    pred_tokens = extract_answer(pred).split()
    gt_tokens = extract_answer(gt).split()
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    tp = sum(common.values())
    p = tp / len(pred_tokens)
    r = tp / len(gt_tokens)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def answer_accuracy(pred: str, gt: str) -> int:
    return int(extract_answer(gt) in extract_answer(pred))


def extract_physics_kv(text: str) -> dict:
    physics = {}
    for line in text.lower().splitlines():
        if line.startswith("-") and ":" in line:
            k, v = line[1:].split(":", 1)
            physics[k.strip()] = v.strip()
    return physics


def physics_signal_accuracy(pred_text: str, gt_text: str) -> Optional[float]:
    gt = extract_physics_kv(gt_text)
    pred = extract_physics_kv(pred_text)
    if not gt:
        return None
    correct = sum(1 for k, v in gt.items() if k in pred and v in pred[k])
    return correct / len(gt)


def flatten_taxonomy(metadata: dict) -> set:
    labels = set()
    for frame in metadata.get("frames", []):
        for obj in frame.get("objects", {}).values():
            for entry in obj.get("taxonomy", []):
                for label in entry.get("labels", []):
                    labels.add(label.lower())
    return labels


def extract_predicted_physics_labels(text: str) -> set:
    labels = set()
    for line in text.lower().splitlines():
        if line.startswith("-"):
            labels.add(line.split(":", 1)[-1].strip())
    return labels


def taxonomy_f1(pred: set, gt: set) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    p = tp / len(pred)
    r = tp / len(gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_metrics(results: list) -> dict:
    """Compute metrics from a list of result dicts (each with prediction, target, metadata)."""
    if not results:
        return {"answer_accuracy": 0.0, "physics_signal_accuracy": 0.0, "taxonomy_f1": 0.0, "per_question_type": {}}
    ans_scores = [answer_accuracy(r["prediction"], r["target"]) for r in results]
    phys_scores = []
    tax_scores = []
    for r in results:
        p = physics_signal_accuracy(r["prediction"], r["target"])
        if p is not None:
            phys_scores.append(p)
        gt_tax = flatten_taxonomy(r["metadata"])
        pred_tax = extract_predicted_physics_labels(r["prediction"])
        tax_scores.append(taxonomy_f1(pred_tax, gt_tax))
    by_type: dict[str, dict] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        if qt not in by_type:
            by_type[qt] = {"accuracy": [], "f1": []}
        by_type[qt]["accuracy"].append(answer_accuracy(r["prediction"], r["target"]))
        by_type[qt]["f1"].append(answer_token_f1(r["prediction"], r["target"]))
    per_question_type = {
        qt: {
            "accuracy": sum(v["accuracy"]) / len(v["accuracy"]),
            "f1": sum(v["f1"]) / len(v["f1"]),
            "count": len(v["accuracy"]),
        }
        for qt, v in by_type.items()
    }
    return {
        "answer_accuracy": sum(ans_scores) / len(ans_scores),
        "physics_signal_accuracy": sum(phys_scores) / len(phys_scores) if phys_scores else 0.0,
        "taxonomy_f1": sum(tax_scores) / len(tax_scores),
        "per_question_type": per_question_type,
    }


def print_metrics(metrics: dict, title: str) -> None:
    print(title)
    print(f"  answer_accuracy: {metrics['answer_accuracy']:.4f}")
    print(f"  physics_signal_accuracy: {metrics['physics_signal_accuracy']:.4f}")
    print(f"  taxonomy_f1: {metrics['taxonomy_f1']:.4f}")
    per = metrics.get("per_question_type", {})
    if per:
        print("  per_question_type:")
        for qt, v in sorted(per.items(), key=lambda x: -x[1]["count"]):
            print(f"    {qt}: accuracy={v['accuracy']:.4f}, f1={v['f1']:.4f}, count={v['count']}")
