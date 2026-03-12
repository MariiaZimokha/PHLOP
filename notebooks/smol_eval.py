"""
SmolVLM-specific PHLOP eval: zero-shot, fine-tuning, test comparison.
Works with PHLOPDataset (path-based) returned by load_phlop_splits().

Notebook usage:
    from phlop_eval_common import load_phlop_splits
    from smol_eval import run_zero_shot_smolvlm, run_finetune_smolvlm, run_test_comparison_smolvlm
    splits = load_phlop_splits("zimmari-ai/phlop", token=True)
    run_zero_shot_smolvlm(splits, max_samples=50)
    run_finetune_smolvlm(splits, output_dir="./out", max_steps=10)
    run_test_comparison_smolvlm(splits, [("base", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"), ("easy", "./out/easy")])
"""
from __future__ import annotations

import gc
import glob
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import Trainer, TrainingArguments
from transformers.video_utils import VideoMetadata
from peft import LoraConfig, get_peft_model

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from phlop_eval_common import (
    compute_metrics,
    print_metrics,
    load_qa_from_path,
    load_json_file,
    get_physical_props,
    get_taxonomy,
    build_dynamic_prompt,
    flatten_taxonomy,
    DIFFICULTIES,
    FINE_TUNE_CONFIGS,
    get_val_difficulty_filter,
    PHLOPSplits,
)


NUM_VIDEO_FRAMES = 8


def _load_video_as_pil(
    video_path: str, num_frames: int = NUM_VIDEO_FRAMES,
) -> tuple[list[Image.Image], dict]:
    """Load video from path and return (PIL frames, video_info dict).

    video_info contains total_num_frames, fps, duration, and frames_indices
    so the SmolVLM2 processor can generate correct per-frame timestamps.
    Uses PyAV for broad platform support (including macOS ARM64).
    """
    empty_info: dict = {"total_num_frames": 0, "fps": 25.0, "duration": 0.0, "frames_indices": []}
    if not video_path or not os.path.exists(video_path):
        return [], empty_info
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames
        fps = float(stream.average_rate) if stream.average_rate else 25.0

        if total <= 0:
            all_frames = [frame.to_image() for frame in container.decode(video=0)]
            total = len(all_frames)
            if total == 0:
                container.close()
                return [], empty_info
            indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist()
            pil_frames = [all_frames[j] for j in indices]
        else:
            indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist()
            index_set = set(indices)
            sampled = {}
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx in index_set:
                    sampled[frame_idx] = frame.to_image()
                if len(sampled) == len(indices):
                    break
            pil_frames = [sampled[j] for j in indices if j in sampled]

        container.close()
        duration = total / fps
        video_info = {
            "total_num_frames": total,
            "fps": fps,
            "duration": duration,
            "frames_indices": indices,
        }
        return pil_frames, video_info
    except Exception as e:
        print(f"  Warning: failed to load video {video_path}: {e}")
        return [], empty_info


def _format_physics_signals(signals: dict) -> str:
    if not signals:
        return "None"
    return "\n".join(f"- {k}: {v}" for k, v in signals.items())


# ---------------------------------------------------------------------------
# Dataset wrappers that work with PHLOPDataset (path-based) from load_phlop_splits
# ---------------------------------------------------------------------------


class PHLOPTrainDataset(Dataset):
    """Expands each scene into one sample per (scene, question). Optional difficulty_filter.
    Works with PHLOPDataset backends (path-based)."""

    def __init__(
        self,
        base_dataset,
        difficulty_filter: Optional[list[str]] = None,
        camera_mode: str = "static",
        num_frames: int = NUM_VIDEO_FRAMES,
    ):
        self.base = base_dataset
        self.camera_mode = camera_mode
        self.num_frames = num_frames
        self.difficulty_filter = None if difficulty_filter is None else set(d.lower() for d in difficulty_filter)
        self.index = []
        print(f"  Building training index (camera={camera_mode}, difficulty={difficulty_filter})...")
        for scene_idx in tqdm(range(len(self.base)), desc="Indexing scenes"):
            try:
                sample = self.base[scene_idx]
            except Exception:
                continue
            qa_path = (sample.get("qa") or {}).get(camera_mode)
            qa_list = load_qa_from_path(qa_path)
            for qa_idx, qa in enumerate(qa_list):
                if self.difficulty_filter is not None:
                    d = (qa.get("difficulty") or "unknown").lower()
                    if d not in self.difficulty_filter:
                        continue
                self.index.append((scene_idx, qa_idx))
        print(f"  Training index: {len(self.index)} (scene, question) pairs")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene_idx, qa_idx = self.index[idx]
        sample = self.base[scene_idx]

        cam = sample.get("camera_mode", self.camera_mode) or self.camera_mode
        video_path = (sample.get("videos") or {}).get(cam)
        meta_path = (sample.get("metadata") or {}).get(cam)
        qa_path = (sample.get("qa") or {}).get(cam)

        metadata = load_json_file(meta_path)
        physical_props = get_physical_props(metadata)
        qa_list = load_qa_from_path(qa_path)
        qa_entry = qa_list[qa_idx] if qa_idx < len(qa_list) else {}

        video_frames, video_info = _load_video_as_pil(video_path, self.num_frames)

        question = qa_entry.get("question", "")
        answer = qa_entry.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        if qa_entry.get("options"):
            opts = "\n".join(f"- {o}" for o in qa_entry["options"])
            question = f"{question}\nOptions:\n{opts}"

        physics_text = _format_physics_signals(qa_entry.get("physics_signals", {}))
        prompt = (
            "You are a physics reasoning system.\n\n"
            "Known physical setup:\n"
            f"{_format_physics_signals(physical_props)}\n\n"
            "<video>\n\n"
            f"Question:\n{question}\n\n"
            "Respond in the following format:\n"
            "Physics:\n- <key>: <value>\nAnswer:\n"
        )
        target = "Physics:\n" + physics_text + "\n\nAnswer:\n" + str(answer)

        return {
            "video": video_frames,
            "video_info": video_info,
            "prompt": prompt,
            "target": target,
            "metadata": metadata,
            "qa_entry": qa_entry,
        }


class PHLOPValDataset(Dataset):
    """Eval dataset: one sample per (scene, question) — all questions are evaluated.
    Works with PHLOPDataset backends (path-based)."""

    def __init__(self, base_dataset, camera_mode: str = "static", num_frames: int = NUM_VIDEO_FRAMES):
        self.base = base_dataset
        self.camera_mode = camera_mode
        self.num_frames = num_frames
        self.index = []
        print(f"  Building eval index (camera={camera_mode})...")
        for scene_idx in tqdm(range(len(self.base)), desc="Indexing eval scenes"):
            try:
                sample = self.base[scene_idx]
            except Exception:
                continue
            qa_path = (sample.get("qa") or {}).get(camera_mode)
            qa_list = load_qa_from_path(qa_path)
            if not qa_list:
                other_cam = "moving" if camera_mode == "static" else "static"
                qa_path = (sample.get("qa") or {}).get(other_cam)
                qa_list = load_qa_from_path(qa_path)
            for qa_idx in range(len(qa_list)):
                self.index.append((scene_idx, qa_idx))
        print(f"  Eval index: {len(self.index)} (scene, question) pairs from {len(self.base)} scenes")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene_idx, qa_idx = self.index[idx]
        sample = self.base[scene_idx]

        cam = self.camera_mode
        video_path = (sample.get("videos") or {}).get(cam)
        meta_path = (sample.get("metadata") or {}).get(cam)
        qa_path = (sample.get("qa") or {}).get(cam)

        qa_list = load_qa_from_path(qa_path)
        if not qa_list:
            other_cam = "moving" if cam == "static" else "static"
            qa_path = (sample.get("qa") or {}).get(other_cam)
            qa_list = load_qa_from_path(qa_path)
            if qa_list:
                video_path = (sample.get("videos") or {}).get(other_cam)
                meta_path = (sample.get("metadata") or {}).get(other_cam)

        qa_entry = qa_list[qa_idx] if qa_idx < len(qa_list) else {}
        metadata = load_json_file(meta_path)
        video_frames, video_info = _load_video_as_pil(video_path, self.num_frames)

        question = qa_entry.get("question", "")
        answer = qa_entry.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        if qa_entry.get("options"):
            opts = "\n".join(f"- {o}" for o in qa_entry["options"])
            question = f"{question}\nOptions:\n{opts}"

        physics_text = _format_physics_signals(qa_entry.get("physics_signals", {}))
        prompt = (
            "You are a physics reasoning system.\n"
            "First infer the physical properties and events.\n"
            "Then answer the question.\n\n"
            "<video>\n\n"
            f"Question:\n{question}\n\n"
            "Respond in the following format:\n"
            "Physics:\n- <key>: <value>\nAnswer:\n"
        )
        target = "Physics:\n" + physics_text + "\n\nAnswer:\n" + str(answer)

        return {
            "video": video_frames,
            "video_info": video_info,
            "prompt": prompt,
            "target": target,
            "metadata": metadata,
            "qa_entry": qa_entry,
            "camera_mode": cam,
            "video_path": video_path or "",
            "scene_idx": scene_idx,
            "qa_idx": qa_idx,
        }


# ---------------------------------------------------------------------------
# SmolVLM prediction collection and training
# ---------------------------------------------------------------------------


def _build_video_metadata(video_info: dict) -> VideoMetadata:
    """Construct a VideoMetadata from the dict returned by _load_video_as_pil."""
    return VideoMetadata(
        total_num_frames=video_info.get("total_num_frames", 0),
        fps=video_info.get("fps", 25.0),
        duration=video_info.get("duration", 0.0),
        frames_indices=video_info.get("frames_indices", []),
    )


FLUSH_BATCH_SIZE = 500


def collect_predictions_smolvlm(model, processor, dataset, max_samples=None, device=None):
    """Run model on dataset, return list of result dicts.

    Results are flushed to temporary batch files every FLUSH_BATCH_SIZE items
    to keep RAM usage bounded, then merged at the end.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    tmp_dir = tempfile.mkdtemp(prefix="smolvlm_batches_")
    batch_files: list[str] = []
    buffer: list[dict] = []

    def _flush():
        if not buffer:
            return
        batch_path = os.path.join(tmp_dir, f"batch_{len(batch_files):04d}.json")
        with open(batch_path, "w") as f:
            json.dump(buffer, f, default=str)
        batch_files.append(batch_path)
        buffer.clear()
        gc.collect()

    for i in tqdm(range(n), desc="Collecting predictions"):
        try:
            item = dataset[i]
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"  Skipping idx {i}: {e}")
            continue
        if not item.get("video"):
            print(f"  Skipping idx {i}: no video frames loaded")
            continue
        vm = _build_video_metadata(item.get("video_info", {}))
        inputs = processor(
            videos=[[item["video"]]],
            text=[item["prompt"]],
            video_metadata=[[vm]],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        del inputs, outputs
        torch.cuda.empty_cache()

        qa = item.get("qa_entry") or {}
        answer_raw = qa.get("answer", "")
        if isinstance(answer_raw, list):
            answer_str = ", ".join(str(a) for a in answer_raw)
        else:
            answer_str = str(answer_raw)

        tax_labels = list(flatten_taxonomy(item.get("metadata", {})))

        buffer.append({
            "idx": i,
            "scene_idx": item.get("scene_idx"),
            "qa_idx": item.get("qa_idx"),
            "video_path": item.get("video_path", ""),
            "camera_mode": item.get("camera_mode", ""),
            "prompt": item["prompt"],
            "question": qa.get("question", ""),
            "options": qa.get("options"),
            "difficulty": qa.get("difficulty", "unknown"),
            "question_type": qa.get("question_type") or qa.get("category") or "unknown",
            "category": qa.get("category") or "unknown",
            "true_answer": answer_str,
            "prediction": pred,
            "target": item["target"],
            "taxonomy_labels": tax_labels,
            "physics_signals": qa.get("physics_signals", {}),
        })
        del item

        if len(buffer) >= FLUSH_BATCH_SIZE:
            _flush()

    _flush()

    results: list[dict] = []
    for bp in batch_files:
        with open(bp, "r") as f:
            results.extend(json.load(f))
        os.remove(bp)
    os.rmdir(tmp_dir)

    return results


class SmolVLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        videos = [[x["video"]] for x in batch]
        texts = [x["prompt"] + x["target"] for x in batch]
        video_metadata = [[_build_video_metadata(x.get("video_info", {}))] for x in batch]
        model_inputs = self.processor(
            videos=videos,
            text=texts,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


class SmolVLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def run_zero_shot_smolvlm(
    splits: PHLOPSplits,
    camera_mode: str = "static",
    max_samples: Optional[int] = None,
    model=None,
    processor=None,
    device: Optional[str] = None,
    results_dir: Optional[str] = "results",
    eval_splits: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Run zero-shot SmolVLM on the requested splits.

    Args:
        eval_splits: Which splits to evaluate. Defaults to ["validation", "test"].
                     Pass e.g. ["validation"] to run a single split.

    Returns dict[split_name, {"metrics": ..., "n_questions": ...}].
    Saves raw predictions to results_dir if set.
    """
    eval_splits = eval_splits or ["validation", "test"]
    available = [s for s in eval_splits if s in splits]
    if not available:
        print(f"None of {eval_splits} found in splits.")
        return {}
    if processor is None:
        processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    if model is None:
        model = AutoModelForImageTextToText.from_pretrained(
            SMOLVLM_MODEL_ID,
            dtype=torch.bfloat16,
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    print(f"Using device: {device}")
    model = model.to(device)
    out = {}
    for split_name in available:
        val_ds = PHLOPValDataset(splits[split_name], camera_mode=camera_mode)
        n_eval = len(val_ds) if max_samples is None else min(max_samples, len(val_ds))
        print(f"\n--- Zero-shot on {split_name} ({camera_mode}): {len(val_ds)} questions (evaluating {n_eval}) ---")
        results = collect_predictions_smolvlm(model, processor, val_ds, max_samples=max_samples, device=device)
        for r in results:
            r["split"] = split_name
            r["camera_mode"] = camera_mode
        metrics = compute_metrics(results)
        print_metrics(metrics, f"Zero-shot {split_name}")
        out[split_name] = {"metrics": metrics, "n_questions": len(results)}
        if results_dir:
            pred_path = os.path.join(results_dir, f"smolvlm_zero_shot_predictions_{split_name}_{camera_mode}.json")
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved {len(results)} predictions to {pred_path}")
    return out


def run_finetune_smolvlm(
    splits: PHLOPSplits,
    output_dir: str = "./smolvlm_physics_full",
    config_name: Optional[str] = None,
    max_steps: int = 50,
    camera_mode: str = "static",
) -> list[str]:
    """Fine-tune SmolVLM on train split. config_name=None runs all four configs. Returns list of checkpoint dirs."""
    if "train" not in splits:
        print("Need train split for fine-tuning.")
        return []
    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    data_collator = SmolVLMDataCollator(processor)
    train_backend = splits["train"]
    val_backend = splits.get("validation", train_backend)
    configs = {config_name: FINE_TUNE_CONFIGS[config_name]} if config_name else FINE_TUNE_CONFIGS
    saved_dirs = []
    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Fine-tuning config: {cfg_name}")
        train_diff = cfg["train_difficulty"]
        val_diff = get_val_difficulty_filter(train_diff) if cfg["val_on_rest"] else None
        print(f"  Train difficulties: {train_diff}, Val difficulties: {val_diff}")

        train_ds = PHLOPTrainDataset(train_backend, difficulty_filter=train_diff, camera_mode=camera_mode)
        if len(train_ds) == 0:
            print(f"  Skipping config {cfg_name}: no training samples.")
            continue
        if cfg["val_on_rest"] and val_backend is not None:
            val_ds = PHLOPTrainDataset(val_backend, difficulty_filter=val_diff, camera_mode=camera_mode)
        else:
            val_ds = PHLOPValDataset(val_backend, camera_mode=camera_mode)

        model = AutoModelForImageTextToText.from_pretrained(
            SMOLVLM_MODEL_ID,
            dtype=torch.bfloat16,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        ckpt_dir = os.path.join(output_dir, cfg_name)
        training_args = TrainingArguments(
            output_dir=ckpt_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            max_steps=max_steps,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            eval_strategy="steps",
            eval_steps=20,
            remove_unused_columns=False,
            report_to="none",
        )
        trainer = SmolVLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model()
        saved_dirs.append(ckpt_dir)
        del model
        torch.cuda.empty_cache()
    return saved_dirs


def run_test_comparison_smolvlm(
    splits: PHLOPSplits,
    model_checkpoints: list[tuple[str, str]],
    camera_mode: str = "static",
    results_dir: Optional[str] = "results",
    eval_splits: Optional[list[str]] = None,
) -> list[tuple[str, dict]]:
    """Evaluate each (model_name, checkpoint_path) on specified splits.
    Returns list of (model_name, metrics_dict) where metrics_dict is keyed by split name.
    Defaults to test-only to avoid leaking validation signal used during training."""
    eval_splits = eval_splits or ["test"]
    available = [s for s in eval_splits if s in splits]
    if not available:
        print(f"None of {eval_splits} found in splits.")
        return []
    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    table = []
    for model_name, ckpt in model_checkpoints:
        print(f"\n--- {model_name} ({camera_mode}) ---")
        model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16)
        model = model.to(device)
        model_metrics = {}
        for split_name in available:
            ds = PHLOPValDataset(splits[split_name], camera_mode=camera_mode)
            results = collect_predictions_smolvlm(model, processor, ds, device=device)
            for r in results:
                r["split"] = split_name
                r["camera_mode"] = camera_mode
                r["model"] = model_name
            metrics = compute_metrics(results)
            model_metrics[split_name] = metrics
            print(f"  {split_name}: answer_acc={metrics['answer_accuracy']:.4f}, "
                  f"physics_acc={metrics['physics_signal_accuracy']:.4f}, "
                  f"tax_f1={metrics['taxonomy_f1']:.4f}")
            if results_dir:
                out_path = os.path.join(results_dir, f"smolvlm_finetuned_predictions_{model_name}_{split_name}_{camera_mode}.json")
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"    Saved {len(results)} predictions to {out_path}")
        table.append((model_name, model_metrics))
        del model
        torch.cuda.empty_cache()
    return table
