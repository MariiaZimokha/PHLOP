"""
SmolVLM-specific PHLOP eval: zero-shot, fine-tuning, test comparison.
Import when using SmolVLM; pass the same `splits` from phlop_eval_common (load once, analyze once).

Notebook usage:
    from phlop_eval_common import load_phlop_splits, run_data_analysis
    from smol_eval import run_zero_shot_smolvlm, run_finetune_smolvlm, run_test_comparison_smolvlm
    splits = load_phlop_splits("zimmari-ai/phlop", token=True)
    run_data_analysis(splits)
    run_zero_shot_smolvlm(splits, max_samples=50)
    run_finetune_smolvlm(splits, output_dir="./out", max_steps=10)
    run_test_comparison_smolvlm(splits, [("base", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"), ("easy", "./out/easy")])
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_dataset import PHLOPVideoDataset, format_physics_signals
from phlop_eval_common import (
    compute_metrics,
    print_metrics,
    DIFFICULTIES,
    FINE_TUNE_CONFIGS,
    get_val_difficulty_filter,
    PHLOPSplits,
)

# -----------------------------------------------------------------------------
# Dataset wrappers for SmolVLM (train = one per QA, val = one random QA per scene)
# -----------------------------------------------------------------------------


class PHLOPTrainDataset(Dataset):
    """Expands each scene into one sample per (scene, question). Optional difficulty_filter."""

    def __init__(
        self,
        base_dataset: PHLOPVideoDataset,
        difficulty_filter: Optional[list[str]] = None,
    ):
        self.base = base_dataset
        self.difficulty_filter = None if difficulty_filter is None else set(d.lower() for d in difficulty_filter)
        self.index = []
        for scene_idx in range(len(self.base)):
            qa_list = self.base.get_qa_list(scene_idx)
            for qa_idx, qa in enumerate(qa_list):
                if self.difficulty_filter is not None:
                    d = (qa.get("difficulty") or "unknown").lower()
                    if d not in self.difficulty_filter:
                        continue
                self.index.append((scene_idx, qa_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene_idx, qa_idx = self.index[idx]
        data = self.base[scene_idx]
        qa_list = data["qa"]
        if isinstance(qa_list, dict):
            qa_list = qa_list.get("questions", qa_list.get("qa", []))
        if not isinstance(qa_list, list):
            qa_list = []
        qa_entry = qa_list[qa_idx]

        video = data["video"]
        if video is not None:
            video_frames = [Image.fromarray(f.astype("uint8")) for f in video]
        else:
            video_frames = []

        question = qa_entry.get("question", "")
        answer = qa_entry.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        physics_signals = qa_entry.get("physics_signals", {})
        if qa_entry.get("options"):
            opts = "\n".join(f"- {o}" for o in qa_entry["options"])
            question = f"{question}\nOptions:\n{opts}"
        physics_text = format_physics_signals(physics_signals)
        image_tokens = "\n".join(["<image>"] * len(video_frames)) if video_frames else "<image>"
        prompt = (
            "You are a physics reasoning system.\n\n"
            "Known physical setup:\n"
            f"{data.get('physics_summary', '')}\n\n"
            f"{image_tokens}\n\n"
            f"Question:\n{question}\n\n"
            "Respond in the following format:\n"
            "Physics:\n- <key>: <value>\nAnswer:\n"
        )
        target = "Physics:\n" + physics_text + "\n\nAnswer:\n" + str(answer)

        return {
            "video": video_frames,
            "prompt": prompt,
            "target": target,
            "metadata": data.get("metadata", {}),
            "qa_entry": qa_entry,
        }


class PHLOPValDataset(Dataset):
    """Eval dataset: one sample per scene, one random QA per scene."""

    def __init__(self, base_dataset: PHLOPVideoDataset, camera_mode: str = "static"):
        assert camera_mode in ("static", "moving")
        self.base = base_dataset
        self.camera_mode = camera_mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        qa_list = data.get("qa")
        if isinstance(qa_list, dict):
            qa_list = qa_list.get("questions", qa_list.get("qa", []))
        if not isinstance(qa_list, list):
            qa_list = []
        if not qa_list:
            raise ValueError(f"Empty QA list for scene idx={idx}")
        qa_entry = random.choice(qa_list)

        video = data.get("video")
        if video is not None:
            video_frames = [Image.fromarray(f.astype("uint8")) for f in video]
        else:
            video_frames = []

        question = qa_entry.get("question", "")
        answer = qa_entry.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        physics_signals = qa_entry.get("physics_signals", {})
        if qa_entry.get("options"):
            opts = "\n".join([f"- {o}" for o in qa_entry["options"]])
            question = f"{question}\nOptions:\n{opts}"
        image_tokens = "\n".join(["<image>"] * len(video_frames)) if video_frames else "<image>"
        prompt = (
            "You are a physics reasoning system.\n"
            "First infer the physical properties and events.\n"
            "Then answer the question.\n\n"
            f"{image_tokens}\n\n"
            f"Question:\n{question}\n\n"
            "Respond in the following format:\n"
            "Physics:\n- <key>: <value>\nAnswer:\n"
        )
        target = (
            "Physics:\n"
            + format_physics_signals(physics_signals)
            + "\n\nAnswer:\n"
            + str(answer)
        )
        return {
            "video": video_frames,
            "prompt": prompt,
            "target": target,
            "metadata": data.get("metadata", {}),
            "qa_entry": qa_entry,
            "camera_mode": self.camera_mode,
        }


# -----------------------------------------------------------------------------
# SmolVLM prediction collection and training
# -----------------------------------------------------------------------------


def collect_predictions_smolvlm(model, processor, dataset, max_samples=None, device=None):
    """SmolVLM-specific: run model on dataset, return list of dicts (prediction, target, metadata, ...)."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    results = []
    for i in tqdm(range(n), desc="Collecting predictions"):
        item = dataset[i]
        inputs = processor(
            videos=[item["video"]],
            text=[item["prompt"]],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        qa = item.get("qa_entry") or {}
        results.append({
            "prediction": pred,
            "target": item["target"],
            "metadata": item.get("metadata", {}),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "question_type": qa.get("question_type") or qa.get("category") or "unknown",
            "category": qa.get("category") or "unknown",
            "idx": i,
        })
    return results


class SmolVLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        videos = [x["video"] for x in batch]
        texts = [x["prompt"] + x["target"] for x in batch]
        model_inputs = self.processor(
            videos=videos,
            text=texts,
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


# -----------------------------------------------------------------------------
# Public API: run_zero_shot_smolvlm, run_finetune_smolvlm, run_test_comparison_smolvlm
# -----------------------------------------------------------------------------

SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def run_zero_shot_smolvlm(
    splits: PHLOPSplits,
    camera_mode: str = "static",
    max_samples: Optional[int] = None,
    model=None,
    processor=None,
    device: Optional[str] = None,
) -> dict[str, dict]:
    """Run zero-shot SmolVLM on validation and test. Returns dict[split_name, metrics]."""
    if "validation" not in splits or "test" not in splits:
        print("Need validation and test splits for zero-shot.")
        return {}
    if processor is None:
        processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    if model is None:
        model = AutoModelForImageTextToText.from_pretrained(
            SMOLVLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "mps" if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    model = model.to(device)
    out = {}
    for split_name in ("validation", "test"):
        val_ds = PHLOPValDataset(splits[split_name], camera_mode=camera_mode)
        print(f"\n--- Zero-shot on {split_name} ---")
        results = collect_predictions_smolvlm(model, processor, val_ds, max_samples=max_samples, device=device)
        metrics = compute_metrics(results)
        print_metrics(metrics, f"Zero-shot {split_name}")
        out[split_name] = metrics
    return out


def run_finetune_smolvlm(
    splits: PHLOPSplits,
    output_dir: str = "./smolvlm_physics_full",
    config_name: Optional[str] = None,
    max_steps: int = 50,
    camera_mode: str = "static",
) -> list[str]:
    """Fine-tune SmolVLM on train (optionally filtered by difficulty). config_name=None runs all four configs. Returns list of checkpoint dirs."""
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
    train_video_ds = splits["train"]
    val_video_ds = splits.get("validation", train_video_ds)
    configs = {config_name: FINE_TUNE_CONFIGS[config_name]} if config_name else FINE_TUNE_CONFIGS
    saved_dirs = []
    for cfg_name, cfg in configs.items():
        train_diff = cfg["train_difficulty"]
        val_diff = get_val_difficulty_filter(train_diff) if cfg["val_on_rest"] else None
        train_ds = PHLOPTrainDataset(train_video_ds, difficulty_filter=train_diff)
        if len(train_ds) == 0:
            print(f"Skipping config {cfg_name}: no training samples.")
            continue
        if cfg["val_on_rest"] and val_video_ds is not None:
            val_ds = PHLOPTrainDataset(val_video_ds, difficulty_filter=val_diff)
        else:
            val_ds = PHLOPValDataset(val_video_ds, camera_mode=camera_mode)
        model = AutoModelForImageTextToText.from_pretrained(
            SMOLVLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
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
            evaluation_strategy="steps",
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
    return saved_dirs


def run_test_comparison_smolvlm(
    splits: PHLOPSplits,
    model_checkpoints: list[tuple[str, str]],
    camera_mode: str = "static",
    results_dir: Optional[str] = "results",
) -> list[tuple[str, dict]]:
    """Evaluate each (model_name, checkpoint_path) on test split. Returns list of (model_name, metrics).
    When results_dir is set, saves per-model raw predictions to JSON."""
    if "test" not in splits:
        print("Need test split for comparison.")
        return []
    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    test_ds = PHLOPValDataset(splits["test"], camera_mode=camera_mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    print("\n--- Test comparison (model vs dataset) ---")
    table = []
    for model_name, ckpt in model_checkpoints:
        model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
        model = model.to(device)
        results = collect_predictions_smolvlm(model, processor, test_ds, device=device)
        metrics = compute_metrics(results)
        table.append((model_name, metrics))
        print(f"  {model_name}: answer_acc={metrics['answer_accuracy']:.4f}, physics_acc={metrics['physics_signal_accuracy']:.4f}, tax_f1={metrics['taxonomy_f1']:.4f}")
        if results_dir:
            import json
            out_path = os.path.join(results_dir, f"smolvlm_predictions_{model_name}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"    Saved {len(results)} predictions to {out_path}")
        del model
        torch.cuda.empty_cache()
    print(f"\n{'Model':<20} {'AnswerAcc':>10} {'PhysicsAcc':>10} {'TaxonomyF1':>10}")
    for name, m in table:
        print(f"{name:<20} {m['answer_accuracy']:>10.4f} {m['physics_signal_accuracy']:>10.4f} {m['taxonomy_f1']:>10.4f}")
    return table
