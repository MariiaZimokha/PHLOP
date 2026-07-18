"""
Qwen2-VL-specific PHLOP eval: zero-shot, fine-tuning, test comparison.
Works with PHLOPDataset (path-based) returned by load_phlop_splits().

Notebook usage:
    from phlop_eval_common import load_phlop_splits, FINE_TUNE_CONFIGS
    from qwen2vl_eval import finetune_single_config_qwen2vl

    splits = load_phlop_splits("zimmari-ai/phlop", token=True)
    OUTPUT_DIR = "./out"
    MAX_STEPS = 50

    for cfg_name in FINE_TUNE_CONFIGS:
        for use_physics in [False, True]:
            ckpt = finetune_single_config_qwen2vl(
                splits,
                cfg_name=cfg_name,
                output_dir=OUTPUT_DIR,
                max_steps=MAX_STEPS,
                camera_mode="static",
                use_physics=use_physics,
            )
"""
from __future__ import annotations

import collections
import gc
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from phlop_eval_common import (
    clean_prediction,
    compute_metrics,
    print_metrics,
    load_qa_from_path,
    load_json_file,
    get_physical_props,
    flatten_taxonomy,
    build_training_index,
    preload_scenes,
    FINE_TUNE_CONFIGS,
    get_val_difficulty_filter,
    PHLOPSplits,
)


QWEN2_VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VIDEO_FPS = 2.0
INFERENCE_BATCH_SIZE = 1


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _best_dtype(device: str) -> torch.dtype:
    if device == "mps":
        return torch.float16
    return torch.bfloat16


def _empty_device_cache(device: str) -> None:
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.empty_cache()
    elif device == "mps" or (isinstance(device, torch.device) and device.type == "mps"):
        torch.mps.empty_cache()


def _build_prompt(question: str, physical_props: dict = None) -> str:
    """Build a text prompt for the question, optionally including physics properties."""
    props_section = ""
    if physical_props:
        props_lines = "\n".join(
            f"- {v['color']} {v['shape']}: mass={v['mass']}, friction={v['friction'][0]:.2f}"
            for v in physical_props.values()
        )
        props_section = f"\nObject properties:\n{props_lines}\n"

    return (
        "You are a physics reasoning system.\n"
        "Watch the video and answer the question.\n\n"
        f"{props_section}\n"
        f"Question:\n{question}\n\n"
        "Respond with ONLY the final answer:\n"
        '- yes/no questions: answer "yes" or "no"\n'
        "- counting questions: answer with a number\n"
        '- time questions: answer like "0.5s"\n'
        "- multiple choice: pick exactly one option\n"
        "Provide ONLY the final answer with no explanation.\n\n"
        "Answer:\n"
    )


def _build_conversation(
    video_path: str,
    prompt: str,
    fps: float = VIDEO_FPS,
    answer: str = None,
) -> list[dict]:
    """Build a Qwen2-VL conversation list with capped resolution."""
    conv = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "fps": fps,
                    "min_pixels": 128 * 128,
                    "max_pixels": 256 * 256,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        conv.append({"role": "assistant", "content": answer})
    return conv


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class PHLOPTrainDataset(Dataset):
    """Training dataset with eager preloading — no lazy I/O in __getitem__.
    Video is loaded by the processor inside the collator (runs in dataloader workers)."""

    def __init__(
        self,
        base_dataset,
        difficulty_filter: Optional[list[str]] = None,
        camera_mode: str = "static",
        use_physics: bool = False,
    ):
        self.camera_mode = camera_mode
        self.use_physics = use_physics
        print(f"  Building training index (camera={camera_mode}, difficulty={difficulty_filter})...")
        raw_index = build_training_index(
            base_dataset, difficulty_filter=difficulty_filter, camera_mode=camera_mode,
        )
        print(f"  Raw training index: {len(raw_index)} (scene, question) pairs")

        self._scene_cache = preload_scenes(
            base_dataset, raw_index, camera_mode, use_physics,
        )

        self.index = self._filter_valid_qa(raw_index)
        dropped = len(raw_index) - len(self.index)
        if dropped:
            print(f"  Filtered training index: {len(self.index)} valid QA pairs (dropped {dropped})")

    def _filter_valid_qa(self, raw_index):
        """Keep only QA entries that have a non-empty answer field."""
        valid = []
        for scene_idx, qa_idx in raw_index:
            cached = self._scene_cache.get(scene_idx)
            if not cached:
                continue
            qa_list = cached["qa_list"]
            if qa_idx >= len(qa_list):
                continue
            qa_entry = qa_list[qa_idx]
            if not qa_entry.get("answer") and qa_entry.get("answer") != 0:
                continue
            valid.append((scene_idx, qa_idx))
        return valid

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene_idx, qa_idx = self.index[idx]
        cached = self._scene_cache[scene_idx]

        qa_list = cached["qa_list"]
        qa_entry = qa_list[qa_idx] if qa_idx < len(qa_list) else {}

        question = qa_entry.get("question", "")
        answer = qa_entry.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        if qa_entry.get("options"):
            opts = "\n".join(f"- {o}" for o in qa_entry["options"])
            question = f"{question}\nOptions:\n{opts}"

        prompt = _build_prompt(question, cached["physics"])
        target = str(answer)

        return {
            "video_path": cached["video_path"],
            "prompt": prompt,
            "target": target,
            "metadata": {},
            "qa_entry": qa_entry,
        }


class PHLOPValDataset(Dataset):
    """Eval dataset with eager preloading — no lazy I/O in __getitem__."""

    def __init__(
        self,
        base_dataset,
        camera_mode: str = "static",
        use_physics: bool = False,
    ):
        self.camera_mode = camera_mode
        self.use_physics = use_physics
        print(f"  Building eval index (camera={camera_mode})...")
        raw_index = build_training_index(
            base_dataset, difficulty_filter=None, camera_mode=camera_mode,
        )
        print(f"  Raw eval index: {len(raw_index)} (scene, question) pairs from {len(base_dataset)} scenes")

        self._scene_cache = preload_scenes(
            base_dataset, raw_index, camera_mode, use_physics,
        )

        self.index = self._filter_valid_qa(raw_index)
        print(f"  Filtered eval index: {len(self.index)} valid QA pairs "
              f"(dropped {len(raw_index) - len(self.index)} entries without answer)")

    def _filter_valid_qa(self, raw_index):
        """Keep only QA entries that have a non-empty answer field."""
        valid = []
        for scene_idx, qa_idx in raw_index:
            cached = self._scene_cache.get(scene_idx)
            if not cached:
                continue
            qa_list = cached["qa_list"]
            if qa_idx >= len(qa_list):
                continue
            qa_entry = qa_list[qa_idx]
            if not qa_entry.get("answer") and qa_entry.get("answer") != 0:
                continue
            valid.append((scene_idx, qa_idx))
        return valid

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene_idx, qa_idx = self.index[idx]
        cached = self._scene_cache[scene_idx]

        qa_list = cached["qa_list"]
        qa_entry = qa_list[qa_idx] if qa_idx < len(qa_list) else {}

        question = qa_entry.get("question", "")
        answer = qa_entry.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        if qa_entry.get("options"):
            opts = "\n".join(f"- {o}" for o in qa_entry["options"])
            question = f"{question}\nOptions:\n{opts}"

        prompt = _build_prompt(question, cached["physics"])
        target = str(answer)

        return {
            "video_path": cached["video_path"],
            "prompt": prompt,
            "target": target,
            "metadata": {},
            "qa_entry": qa_entry,
            "camera_mode": self.camera_mode,
            "scene_idx": scene_idx,
            "qa_idx": qa_idx,
        }


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

FLUSH_BATCH_SIZE = 500


def collect_predictions_qwen2vl(
    model,
    processor,
    dataset,
    max_samples=None,
    device=None,
    fps: float = VIDEO_FPS,
    max_new_tokens: int = 200,
    prefetch_workers: int = 4,
):
    """Run inference with ThreadPoolExecutor prefetching for video decode overlap."""
    model.eval()
    if device is None:
        device = str(next(model.parameters()).device)
    device_str = str(device)
    model_dtype = _best_dtype(device_str)
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    tmp_dir = tempfile.mkdtemp(prefix="qwen2vl_batches_")
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

    _prepare_errors = collections.Counter()

    def _prepare_item(idx):
        """Pre-load dataset item and run processor (video decode) in a thread."""
        try:
            item = dataset[idx]
        except (ValueError, KeyError, FileNotFoundError) as e:
            _prepare_errors["dataset_load"] += 1
            if _prepare_errors["dataset_load"] <= 3:
                print(f"  [prepare {idx}] dataset load error: {e}")
            return idx, None
        video_path = item.get("video_path", "")
        if not video_path or not os.path.exists(video_path):
            _prepare_errors["missing_video"] += 1
            if _prepare_errors["missing_video"] <= 3:
                print(f"  [prepare {idx}] missing video: {video_path!r}")
            return idx, None
        try:
            conv = _build_conversation(video_path, item["prompt"], fps=fps)
            texts = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(conv)
            inputs = processor(
                text=[texts],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception as e:
            _prepare_errors["process"] += 1
            if _prepare_errors["process"] <= 5:
                print(f"  [prepare {idx}] processing error ({type(e).__name__}): {e}")
            return idx, None
        return idx, (item, inputs)

    prefetch_size = max(prefetch_workers * 2, 8)

    with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        futures = collections.deque()
        for i in range(min(prefetch_size, n)):
            futures.append(pool.submit(_prepare_item, i))
        next_submit = prefetch_size

        for _ in tqdm(range(n), desc="Collecting predictions", mininterval=5):
            future = futures.popleft()
            if next_submit < n:
                futures.append(pool.submit(_prepare_item, next_submit))
                next_submit += 1

            i, result = future.result()
            if result is None:
                continue

            item, inputs = result
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values_videos" in inputs:
                inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(model_dtype)

            with torch.no_grad():
                input_len = inputs["input_ids"].shape[1]
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.1,
                )
                generated_only = output_ids[:, input_len:]
            raw_pred = processor.batch_decode(generated_only, skip_special_tokens=True)[0]
            pred = clean_prediction(raw_pred)

            del inputs, output_ids, generated_only
            if i % 50 == 0:
                _empty_device_cache(device_str)

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

            if len(buffer) >= FLUSH_BATCH_SIZE:
                _flush()

    _flush()

    if _prepare_errors:
        total_skipped = sum(_prepare_errors.values())
        print(f"\n  Prepare errors summary ({total_skipped}/{n} items skipped):")
        for reason, count in _prepare_errors.most_common():
            print(f"    {reason}: {count}")

    results: list[dict] = []
    for bp in batch_files:
        with open(bp, "r") as f:
            results.extend(json.load(f))
        os.remove(bp)
    os.rmdir(tmp_dir)

    return results


# ---------------------------------------------------------------------------
# Training collator and trainer
# ---------------------------------------------------------------------------

class Qwen2VLDataCollator:
    """Collator that builds Qwen2-VL conversations with video for training."""

    def __init__(self, processor, fps: float = VIDEO_FPS):
        self.processor = processor
        self.fps = fps

    def __call__(self, batch):
        valid_convs = []

        for item in batch:
            conv = _build_conversation(
                item["video_path"], item["prompt"],
                fps=self.fps, answer=item["target"]
            )

            try:
                _ = process_vision_info([conv])
                valid_convs.append(conv)
            except Exception as e:
                print(f"\n  [Warning] Skipping corrupted video ({item.get('video_path')}): {e}")
                fallback_conv = [
                    {"role": "user", "content": [{"type": "text", "text": item["prompt"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]}
                ]
                valid_convs.append(fallback_conv)

        texts = [
            self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in valid_convs
        ]

        image_inputs, video_inputs = process_vision_info(valid_convs)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


class Qwen2VLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_zero_shot_qwen2vl(
    splits: PHLOPSplits,
    camera_mode: str = "static",
    max_samples: Optional[int] = None,
    model=None,
    processor=None,
    device: Optional[str] = None,
    results_dir: Optional[str] = "results",
    eval_splits: Optional[list[str]] = None,
    compile_model: bool = False,
    fps: float = VIDEO_FPS,
    use_physics: bool = False,
    max_new_tokens: int = 200,
) -> dict[str, dict]:
    eval_splits = eval_splits or ["test"]
    physics_tag = "physics" if use_physics else "no_physics"
    available = [s for s in eval_splits if s in splits]
    if not available:
        print(f"None of {eval_splits} found in splits.")
        return {}

    if device is None:
        device = _get_device()
    dtype = _best_dtype(device)

    if processor is None:
        processor = AutoProcessor.from_pretrained(QWEN2_VL_MODEL_ID)
    if model is None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN2_VL_MODEL_ID,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    print(f"Using device: {device} (dtype={dtype}, fps={fps}, physics={use_physics})")

    if compile_model:
        try:
            model = torch.compile(model)
            print("  torch.compile() applied")
        except Exception as e:
            print(f"  torch.compile() failed, continuing without: {e}")

    out = {}
    for split_name in available:
        val_ds = PHLOPValDataset(
            splits[split_name], camera_mode=camera_mode,
            use_physics=use_physics,
        )
        n_eval = len(val_ds) if max_samples is None else min(max_samples, len(val_ds))
        print(f"\n--- Zero-shot on {split_name} ({camera_mode}, {physics_tag}): "
              f"{len(val_ds)} questions (evaluating {n_eval}) ---")
        results = collect_predictions_qwen2vl(
            model, processor, val_ds,
            max_samples=max_samples, device=device,
            fps=fps,
            max_new_tokens=max_new_tokens,
        )
        for r in results:
            r["split"] = split_name
            r["camera_mode"] = camera_mode
            r["use_physics"] = use_physics
        metrics = compute_metrics(results)
        print_metrics(metrics, f"Zero-shot {split_name} ({physics_tag})")
        out[split_name] = {"metrics": metrics, "n_questions": len(results)}

        if results_dir:
            pred_path = os.path.join(
                results_dir,
                f"qwen2vl_zero_shot_predictions_{split_name}_{camera_mode}_{physics_tag}.json",
            )
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved {len(results)} predictions to {pred_path}")

    return out


def finetune_single_config_qwen2vl(
    splits: PHLOPSplits,
    cfg_name: str,
    output_dir: str = "./qwen2vl_physics_full",
    max_steps: int = -1,
    num_epochs: int = 1,
    camera_mode: str = "static",
    use_physics: bool = False,
    fps: float = VIDEO_FPS,
    early_stopping_patience: int = 3,
    eval_steps: int = 25,
    base_model=None,
    processor=None,
) -> Optional[str]:
    """Fine-tune Qwen2-VL on a single difficulty config. Returns checkpoint dir or None.

    Training length is controlled by either max_steps or num_epochs:
      - max_steps > 0: train for exactly that many steps (num_epochs is ignored).
      - max_steps == -1 (default): train for num_epochs full passes over the data;
        early stopping will halt training when eval loss stops improving.

    Args:
        eval_steps: Evaluate every N optimizer steps (default 25). Lower values
                    let early stopping react faster but add eval overhead.
        base_model: Pre-loaded base model to reuse across configs.
                    If None, loads from pretrained (slower).
        processor: Pre-loaded processor. If None, loads from pretrained.
    """
    if "train" not in splits:
        print("Need train split for fine-tuning.")
        return None

    physics_tag = "physics" if use_physics else "no_physics"
    cfg = FINE_TUNE_CONFIGS[cfg_name]

    print(f"\n{'=' * 60}")
    print(f"Fine-tuning config: {cfg_name} ({physics_tag})")

    device = _get_device()
    dtype = _best_dtype(device)

    owns_model = base_model is None
    if processor is None:
        processor = AutoProcessor.from_pretrained(QWEN2_VL_MODEL_ID)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    data_collator = Qwen2VLDataCollator(processor, fps=fps)

    train_backend = splits["train"]
    val_backend = splits.get("validation", train_backend)

    train_diff = cfg["train_difficulty"]
    val_diff = get_val_difficulty_filter(train_diff) if cfg["val_on_rest"] else None
    print(f"  Train difficulties: {train_diff}, Val difficulties: {val_diff}")

    train_ds = PHLOPTrainDataset(
        train_backend, difficulty_filter=train_diff,
        camera_mode=camera_mode, use_physics=use_physics,
    )
    if len(train_ds) == 0:
        print(f"  Skipping config {cfg_name}: no training samples.")
        return None

    if cfg["val_on_rest"] and val_backend is not None:
        val_ds = PHLOPTrainDataset(
            val_backend, difficulty_filter=val_diff,
            camera_mode=camera_mode, use_physics=use_physics,
        )
    else:
        val_ds = PHLOPValDataset(
            val_backend, camera_mode=camera_mode,
            use_physics=use_physics,
        )

    if base_model is None:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN2_VL_MODEL_ID,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )

    base_model.config.use_cache = False

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    ckpt_dir = os.path.join(output_dir, f"qwen2vl_{cfg_name}_{physics_tag}")
    use_bf16 = (device != "mps")
    use_fp16 = (device == "mps")

    use_steps = max_steps > 0
    eval_every = eval_steps if eval_steps > 0 else 25
    if use_steps and eval_every > max_steps:
        eval_every = max(1, max_steps // 2)
        print(f"  Capped eval_steps to {eval_every} (was {eval_steps}, max_steps={max_steps})")

    if use_steps:
        warmup = max(1, max_steps // 10)
        print(f"  Training: {max_steps} steps, eval every {eval_every} steps, "
              f"early stopping patience={early_stopping_patience}")
    else:
        warmup = 0
        print(f"  Training: {num_epochs} epoch(s), eval every {eval_every} steps, "
              f"early stopping patience={early_stopping_patience}")

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        max_steps=max_steps if use_steps else -1,
        num_train_epochs=num_epochs if not use_steps else 1,
        lr_scheduler_type="cosine",
        warmup_steps=warmup if use_steps else 10,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_steps=eval_every,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=eval_every,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch_fused",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    trainer = Qwen2VLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(ckpt_dir)
    print(f"  Model saved to: {ckpt_dir}")

    if not owns_model:
        model = model.unload()

    del model, trainer
    gc.collect()
    _empty_device_cache(device)

    if owns_model:
        del base_model
        gc.collect()
        _empty_device_cache(device)

    return ckpt_dir


def run_test_comparison_qwen2vl(
    splits: PHLOPSplits,
    model_checkpoints: list[tuple[str, str]],
    camera_mode: str = "static",
    results_dir: Optional[str] = "results",
    eval_splits: Optional[list[str]] = None,
    use_physics: bool = False,
    fps: float = VIDEO_FPS,
    max_new_tokens: int = 200,
) -> list[tuple[str, dict]]:
    eval_splits = eval_splits or ["test"]
    physics_tag = "physics" if use_physics else "no_physics"
    available = [s for s in eval_splits if s in splits]
    if not available:
        print(f"None of {eval_splits} found in splits.")
        return []

    processor = AutoProcessor.from_pretrained(QWEN2_VL_MODEL_ID)
    device = _get_device()
    dtype = _best_dtype(device)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN2_VL_MODEL_ID,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )

    table = []
    for model_name, ckpt in model_checkpoints:
        print(f"\n--- {model_name} ({camera_mode}, {physics_tag}) ---")

        if ckpt == QWEN2_VL_MODEL_ID:
            model = base_model
        else:
            model = PeftModel.from_pretrained(base_model, ckpt)

        model.eval()

        model_metrics = {}
        for split_name in available:
            ds = PHLOPValDataset(
                splits[split_name], camera_mode=camera_mode,
                use_physics=use_physics,
            )
            results = collect_predictions_qwen2vl(
                model, processor, ds, device=device,
                fps=fps,
                max_new_tokens=max_new_tokens,
            )
            for r in results:
                r["split"] = split_name
                r["camera_mode"] = camera_mode
                r["model"] = model_name
                r["use_physics"] = use_physics
            metrics = compute_metrics(results)
            model_metrics[split_name] = metrics
            print(f"  {split_name}: answer_acc={metrics['answer_accuracy']:.4f}")

            if results_dir:
                out_path = os.path.join(
                    results_dir,
                    f"qwen2vl_finetuned_predictions_{model_name}_{split_name}_{camera_mode}_{physics_tag}.json",
                )
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"    Saved {len(results)} predictions to {out_path}")

        table.append((model_name, model_metrics))
        if model is not base_model:
            del model
        _empty_device_cache(device)

    del base_model
    _empty_device_cache(device)

    return table
