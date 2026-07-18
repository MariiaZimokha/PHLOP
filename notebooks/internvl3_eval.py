"""
InternVL3-2B-specific PHLOP eval: zero-shot, fine-tuning, test comparison.
Works with PHLOPDataset (path-based) returned by load_phlop_splits().

Notebook usage:
    from phlop_eval_common import load_phlop_splits, FINE_TUNE_CONFIGS
    from internvl3_eval import finetune_single_config_internvl3

    splits = load_phlop_splits("zimmari-ai/phlop", token=True)
    OUTPUT_DIR = "./out"
    MAX_STEPS = 50

    for cfg_name in FINE_TUNE_CONFIGS:
        for use_physics in [False, True]:
            ckpt = finetune_single_config_internvl3(
                splits,
                cfg_name=cfg_name,
                output_dir=OUTPUT_DIR,
                max_steps=MAX_STEPS,
                camera_mode="static",
                use_physics=use_physics,
            )
"""
from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from decord import VideoReader, cpu

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


INTERNVL3_MODEL_ID = "OpenGVLab/InternVL3-2B"
NUM_VIDEO_FRAMES = 8
INFERENCE_BATCH_SIZE = 1
INPUT_SIZE = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _best_attn_implementation() -> str:
    # InternVLChatModel supports neither flash_attention_2 nor sdpa;
    # "eager" is the only valid backend for this architecture.
    return "eager"


def _load_internvl3_config(attn_impl: str | None = None):
    """Load the InternVL3 config and propagate attn_implementation to all
    sub-configs (vision, LLM) so inner models don't try to use flash-attn."""
    from transformers import AutoConfig

    if attn_impl is None:
        attn_impl = _best_attn_implementation()
    cfg = AutoConfig.from_pretrained(INTERNVL3_MODEL_ID, trust_remote_code=True)
    cfg._attn_implementation = attn_impl
    for sub in ("llm_config", "vision_config"):
        sub_cfg = getattr(cfg, sub, None)
        if sub_cfg is not None:
            sub_cfg._attn_implementation = attn_impl
    return cfg


def _build_transform(input_size: int = INPUT_SIZE) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_video_frames(
    video_path: str,
    num_frames: int = NUM_VIDEO_FRAMES,
    input_size: int = INPUT_SIZE,
) -> Optional[torch.Tensor]:
    """Load video and return preprocessed pixel_values (N, 3, H, W)."""
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        if total == 0:
            return None
        n = min(num_frames, total)
        seg_size = total / n
        indices = [int(seg_size / 2 + seg_size * i) for i in range(n)]
        frames = vr.get_batch(indices).asnumpy()

        transform = _build_transform(input_size)
        tensors = [transform(Image.fromarray(f)) for f in frames]
        return torch.stack(tensors)
    except Exception as e:
        print(f"  Warning: failed to load video {video_path}: {e}")
        return None


def _get_num_image_tokens(model) -> int:
    """Get the number of IMG_CONTEXT tokens per image from the model."""
    for attr in ("num_image_token", "num_image_tokens"):
        val = getattr(model, attr, None) or getattr(getattr(model, "config", None), attr, None)
        if val:
            return int(val)
    return 256


def _build_prompt(question: str, physical_props: dict = None) -> str:
    """Build prompt text (without <image> tokens — those are added separately)."""
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
        f"Question: {question}\n\n"
        "Provide ONLY the final answer with no explanation.\n\n"
        "Answer:"
    )


def _make_question_with_images(prompt: str, num_frames: int) -> str:
    """Prepend Frame-labelled <image> tokens to the prompt for InternVL3."""
    image_tokens = "".join(f"Frame{i + 1}: <image>\n" for i in range(num_frames))
    return image_tokens + prompt


def _expand_image_tags(text: str, num_image_tokens: int) -> str:
    """Replace each <image> tag with the expanded IMG_CONTEXT sequence."""
    expanded = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
    return text.replace("<image>", expanded)


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class PHLOPTrainDataset(Dataset):
    """Training dataset with eager preloading — no lazy I/O in __getitem__
    except video frame decoding (which runs in dataloader workers)."""

    def __init__(
        self,
        base_dataset,
        difficulty_filter: Optional[list[str]] = None,
        camera_mode: str = "static",
        num_frames: int = NUM_VIDEO_FRAMES,
        use_physics: bool = False,
    ):
        self.camera_mode = camera_mode
        self.num_frames = num_frames
        self.use_physics = use_physics
        print(f"  Building training index (camera={camera_mode}, difficulty={difficulty_filter})...")
        self.index = build_training_index(
            base_dataset, difficulty_filter=difficulty_filter, camera_mode=camera_mode,
        )
        print(f"  Training index: {len(self.index)} (scene, question) pairs")

        self._scene_cache = preload_scenes(
            base_dataset, self.index, camera_mode, use_physics,
        )

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

        pixel_values = _load_video_frames(cached["video_path"], self.num_frames)

        return {
            "pixel_values": pixel_values,
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
        num_frames: int = NUM_VIDEO_FRAMES,
        use_physics: bool = False,
    ):
        self.camera_mode = camera_mode
        self.num_frames = num_frames
        self.use_physics = use_physics
        print(f"  Building eval index (camera={camera_mode})...")
        self.index = build_training_index(
            base_dataset, difficulty_filter=None, camera_mode=camera_mode,
        )
        print(f"  Eval index: {len(self.index)} (scene, question) pairs from {len(base_dataset)} scenes")

        self._scene_cache = preload_scenes(
            base_dataset, self.index, camera_mode, use_physics,
        )

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

        pixel_values = _load_video_frames(cached["video_path"], self.num_frames)

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "target": target,
            "metadata": {},
            "qa_entry": qa_entry,
            "camera_mode": self.camera_mode,
            "video_path": cached["video_path"],
            "scene_idx": scene_idx,
            "qa_idx": qa_idx,
        }


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

FLUSH_BATCH_SIZE = 500


class _SafeInferenceDataset(Dataset):
    """Wrapper that catches errors in __getitem__ so DataLoader workers don't crash."""

    def __init__(self, base, max_samples=None):
        self.base = base
        self.n = len(base) if max_samples is None else min(max_samples, len(base))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        try:
            item = self.base[idx]
            if item.get("pixel_values") is None:
                return {"_skip": True, "_idx": idx}
            item["_idx"] = idx
            return item
        except (ValueError, KeyError, FileNotFoundError):
            return {"_skip": True, "_idx": idx}


def _inference_collate_fn(batch):
    """Pass items through as a list — no tensor stacking."""
    return batch


def collect_predictions_internvl3(
    model,
    tokenizer,
    dataset,
    max_samples=None,
    device=None,
    num_frames: int = NUM_VIDEO_FRAMES,
    max_new_tokens: int = 200,
    num_workers: int = 4,
):
    """Run inference with DataLoader prefetching (InternVL3 chat format)."""
    model.eval()
    if device is None:
        device = str(next(model.parameters()).device)
    device_str = str(device)
    model_dtype = _best_dtype(device_str)

    safe_ds = _SafeInferenceDataset(dataset, max_samples)
    loader = torch.utils.data.DataLoader(
        safe_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_inference_collate_fn,
        pin_memory=False,
    )

    tmp_dir = tempfile.mkdtemp(prefix="internvl3_batches_")
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

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.1,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    for sample_num, batch in enumerate(tqdm(loader, desc="Collecting predictions", mininterval=5)):
        item = batch[0]
        if item.get("_skip"):
            continue

        i = item["_idx"]
        pixel_values = item.get("pixel_values")
        pixel_values = pixel_values.to(device=device, dtype=model_dtype)
        actual_frames = pixel_values.shape[0]
        question = _make_question_with_images(item["prompt"], actual_frames)

        try:
            with torch.no_grad():
                result = model.chat(
                    tokenizer, pixel_values, question=question,
                    generation_config=gen_config, history=None,
                )
            response = result[0] if isinstance(result, tuple) else result
            response = clean_prediction(str(response))
        except RuntimeError as e:
            print(f"  Error at idx {i}: {e}")
            response = "Error"

        del pixel_values
        if sample_num % 50 == 0:
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
            "prediction": str(response),
            "target": item["target"],
            "taxonomy_labels": tax_labels,
            "physics_signals": qa.get("physics_signals", {}),
        })

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


# ---------------------------------------------------------------------------
# Training collator and trainer
# ---------------------------------------------------------------------------

class InternVL3DataCollator:
    """Collator for LoRA fine-tuning. Batch size is expected to be 1."""

    def __init__(self, tokenizer, num_image_tokens: int = 256, num_frames: int = NUM_VIDEO_FRAMES):
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.num_frames = num_frames

    def __call__(self, batch):
        item = batch[0]

        pv = item.get("pixel_values")
        if pv is None:
            pv = torch.zeros((self.num_frames, 3, INPUT_SIZE, INPUT_SIZE))
        actual_frames = pv.shape[0]

        question = _make_question_with_images(item["prompt"], actual_frames)
        question_expanded = _expand_image_tags(question, self.num_image_tokens)
        full_text = question_expanded + item["target"]

        encodings = self.tokenizer(
            full_text, return_tensors="pt",
            padding=True, truncation=True, max_length=8192,
        )

        labels = encodings["input_ids"].clone()
        prompt_enc = self.tokenizer(
            question_expanded, add_special_tokens=True, return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[0, :prompt_len] = -100
        labels[encodings["attention_mask"] == 0] = -100
        encodings["labels"] = labels

        encodings["pixel_values"] = pv.to(torch.bfloat16)
        encodings["image_flags"] = torch.ones(actual_frames, 1, dtype=torch.long)

        return encodings


_VISUAL_STASH: dict = {}


class InternVL3Trainer(Trainer):
    _INTERNVL_FORWARD_KEYS = {
        "pixel_values", "input_ids", "attention_mask", "position_ids",
        "image_flags", "past_key_values", "labels", "use_cache",
        "output_attentions", "output_hidden_states", "return_dict",
    }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v for k, v in inputs.items() if k in self._INTERNVL_FORWARD_KEYS}

        # Stash visual inputs in a module-level dict so _compat_fwd can
        # re-inject them after PEFT strips unknown kwargs.  We can't stash
        # on `model` because accelerate wraps it in a different object.
        _VISUAL_STASH.update({
            k: inputs[k] for k in ("pixel_values", "image_flags") if k in inputs
        })

        outputs = model(**inputs)
        _VISUAL_STASH.clear()

        return (outputs.loss, outputs) if return_outputs else outputs.loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_zero_shot_internvl3(
    splits: PHLOPSplits,
    camera_mode: str = "static",
    max_samples: Optional[int] = None,
    model=None,
    tokenizer=None,
    device: Optional[str] = None,
    results_dir: Optional[str] = "results",
    eval_splits: Optional[list[str]] = None,
    compile_model: bool = False,
    num_frames: int = NUM_VIDEO_FRAMES,
    use_physics: bool = False,
    max_new_tokens: int = 200,
) -> dict[str, dict]:
    eval_splits = eval_splits or ["test"]
    physics_tag = "with_physics" if use_physics else "no_physics"
    available = [s for s in eval_splits if s in splits]
    if not available:
        print(f"None of {eval_splits} found in splits.")
        return {}

    if device is None:
        device = _get_device()
    dtype = _best_dtype(device)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(INTERNVL3_MODEL_ID, trust_remote_code=True)
    if model is None:
        model = AutoModel.from_pretrained(
            INTERNVL3_MODEL_ID,
            config=_load_internvl3_config(),
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            _fast_init=False,
            attn_implementation=_best_attn_implementation(),
        ).to(device).eval()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    print(f"Using device: {device} (dtype={dtype}, frames={num_frames}, physics={use_physics})")

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
            num_frames=num_frames, use_physics=use_physics,
        )
        n_eval = len(val_ds) if max_samples is None else min(max_samples, len(val_ds))
        print(f"\n--- Zero-shot on {split_name} ({camera_mode}, {physics_tag}): "
              f"{len(val_ds)} questions (evaluating {n_eval}) ---")
        results = collect_predictions_internvl3(
            model, tokenizer, val_ds,
            max_samples=max_samples, device=device,
            num_frames=num_frames, max_new_tokens=max_new_tokens,
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
                f"internvl3_zero_shot_predictions_{split_name}_{camera_mode}_{physics_tag}.json",
            )
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved {len(results)} predictions to {pred_path}")

    return out


def finetune_single_config_internvl3(
    splits: PHLOPSplits,
    cfg_name: str,
    output_dir: str = "./internvl3_checkpoints",
    max_steps: int = -1,
    num_epochs: int = 1,
    camera_mode: str = "static",
    use_physics: bool = False,
    num_frames: int = NUM_VIDEO_FRAMES,
    early_stopping_patience: int = 3,
    eval_steps: int = 100,
) -> Optional[str]:
    """Fine-tune InternVL3-2B on a single difficulty config. Returns checkpoint dir or None.

    Training length is controlled by either max_steps or num_epochs:
      - max_steps > 0: train for exactly that many steps (num_epochs is ignored).
      - max_steps == -1 (default): train for num_epochs full passes over the data;
        early stopping will halt training when eval loss stops improving.

    Args:
        eval_steps: Evaluate every N optimizer steps (default 100). Lower values
                    let early stopping react faster but add eval overhead.
    """
    if "train" not in splits:
        print("Need train split for fine-tuning.")
        return None

    physics_tag = "with_physics" if use_physics else "no_physics"
    cfg = FINE_TUNE_CONFIGS[cfg_name]

    print(f"\n{'=' * 60}")
    print(f"Fine-tuning config: {cfg_name} ({physics_tag})")

    device = _get_device()
    dtype = _best_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(INTERNVL3_MODEL_ID, trust_remote_code=True)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_backend = splits["train"]
    val_backend = splits.get("validation", train_backend)

    train_diff = cfg["train_difficulty"]
    val_diff = get_val_difficulty_filter(train_diff) if cfg["val_on_rest"] else None
    print(f"  Train difficulties: {train_diff}, Val difficulties: {val_diff}")

    train_ds = PHLOPTrainDataset(
        train_backend, difficulty_filter=train_diff,
        camera_mode=camera_mode, num_frames=num_frames, use_physics=use_physics,
    )
    if len(train_ds) == 0:
        print(f"  Skipping config {cfg_name}: no training samples.")
        return None

    if cfg["val_on_rest"] and val_backend is not None:
        val_ds = PHLOPTrainDataset(
            val_backend, difficulty_filter=val_diff,
            camera_mode=camera_mode, num_frames=num_frames, use_physics=use_physics,
        )
    else:
        val_ds = PHLOPValDataset(
            val_backend, camera_mode=camera_mode,
            num_frames=num_frames, use_physics=use_physics,
        )

    internvl_cfg = _load_internvl3_config()
    attn_impl = internvl_cfg._attn_implementation
    print(f"  Attention implementation: {attn_impl}")
    model = AutoModel.from_pretrained(
        INTERNVL3_MODEL_ID,
        config=internvl_cfg,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=False,
        _fast_init=False,
        attn_implementation=attn_impl,
    ).to(device)

    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_ctx_id
    model.config.img_context_token_id = img_ctx_id
    model.config.image_token_id = img_ctx_id

    num_image_tokens = _get_num_image_tokens(model)
    print(f"  Image tokens per frame: {num_image_tokens}")

    data_collator = InternVL3DataCollator(
        tokenizer, num_image_tokens=num_image_tokens, num_frames=num_frames,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # PEFT's PeftModelForCausalLM.forward() always passes `inputs_embeds`
    # to the base model, but InternVLChatModel.forward() doesn't accept it.
    _orig_fwd = model.base_model.model.forward

    def _compat_fwd(*args, **kwargs):
        kwargs.pop("inputs_embeds", None)
        kwargs.update(_VISUAL_STASH)
        return _orig_fwd(*args, **kwargs)

    model.base_model.model.forward = _compat_fwd

    ckpt_dir = os.path.join(output_dir, f"internvl3_{cfg_name}_{physics_tag}")
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = (device == "mps") or (device == "cuda" and not use_bf16)

    use_steps = max_steps > 0
    eval_every = eval_steps if eval_steps > 0 else 100

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
        gradient_accumulation_steps=16,
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
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
    )

    trainer = InternVL3Trainer(
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

    del model
    _empty_device_cache(device)

    return ckpt_dir


def run_test_comparison_internvl3(
    splits: PHLOPSplits,
    model_checkpoints: list[tuple[str, str]],
    camera_mode: str = "static",
    results_dir: Optional[str] = "results",
    eval_splits: Optional[list[str]] = None,
    use_physics: bool = False,
    num_frames: int = NUM_VIDEO_FRAMES,
    max_new_tokens: int = 200,
) -> list[tuple[str, dict]]:
    eval_splits = eval_splits or ["test"]
    physics_tag = "with_physics" if use_physics else "no_physics"
    available = [s for s in eval_splits if s in splits]
    if not available:
        print(f"None of {eval_splits} found in splits.")
        return []

    tokenizer = AutoTokenizer.from_pretrained(INTERNVL3_MODEL_ID, trust_remote_code=True)
    device = _get_device()
    dtype = _best_dtype(device)
    attn_impl = _best_attn_implementation()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    base_model = AutoModel.from_pretrained(
        INTERNVL3_MODEL_ID,
        config=_load_internvl3_config(attn_impl),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        _fast_init=False,
        attn_implementation=attn_impl,
    ).to(device)

    table = []
    for model_name, ckpt in model_checkpoints:
        print(f"\n--- {model_name} ({camera_mode}, {physics_tag}) ---")

        if ckpt == INTERNVL3_MODEL_ID:
            model = base_model
        else:
            model = PeftModel.from_pretrained(base_model, ckpt)

        model.eval()

        model_metrics = {}
        for split_name in available:
            ds = PHLOPValDataset(
                splits[split_name], camera_mode=camera_mode,
                num_frames=num_frames, use_physics=use_physics,
            )
            results = collect_predictions_internvl3(
                model, tokenizer, ds, device=device,
                num_frames=num_frames, max_new_tokens=max_new_tokens,
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
                    f"internvl3_finetuned_predictions_{model_name}_{split_name}_{camera_mode}_{physics_tag}.json",
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
