import os
import json
import random
from collections import Counter

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from tqdm import tqdm

from utils import format_physics_signals, get_physics_summary, load_video, load_json


TAXONOMY_LABELS = [
    "Stationary",
    "Accelerating",
    "Decelerating",
    "Constant Velocity",
    "Pure Rotation",
    "Rolling Motion",
    "Rolling Motion with Slipping",
    "Moving to Stopping",
    "Stationary to Moving",
    "Elastic Collision",
    "Partially Inelastic Collision",
    "Highly Inelastic Collision",
    "Inelastic Collision",
    "Friction Stop",
    "Sliding with Friction",
]
TAXONOMY_LABELS = [label.lower() for label in TAXONOMY_LABELS]

labels_to_idx = {label: idx for idx, label in enumerate(TAXONOMY_LABELS)}
NUM_TAX = len(TAXONOMY_LABELS)


class PHLOPVideoDataset(Dataset):
    def __init__(
        self,
        hf_dataset_split,
        num_frames=8,
        split="train",
        root_dir="./",
        camera_mode="static",
    ):
        self.data = hf_dataset_split
        self.num_frames = num_frames
        self.split = split
        self.root_dir = root_dir
        self.camera_mode = camera_mode

    def get_qa_list(self, scene_idx):
        """Load only the QA JSON for a scene (no video). Used to build indices."""
        item = self.data[scene_idx]
        cam = item["camera_mode"] if self.split == "train" else self.camera_mode
        raw = load_json(os.path.join(self.root_dir, item["qa"][cam]))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return raw.get("questions", raw.get("qa", []))
        return []

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path):
        return load_video(video_path, self.root_dir, self.num_frames)

    def _get_taxonomy_targets(self, metadata):
        """Get one-hot encoded taxonomy targets for all objects in the scene."""
        target = torch.zeros(NUM_TAX)

        for frame in metadata.get("frames", []):
            for obj_id, obj_data in frame.get("objects", {}).items():
                # get labels if object is in a frame (bbox is not [[0, 0], [0, 0]])
                if obj_data.get("bbox", [[0, 0], [0, 0]]) != [[0, 0], [0, 0]]:  
                    for entry in obj_data.get("taxonomy", []):
                        for label in entry.get("labels", []):
                            label = label.lower()
                            if label in labels_to_idx:
                                target[labels_to_idx[label]] = 1.0
        return target

    def __getitem__(self, idx):
        item = self.data[idx]
        camera_mode = item["camera_mode"] if self.split == "train" else self.camera_mode

        id = item["id"]
        video_path = item["videos"][camera_mode]
        qa_path = item["qa"][camera_mode]
        meta_path = item["metadata"][camera_mode]
        metadata = load_json(os.path.join(self.root_dir, meta_path))
        return {
            "video": self._load_video(video_path),
            "qa": load_json(os.path.join(self.root_dir, qa_path)),
            "metadata": metadata,
            "physics_summary": get_physics_summary(metadata),
            "taxonomy_target": self._get_taxonomy_targets(metadata),
            "camera_mode": camera_mode,
            "id": item.get("id", idx),
        }


class PHLOPTrainDataset(PHLOPVideoDataset):
    """Expands each scene into one sample per (scene, question): uses all QAs per video."""

    def __init__(
        self,
        base_dataset=None,
        *,
        hf_dataset_split=None,
        num_frames=8,
        split="train",
        root_dir="./",
        camera_mode="static",
    ):
        if base_dataset is not None:
            super().__init__(
                base_dataset.data,
                num_frames=base_dataset.num_frames,
                split=base_dataset.split,
                root_dir=base_dataset.root_dir,
                camera_mode=base_dataset.camera_mode,
            )
        elif hf_dataset_split is not None:
            super().__init__(
                hf_dataset_split,
                num_frames=num_frames,
                split=split,
                root_dir=root_dir,
                camera_mode=camera_mode,
            )
        else:
            raise ValueError("Provide either base_dataset or hf_dataset_split")

        self.index = []
        for scene_idx in range(len(self.data)):
            qa_list = self.get_qa_list(scene_idx)
            for qa_idx in range(len(qa_list)):
                self.index.append((scene_idx, qa_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene_idx, qa_idx = self.index[idx]
        data = PHLOPVideoDataset.__getitem__(self, scene_idx)
        qa_list = data["qa"]
        if isinstance(qa_list, dict):
            qa_list = qa_list.get("questions", qa_list.get("qa", []))
        if not isinstance(qa_list, list):
            qa_list = []
        qa_entry = qa_list[qa_idx]

        video_frames = [
            Image.fromarray(f.astype("uint8"))
            for f in data["video"]
        ]

        question = qa_entry["question"]
        answer = qa_entry["answer"]
        physics_signals = qa_entry.get("physics_signals", {})

        if qa_entry.get("options"):
            opts = "\n".join(f"- {o}" for o in qa_entry["options"])
            question = f"{question}\nOptions:\n{opts}"

        physics_text = format_physics_signals(physics_signals)
        image_tokens = "\n".join(["<image>"] * len(video_frames))

        prompt = (
            "You are a physics reasoning system.\n\n"
            "Known physical setup:\n"
            f"{data['physics_summary']}\n\n"
            # "Observed physical events:\n"
            # f"{data['taxonomy_summary']}\n\n"
            f"{image_tokens}\n\n"
            f"Question:\n{question}\n\n"
            "Respond in the following format:\n"
            "Physics:\n"
            "- <key>: <value>\n"
            "Answer:\n"
        )

        target = (
            "Physics:\n"
            f"{physics_text}\n\n"
            "Answer:\n"
            f"{answer}"
        )

        out = {
            "video": video_frames,
            "prompt": prompt,
            "target": target,
            "taxonomy_target": data["taxonomy_target"],
            "metadata": data.get("metadata", {}),
            "qa_entry": qa_entry,
        }
        return out

local_repo = snapshot_download(
    repo_id="zimmari-ai/phlop",
    repo_type="dataset",
    local_dir="./phlop_data",
    local_dir_use_symlinks=False,
)
dataset = load_dataset(
    "zimmari-ai/phlop",
    data_files={
        "train": "data/train/**/*.parquet",
        "validation": "data/val/**/*.parquet",
    }
)

train_dataset = PHLOPTrainDataset(
    hf_dataset_split=dataset["train"],
    num_frames=8,
    split="train",
    root_dir=local_repo,
)

val_dataset = PHLOPTrainDataset(
    hf_dataset_split=dataset["validation"],
    num_frames=16,
    split="validation",
    root_dir=local_repo,
    camera_mode="static",
)



from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
)

base_model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    torch_dtype=torch.bfloat16,
    # device_map="auto"
)


# adding taxonomy head to the model

import torch.nn as nn
from transformers import PreTrainedModel



class SmolVLMWithTaxonomy(PreTrainedModel):
    base_model_prefix = "vlm"

    def __init__(self, vlm, num_tax):
        super().__init__(vlm.config)
        self.vlm = vlm
        
        vision_hidden_size = vlm.config.text_config.hidden_size
        
        self.taxonomy_head = nn.Sequential(
            nn.Linear(vision_hidden_size, vision_hidden_size),
            nn.ReLU(),
            nn.Linear(vision_hidden_size, num_tax),
        )

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    def generate(self, *args, **kwargs):
        return self.vlm.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vlm.prepare_inputs_for_generation(*args, **kwargs)


    def forward(self, pixel_values=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        taxonomy_target=None,
        **kwargs):

        outputs = self.vlm(pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            output_hidden_states=True,
                            output_attentions=False,
                            **kwargs
        )
        loss = outputs.loss

        if taxonomy_target is not None:
            # vision_hidden_states = outputs.vision_outputs.last_hidden_state
            if hasattr(outputs, "vision_outputs"):
                vision_hidden_states = outputs.vision_outputs.last_hidden_state
            else:
                vision_hidden_states = outputs.hidden_states[0]
            pooled_vision_hidden_states = vision_hidden_states.mean(dim=1)

            taxonomy_logits = self.taxonomy_head(pooled_vision_hidden_states)
            taxonomy_loss = nn.BCEWithLogitsLoss()(taxonomy_logits, taxonomy_target.to(taxonomy_logits.device))

            loss = loss + 0.2 * taxonomy_loss # weight for taxonomy loss to guide the model to learn the taxonomy

        outputs.loss = loss
        return outputs



class SmolVLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        videos = [x["video"] for x in batch]
        texts  = [x["prompt"] + x["target"] for x in batch]

        model_inputs = self.processor(
            videos=videos,
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        # Labels = input_ids (standard causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].clone()

        if "taxonomy_target" in batch[0]:
            model_inputs["taxonomy_target"] = torch.stack(
                [x["taxonomy_target"] for x in batch]
            )
        return model_inputs

data_collator = SmolVLMDataCollator(processor)


lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

base_model = get_peft_model(base_model, lora_config)
base_model.print_trainable_parameters()


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./smolvlm_physics",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=50,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=20,    
    remove_unused_columns=False,
    report_to="none",
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    # optim="adamw_bnb_8bit",
)

class SmolVLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)

        taxonomy_target = inputs.pop("taxonomy_target", None)

        outputs = model(
            **inputs,
            taxonomy_target=taxonomy_target,
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

model = SmolVLMWithTaxonomy(base_model, NUM_TAX)

trainer = SmolVLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()


# =================
# Validation (post-training evaluation on PHLOPValDataset)
# =================


class PHLOPValDataset(PHLOPVideoDataset):
    def __init__(self, *args, camera_mode="static", **kwargs):
        assert camera_mode in ["static", "moving"]
        super().__init__(*args, camera_mode=camera_mode, **kwargs)

    def __getitem__(self, idx):
        item = self.data[idx]

        video_path = item["videos"][self.camera_mode]
        qa_path = item["qa"][self.camera_mode]
        meta_path = item["metadata"][self.camera_mode]

        video = self._load_video(video_path)
        raw = load_json(os.path.join(self.root_dir, qa_path))
        if isinstance(raw, list):
            qa_list = raw
        elif isinstance(raw, dict):
            qa_list = raw.get("questions", raw.get("qa", []))
        else:
            qa_list = []
        if not qa_list:
            raise ValueError(f"Empty QA list for {qa_path} (idx={idx})")
        qa_entry = random.choice(qa_list)

        metadata = load_json(os.path.join(self.root_dir, meta_path))

        video_frames = [
            Image.fromarray(f.astype("uint8"))
            for f in video
        ]

        question = qa_entry["question"]
        answer = qa_entry["answer"]
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        elif not isinstance(answer, str):
            answer = str(answer)
        physics_signals = qa_entry.get("physics_signals", {})

        if qa_entry.get("options"):
            opts = "\n".join([f"- {o}" for o in qa_entry["options"]])
            question = f"{question}\nOptions:\n{opts}"

        image_tokens = "\n".join(["<image>"] * len(video_frames))

        prompt = (
            "You are a physics reasoning system.\n"
            "First infer the physical properties and events.\n"
            "Then answer the question.\n\n"
            f"{image_tokens}\n\n"
            f"Question:\n{question}\n\n"
            "Respond in the following format:\n"
            "Physics:\n"
            "- <key>: <value>\n"
            "Answer:\n"
        )

        target = (
            "Physics:\n"
            + format_physics_signals(physics_signals)
            + "\n\nAnswer:\n"
            + answer
        )

        return {
            "video": video_frames,
            "prompt": prompt,
            "target": target,
            "metadata": metadata,
            "qa_entry": qa_entry,
            "camera_mode": self.camera_mode,
        }


def extract_answer(text):
    text = text.lower()
    if "answer:" in text:
        return text.split("answer:", 1)[1].strip()
    return text.strip()


def answer_token_f1(pred, gt):
    """Word-level F1 between the extracted answer parts of pred and gt."""
    pred_str = extract_answer(pred)
    gt_str = extract_answer(gt)
    pred_tokens = pred_str.lower().split()
    gt_tokens = gt_str.lower().split()

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    tp = sum(common.values())
    precision = tp / len(pred_tokens)
    recall = tp / len(gt_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def answer_accuracy(pred, gt):
    return int(extract_answer(gt) in extract_answer(pred))


def extract_physics_kv(text):
    physics = {}
    for line in text.lower().splitlines():
        if line.startswith("-") and ":" in line:
            k, v = line[1:].split(":", 1)
            physics[k.strip()] = v.strip()
    return physics


def physics_signal_accuracy(pred_text, gt_text):
    gt = extract_physics_kv(gt_text)
    pred = extract_physics_kv(pred_text)

    if not gt:
        return None

    correct = sum(
        1 for k, v in gt.items()
        if k in pred and v in pred[k]
    )
    return correct / len(gt)


def flatten_taxonomy(metadata):
    labels = set()
    for frame in metadata.get("frames", []):
        for obj in frame.get("objects", {}).values():
            for entry in obj.get("taxonomy", []):
                for label in entry.get("labels", []):
                    labels.add(label.lower())
    return labels


def extract_predicted_physics_labels(text):
    labels = set()
    for line in text.lower().splitlines():
        if line.startswith("-"):
            labels.add(line.split(":", 1)[-1].strip())
    return labels


def taxonomy_f1(pred, gt):
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    tp = len(pred & gt)
    p = tp / len(pred)
    r = tp / len(gt)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def collect_predictions(model, processor, dataset, max_samples=None):
    model.eval()
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

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
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


def compute_metrics(results):
    if not results:
        return {
            "answer_accuracy": 0.0,
            "physics_signal_accuracy": 0.0,
            "taxonomy_f1": 0.0,
            "per_question_type": {},
        }

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

    # Per-question-type: accuracy and F1
    by_type = {}
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


def evaluate_model(model, processor, dataset, max_samples=100):
    """Collect all predictions, then compute metrics from the full list."""
    results = collect_predictions(model, processor, dataset, max_samples=max_samples)
    return compute_metrics(results)


val_static = PHLOPValDataset(
    hf_dataset_split=dataset["validation"],
    num_frames=16,
    split="val",
    root_dir=local_repo,
    camera_mode="static",
)

val_moving = PHLOPValDataset(
    hf_dataset_split=dataset["validation"],
    num_frames=16,
    split="val",
    root_dir=local_repo,
    camera_mode="moving",
)

static_metrics = evaluate_model(model, processor, val_static)
moving_metrics = evaluate_model(model, processor, val_moving)


def print_metrics(metrics, title):
    print(title)
    print(f"  answer_accuracy: {metrics['answer_accuracy']:.4f}")
    print(f"  physics_signal_accuracy: {metrics['physics_signal_accuracy']:.4f}")
    print(f"  taxonomy_f1: {metrics['taxonomy_f1']:.4f}")
    per = metrics.get("per_question_type", {})
    if per:
        print("  per_question_type:")
        for qt, v in sorted(per.items(), key=lambda x: -x[1]["count"]):
            print(f"    {qt}: accuracy={v['accuracy']:.4f}, f1={v['f1']:.4f}, count={v['count']}")


print_metrics(static_metrics, "STATIC CAMERA METRICS")
print()
print_metrics(moving_metrics, "MOVING CAMERA METRICS")