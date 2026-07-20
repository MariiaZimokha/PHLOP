import os
import re
import json
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Import your existing utilities
from phlop_eval_common import load_phlop_splits, compute_metrics, load_json_file, get_physical_props
from smol_eval import _load_video_as_pil

# --- EXPERIMENT CONFIGURATION ---
MODEL_ID = "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
BASE_DIR = "./ablation_experiments"
TOTAL_ITERS = 200
EVAL_EVERY = 25
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_FRAMES_TO_EXTRACT = 6 

FINE_TUNE_CONFIGS = {
    "easy": {"train_difficulty": ["easy"], "val_on_rest": True},
    "easy_medium": {"train_difficulty": ["easy", "medium"], "val_on_rest": True},
    "hard": {"train_difficulty": ["hard"], "val_on_rest": True},
    "full": {"train_difficulty": None, "val_on_rest": False},
}

os.makedirs(BASE_DIR, exist_ok=True)


def _build_prompt(question: str, options: list = None, physical_props: dict = None,
                  num_frames: int = NUM_FRAMES_TO_EXTRACT) -> str:
    """Build a structured prompt matching the smol_eval format."""
    options_section = ""
    if options:
        opts_formatted = "\n".join(f"  - {o}" for o in options)
        options_section = f"\nOptions:\n{opts_formatted}\nPick exactly one option (reply with the EXACT option text)."

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
        f"Frames sampled: {num_frames}\n"
        f"{props_section}\n"
        f"Question:\n{question}{options_section}\n\n"
        "Respond with ONLY the final answer:\n"
        "- yes/no questions: answer \"yes\" or \"no\"\n"
        "- counting questions: answer with a number\n"
        "- time questions: answer like \"0.5s\"\n"
        "- multiple choice: reply with the EXACT option text, nothing else\n\n"
        "Answer:\n"
    )


def prepare_mlx_dataset(splits, split_name, output_file, frames_dir, allowed_difficulties, use_physics, val_on_rest=False):
    """Formats data based on difficulty configs and physics toggles."""
    print(f"  -> Preparing {split_name} dataset... (Physics: {use_physics})")
    ds = splits[split_name]
    
    with open(output_file, 'w') as f:
        count = 0
        for i in tqdm(range(len(ds)), desc=f"Exporting {split_name}", leave=False):
            
            sample = ds[i]
            video_path = (sample.get("videos") or {}).get("static")
            qa_path = (sample.get("qa") or {}).get("static")
            meta_path = (sample.get("metadata") or {}).get("static")
            if not video_path or not qa_path: continue
            
            with open(qa_path, 'r') as qaf:
                qa_data = json.load(qaf)
            
            frame_paths = None
            physical_props = None
            if use_physics and meta_path:
                metadata = load_json_file(meta_path)
                physical_props = get_physical_props(metadata) or None
            
            for qa_item in qa_data:
                diff = qa_item.get("difficulty", "unknown")
                
                if allowed_difficulties is not None:
                    is_in_target = diff in allowed_difficulties
                    if split_name == "train" and not is_in_target:
                        continue 
                    if split_name == "validation" and val_on_rest and is_in_target:
                        continue 
                
                if frame_paths is None:
                    frames, _ = _load_video_as_pil(video_path, num_frames=NUM_FRAMES_TO_EXTRACT)
                    if not frames: break
                    frame_paths = []
                    for fi, frame in enumerate(frames):
                        fp = os.path.join(frames_dir, f"{split_name}_{i}_f{fi}.jpg")
                        frame.save(fp)
                        frame_paths.append(fp)
                
                question = qa_item.get("question", "")
                answer = qa_item.get("answer", "")
                options = qa_item.get("options", None)
                
                prompt = _build_prompt(
                    question, options=options,
                    physical_props=physical_props,
                )
                
                row = {
                    "question": prompt,
                    "answer": str(answer),
                    "images": frame_paths,
                    "question_type": qa_item.get("question_type") or qa_item.get("category") or "unknown",
                    "difficulty": diff,
                }
                f.write(json.dumps(row) + '\n')
                count += 1


def run_mlx_training_step(iters: int, data_dir: str, adapter_path: str):
    """Runs a chunk of training."""
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")
    cmd = [
        "python", "-m", "mlx_vlm.lora",
        "--model-path", MODEL_ID,
        "--dataset", data_dir,
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(iters),
        "--learning-rate", str(LEARNING_RATE),
        "--max-seq-length", "8000",
        "--output-path", adapter_file
    ]
    
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        cmd.extend(["--adapter-path", adapter_path])
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    losses = []
    
    # Stream the output live to the terminal
    for line in process.stdout:
        print(line, end="") 
        match = re.search(r"Train loss (\d+\.\d+)", line)
        if match:
            losses.append(float(match.group(1)))
            
    process.wait()
    
    # HARD CRASH if training fails, preventing the script from proceeding to evaluation
    if process.returncode != 0:
        raise RuntimeError(
            f"\n\n❌ MLX Training crashed with exit code {process.returncode}. "
            f"Please scroll up slightly to read the exact error printed by the MLX subprocess."
        )
        
    return losses

def evaluate_model(eval_file: str, adapter_path: str, save_path: str = None):
    """Evaluates the model on a JSONL file and clears memory.
    
    Args:
        eval_file: Path to the JSONL file (e.g. valid.jsonl or test.jsonl).
        adapter_path: Directory containing the trained adapter weights.
        save_path: If set, saves per-sample predictions to this JSON file.
    
    Returns:
        dict with answer_accuracy and per_question_type metrics.
    """
    print(f"  -> Evaluating on {eval_file} ...")
    model, processor = load(MODEL_ID, adapter_path=adapter_path)
    
    results = []
    with open(eval_file, "r") as f:
        for line in tqdm(f.readlines(), desc="Inference", leave=False):
            data = json.loads(line)
            
            prompt = data["question"]
            target = data["answer"]
            image_paths = data["images"]
            
            formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=len(image_paths))
            prediction = generate(model, processor, formatted_prompt, image_paths, verbose=False, max_tokens=100)
            
            results.append({
                "prediction": prediction.text.strip(),
                "target": target,
                "question_type": data.get("question_type", "unknown"),
                "difficulty": data.get("difficulty", "unknown"),
            })
    
    del model
    del processor
    mx.metal.clear_cache()
    
    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> Saved {len(results)} predictions to {save_path}")
    
    return compute_metrics(results)


def run_experiment(splits, config_name, config_params, use_physics):
    """Orchestrates a single run of the ablation matrix."""
    exp_name = f"{config_name}_physics_{use_physics}"
    print(f"\n{'='*50}\nSTARTING EXPERIMENT: {exp_name}\n{'='*50}")
    
    exp_dir = os.path.join(BASE_DIR, exp_name)
    data_dir = os.path.join(exp_dir, "data")
    frames_dir = os.path.join(exp_dir, "frames")
    adapter_path = os.path.join(exp_dir, "adapter")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(adapter_path, exist_ok=True)
    
    train_diffs = config_params["train_difficulty"]
    val_on_rest = config_params["val_on_rest"]
    
    train_file = os.path.join(data_dir, "train.jsonl")
    valid_file = os.path.join(data_dir, "valid.jsonl")
    test_file  = os.path.join(data_dir, "test.jsonl")
    
    prepare_mlx_dataset(splits, "train", train_file, frames_dir, train_diffs, use_physics)
    prepare_mlx_dataset(splits, "validation", valid_file, frames_dir, train_diffs, use_physics, val_on_rest=val_on_rest)
    
    # --- Training loop with periodic validation ---
    current_step = 0
    while current_step < TOTAL_ITERS:
        run_mlx_training_step(EVAL_EVERY, data_dir, adapter_path)
        current_step += EVAL_EVERY
        
        val_metrics = evaluate_model(valid_file, adapter_path)
        val_acc = val_metrics["answer_accuracy"]
        print(f"\n  [Exp: {exp_name}] Step {current_step} - Val Accuracy: {val_acc:.4f}")
    
    # --- Final evaluation on the test split ---
    prepare_mlx_dataset(splits, "test", test_file, frames_dir,
                        allowed_difficulties=None, use_physics=use_physics)
    
    predictions_file = os.path.join(exp_dir, "test_predictions.json")
    test_metrics = evaluate_model(test_file, adapter_path, save_path=predictions_file)
    
    print(f"\n  [Exp: {exp_name}] === FINAL TEST RESULTS ===")
    print(f"  Accuracy: {test_metrics['answer_accuracy']:.4f}")
    for qt, v in sorted(test_metrics["per_question_type"].items(), key=lambda x: -x[1]["count"]):
        print(f"    {qt}: accuracy={v['accuracy']:.4f}, f1={v['f1']:.4f}, count={v['count']}")
    
    return test_metrics

if __name__ == "__main__":
    print("Loading PHLOP splits...")
    splits = load_phlop_splits("zimmari-ai/phlop", token=True)
    
    results_summary = {}
    
    for config_name, config_params in FINE_TUNE_CONFIGS.items():
        for use_physics in [False, True]:
            test_metrics = run_experiment(splits, config_name, config_params, use_physics)
            results_summary[f"{config_name} (Physics: {use_physics})"] = test_metrics
    
    # --- Save full results to JSON ---
    summary_path = os.path.join(BASE_DIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nFull results saved to {summary_path}")
    
    # --- Print summary table ---
    print("\n\n" + "="*60)
    print("ABLATION STUDY RESULTS (Test Set)")
    print("="*60)
    for exp, metrics in results_summary.items():
        acc = metrics["answer_accuracy"]
        per_qt = metrics["per_question_type"]
        avg_f1 = (sum(v["f1"] for v in per_qt.values()) / len(per_qt)) if per_qt else 0.0
        print(f"\n  {exp}")
        print(f"    Accuracy: {acc:.4f}  |  Avg F1: {avg_f1:.4f}")
        for qt, v in sorted(per_qt.items(), key=lambda x: -x[1]["count"]):
            print(f"      {qt:<25} acc={v['accuracy']:.4f}  f1={v['f1']:.4f}  n={v['count']}")
