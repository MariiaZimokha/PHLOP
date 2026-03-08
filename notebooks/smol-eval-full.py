"""
PHLOP full eval CLI: data analysis, zero-shot, fine-tune, test comparison for SmolVLM.
Thin wrapper that imports from phlop_eval_common and smol_eval; use those modules
directly in notebooks to load once, analyze once, then run model-specific steps.

Notebook usage (recommended):
    from phlop_eval_common import load_phlop_splits, run_data_analysis
    from smol_eval import run_zero_shot_smolvlm, run_finetune_smolvlm, run_test_comparison_smolvlm
    splits = load_phlop_splits("zimmari-ai/phlop", token=True)
    run_data_analysis(splits)
    run_zero_shot_smolvlm(splits, max_samples=50)
    run_finetune_smolvlm(splits, output_dir="./out", max_steps=10)
    run_test_comparison_smolvlm(splits, [("base", "HuggingFaceTB/..."), ("easy", "./out/easy")])
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from phlop_eval_common import (
    load_phlop_splits,
    run_data_analysis,
    DEFAULT_REPO_ID,
    FINE_TUNE_CONFIGS,
)
from smol_eval import (
    run_zero_shot_smolvlm,
    run_finetune_smolvlm,
    run_test_comparison_smolvlm,
)


def main():
    parser = argparse.ArgumentParser(description="PHLOP full eval: analysis, zero-shot, fine-tune, test")
    parser.add_argument("--repo-id", type=str, default=os.environ.get("PHLOP_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--analyze-only", action="store_true", help="Only run data analysis and exit")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip data analysis step")
    parser.add_argument("--zero-shot-only", action="store_true", help="Only run zero-shot on val/test and exit")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--skip-test-comparison", action="store_true")
    parser.add_argument("--max-zero-shot", type=int, default=None, help="Cap zero-shot samples per split")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./smolvlm_physics_full")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--camera-mode", type=str, default="static", choices=["static", "moving"])
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token for private repo (or set HF_TOKEN)")
    args = parser.parse_args()

    repo_id = args.repo_id
    print(f"Repo: {repo_id}")

    splits = load_phlop_splits(
        repo_id,
        token=args.token or os.environ.get("HF_TOKEN") or True,
        extract_root=os.environ.get("PHLOP_EXTRACT_ROOT"),
    )
    if not splits:
        print("No splits loaded. Check repo layout (train/validation/test).")
        return
    print(f"Loaded splits: {list(splits.keys())}")

    if not args.skip_analysis:
        run_data_analysis(splits)
    if args.analyze_only:
        return

    if "validation" not in splits or "test" not in splits:
        print("Need validation and test splits for zero-shot.")
    elif not args.skip_zero_shot:
        run_zero_shot_smolvlm(
            splits,
            camera_mode=args.camera_mode,
            max_samples=args.max_zero_shot,
        )
        if args.zero_shot_only:
            return

    if not args.skip_finetune:
        run_finetune_smolvlm(
            splits,
            output_dir=args.output_dir,
            config_name=None,
            max_steps=args.max_steps,
            camera_mode=args.camera_mode,
        )

    if not args.skip_test_comparison and "test" in splits:
        model_checkpoints = [("base", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")]
        for config_name in FINE_TUNE_CONFIGS:
            ckpt = os.path.join(args.output_dir, config_name)
            if os.path.isdir(ckpt):
                model_checkpoints.append((config_name, ckpt))
        run_test_comparison_smolvlm(
            splits,
            model_checkpoints,
            camera_mode=args.camera_mode,
        )


if __name__ == "__main__":
    main()
