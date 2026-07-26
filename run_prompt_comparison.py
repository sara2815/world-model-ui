#!/usr/bin/env python3
"""
run_prompt_comparison.py

Fixed/combined version of the original run_prompt_comparison.py scaffold:
  - predict_world_model() now actually calls FLUX.2-klein-4B (from
    test_prompt_variants.py), instead of returning the input image unchanged.
  - Prompts are built from the JSON's real "action_new" field (the schema
    your actions_triplets_filtered.json actually has), not the placeholder
    "action"/"point"/"thought" keys that don't exist in your data.
  - before/gt image pairing uses the JSON's own "screenshot_before" /
    "screenshot_after" fields directly, instead of guessing by sorted
    filename position (which breaks past step_9 due to lexicographic sort).
  - METRICS_SCRIPT points at change_metric.py (matches the filename you're
    actually using).

Usage:
  python run_prompt_comparison.py --root /path/to/filtered_world_model
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

METRICS_SCRIPT = Path(__file__).resolve().parent / "metric.py"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _swap_annotated(filename: str) -> str:
    """Match dataset.py's WorldModelDataset(use_annotated_image=True) behavior:
    the JSON always records the '_not_annotated_' filename, but the actual
    files on disk in filtered_world_model/ are the '_annotated_' versions."""
    return filename.replace("_not_annotated_with_cursor.png", "_annotated_with_cursor.png")


# --- Prompt levels, built from the JSON's real "action_new" field. ---
PROMPT_LEVELS = {
    "prompt1": lambda act: act.get("action_new", ""),
    "prompt2": lambda act: (
        "This is the current UI state. Predict the next UI state after this "
        f"action: {act.get('action_new', '')}"
    ),
    "prompt3": lambda act: (
        "You are a world model for UI screenshots. Given the current screenshot "
        "and an action, generate the screenshot of the resulting next state. "
        f"Action: {act.get('action_new', '')}. Keep the layout, text, and unrelated "
        "elements unchanged; only reflect the effect of this action."
    ),
    "prompt4": lambda act: (
        "You are simulating a deterministic world model of a graphical user "
        "interface. You are given the current UI screenshot (state S_t) and a "
        "single user action. Your task is to render the screenshot of the next "
        "state S_t+1 that would result from applying this action to S_t.\n\n"
        f"Action to apply: {act.get('action_new', '')}\n\n"
        "Requirements:\n"
        "- Only change what the action would plausibly change (e.g. a clicked "
        "element's state, a newly opened menu/dialog, updated text, a scrolled "
        "viewport).\n"
        "- Preserve all unrelated UI elements, layout, fonts, colors, and text "
        "exactly as they appear in the input image.\n"
        "- The output must be pixel-accurate, with crisp, legible text and a "
        "clean, consistent layout matching the input's visual style.\n"
        "- Do not introduce new UI elements, panels, or content that the action "
        "does not imply."
    ),
}

# ==============================================================================
# 1 & 2: STRUCTURE & PREDICTION PIPELINE
# ==============================================================================

_pipe = None  # lazy-loaded, shared across all calls so we only load once


def get_pipeline(model_id: str = "black-forest-labs/FLUX.2-klein-4B") -> Flux2KleinPipeline:
    global _pipe
    if _pipe is None:
        print(f"[load] loading {model_id} on {DEVICE}...")
        _pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        _pipe = _pipe.to(DEVICE)
    return _pipe


def predict_world_model(before_path: Path, prompt_text: str,
                         steps: int = 4, guidance: float = 1.0, seed: int = 42) -> Image.Image:
    """Runs FLUX.2-klein-4B image-editing inference. Same resize/inference
    logic as test_prompt_variants.py's run(), so results are directly
    comparable to earlier single-image prompt tests."""
    pipe = get_pipeline()

    img = Image.open(before_path).convert("RGB")
    w, h = img.size
    w = (w // 64) * 64
    h = (h // 64) * 64
    img = img.resize((w, h), Image.LANCZOS)

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    result = pipe(
        prompt=prompt_text,
        image=img,
        height=h,
        width=w,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=gen,
    ).images[0]
    return result


def build_structure_and_predict(dataset_root: Path, limit: int | None = None):
    print("=== Step 1 & 2: Structuring Folders & Generating Predictions ===")

    json_paths = list(dataset_root.rglob("actions_triplets_filtered.json"))
    if not json_paths:
        print(f"Warning: No actions_triplets_filtered.json found under {dataset_root}")
        return

    for json_path in json_paths:
        sample_dir = json_path.parent
        print(f"\nProcessing sample directory: {sample_dir}")

        with open(json_path, "r", encoding="utf-8") as f:
            actions_data = json.load(f)

        rows = actions_data[:limit] if limit is not None else actions_data

        for action_info in rows:
            step_number = action_info.get("step_number")
            before_name = _swap_annotated(action_info.get("screenshot_before", ""))
            after_name = _swap_annotated(action_info.get("screenshot_after", ""))
            action_text = action_info.get("action_new", "")

            if not before_name or not after_name or not action_text:
                print(f"  [skip] step {step_number}: missing screenshot_before/after or action_new")
                continue

            orig_before = sample_dir / before_name
            orig_gt = sample_dir / after_name
            if not orig_before.is_file() or not orig_gt.is_file():
                print(f"  [skip] step {step_number}: image file(s) not found "
                      f"(before_exists={orig_before.is_file()}, gt_exists={orig_gt.is_file()})")
                continue

            # Subfolder: step_0, step_1, etc. -- keyed by the JSON's real step_number
            step_dir = sample_dir / f"step_{step_number}"
            step_dir.mkdir(exist_ok=True)

            before_path = step_dir / "before.png"
            gt_path = step_dir / "gt_after.png"

            if not before_path.exists():
                Image.open(orig_before).convert("RGB").save(before_path)
            if not gt_path.exists():
                Image.open(orig_gt).convert("RGB").save(gt_path)

            # Build prompt text for every level from the REAL action_new field
            prompts_dict = {
                p_name: p_fn(action_info)
                for p_name, p_fn in PROMPT_LEVELS.items()
            }

            (step_dir / "prompts.txt").write_text(
                "".join(f"=== {name} ===\n{text}\n\n" for name, text in prompts_dict.items()),
                encoding="utf-8",
            )
            (step_dir / "prompts.json").write_text(json.dumps(prompts_dict, indent=2), encoding="utf-8")

            for prompt_name, prompt_text in prompts_dict.items():
                pred_out_path = step_dir / f"{prompt_name}_predicted.png"
                print(f" -> Generating step_{step_number}/{prompt_name}_predicted.png...")

                pred_img = predict_world_model(before_path, prompt_text)
                pred_img.save(pred_out_path)


# ==============================================================================
# 3: METRICS EVALUATION PIPELINE
# ==============================================================================

def evaluate_metrics(dataset_root: Path):
    print("\n=== Step 3: Computing Change Metrics across Prompt Levels ===")

    if not METRICS_SCRIPT.exists():
        print(f"Error: Could not find {METRICS_SCRIPT}.")
        return

    step_dirs = [p.parent for p in dataset_root.rglob("before.png")]
    if not step_dirs:
        print(f"Error: No step folders containing 'before.png' were found under {dataset_root}.")
        return

    print(f"Found {len(step_dirs)} valid step folder(s) across dataset.")

    summary_data = []

    for prompt_name in PROMPT_LEVELS.keys():
        pred_filename = f"{prompt_name}_predicted.png"
        csv_out_path = dataset_root / f"results_{prompt_name}.csv"

        print(f"\nRunning {METRICS_SCRIPT.name} for {prompt_name}...")

        cmd = [
            sys.executable, str(METRICS_SCRIPT),
            "--scan-root", str(dataset_root.resolve()),
            "--before-name", "before.png",
            "--gt-name", "gt_after.png",
            "--pred-name", pred_filename,
            "--out", str(csv_out_path.resolve()),
        ]

        try:
            subprocess.run(cmd, check=True)

            if csv_out_path.exists():
                df = pd.read_csv(csv_out_path)
                metrics_cols = ["change_ssim", "change_rmse", "localization_iou",
                                 "background_ssim", "composite_score"]
                valid_cols = [col for col in metrics_cols if col in df.columns]
                mean_scores = df[valid_cols].mean(numeric_only=True).to_dict()

                summary_data.append({
                    "prompt_level": prompt_name,
                    "num_triplets": len(df),
                    **mean_scores,
                })
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {prompt_name}: {e}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = dataset_root / "summary_metrics_comparison.csv"
        summary_df.to_csv(summary_csv, index=False)

        print("\n" + "=" * 60)
        print("SUMMARY COMPARISON ACROSS PROMPT LEVELS:")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print(f"\nSaved full aggregate comparison to: {summary_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                     help="Root of the filtered_world_model dataset (absolute path recommended).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N steps per episode (useful for a quick test).")
    ap.add_argument("--skip_predict", action="store_true",
                     help="Skip generation, only run evaluation on existing predictions.")
    args = ap.parse_args()

    if not args.skip_predict:
        build_structure_and_predict(args.root, limit=args.limit)
    evaluate_metrics(args.root)


if __name__ == "__main__":
    main()