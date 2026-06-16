"""
batch_inference.py

Randomly samples N (screenshot_before, action, screenshot_after) triplets from a
DashboardQA dataset and generates predicted next screenshots using FLUX.2-klein-4B.

Usage:
  python batch_inference.py --root /path/to/results_cua_screenshot --n 20 --output_dir ./predictions
"""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")


def load_pipeline(model_id: str = "black-forest-labs/FLUX.2-klein-4B") -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(DEVICE)
    return pipe


def collect_triplets(root: Path) -> list[dict]:
    """Walk all episode dirs and collect every (before, action, after, episode) entry."""
    triplets = []
    for ep_dir in sorted(root.iterdir()):
        json_path = ep_dir / "actions_triplets.json"
        if not ep_dir.is_dir() or not json_path.is_file():
            continue
        with open(json_path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            before = ep_dir / row.get("screenshot_before", "")
            after  = ep_dir / row.get("screenshot_after", "")
            action = row.get("action_full") or row.get("action_raw", "")
            if before.is_file() and after.is_file() and action:
                triplets.append({
                    "before": before,
                    "after":  after,
                    "action": action,
                    "episode": ep_dir.name,
                    "step_number": row.get("step_number", "?"),
                })
    return triplets


def run(screenshot: Path, action: str, pipe: Flux2KleinPipeline,
        steps: int = 4, guidance: float = 1.0, seed: int | None = None) -> Image.Image:
    img = Image.open(screenshot).convert("RGB")
    w, h = img.size
    w = (w // 64) * 64
    h = (h // 64) * 64
    img = img.resize((w, h), Image.LANCZOS)

    prompt = (
        f"A UI screenshot after the following action was performed: {action.strip()}. "
        "The screen clearly shows the result of this action. "
        "Pixel-accurate UI, crisp text, clean layout."
    )

    gen = torch.Generator(device=DEVICE)
    if seed is not None:
        gen.manual_seed(seed)

    return pipe(
        prompt=prompt,
        image=img,
        height=h,
        width=w,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=gen,
    ).images[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",       required=True, type=Path, help="Path to results_cua_screenshot.")
    p.add_argument("--n",          type=int, default=20,     help="Number of samples to run.")
    p.add_argument("--output_dir", type=Path, default=Path("predictions"))
    p.add_argument("--steps",      type=int, default=4)
    p.add_argument("--guidance",   type=float, default=1.0)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--model",      default="black-forest-labs/FLUX.2-klein-4B")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[collect] scanning dataset...")
    triplets = collect_triplets(args.root)
    print(f"[collect] found {len(triplets)} valid triplets")

    if len(triplets) == 0:
        raise RuntimeError("No valid triplets found. Check --root path.")

    random.seed(args.seed)
    samples = random.sample(triplets, min(args.n, len(triplets)))

    print("[load] loading pipeline...")
    pipe = load_pipeline(args.model)

    for i, s in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] ep={s['episode']} step={s['step_number']} | {s['action'][:80]}")
        out_dir = args.output_dir / s["episode"] / str(s["step_number"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy ground truth before/after for comparison
        Image.open(s["before"]).save(out_dir / "before.png")
        Image.open(s["after"]).save(out_dir / "gt_after.png")

        # Generate predicted next screenshot
        pred = run(s["before"], s["action"], pipe,
                   steps=args.steps, guidance=args.guidance, seed=args.seed + i)
        pred.save(out_dir / "pred_after.png")

        # Save action text
        (out_dir / "action.txt").write_text(s["action"], encoding="utf-8")

        print(f"    saved to {out_dir}")

    print(f"\n[done] results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()