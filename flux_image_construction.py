"""
flux_screenshot_action.py

Takes a screenshot + action text, generates the next predicted screenshot
using FLUX.2-klein-4B's built-in image editing capability.

Usage:
  python flux_screenshot_action.py --screenshot current.png --action "click the create button" --output next.png
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


def load_pipeline(model_id: str = "black-forest-labs/FLUX.2-klein-4B",
                  dtype=torch.bfloat16) -> Flux2KleinPipeline:
    
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe


def predict_next_screenshot(
    screenshot,
    action: str,
    *,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    pipe: Optional[Flux2KleinPipeline] = None,
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    dtype=torch.bfloat16,
) -> Image.Image:

    # Load image
    if not isinstance(screenshot, Image.Image):
        screenshot = Image.open(screenshot).convert("RGB")
    else:
        screenshot = screenshot.convert("RGB")

    # Align dimensions to 64px grid (required by FLUX)
    w, h = screenshot.size
    w_adj = (w // 64) * 64
    h_adj = (h // 64) * 64
    if (w_adj, h_adj) != (w, h):
        screenshot = screenshot.resize((w_adj, h_adj), Image.LANCZOS)
        print(f"[resize] {w}x{h} -> {w_adj}x{h_adj}")

    # Load pipeline if not provided
    if pipe is None:
        print(f"[load] {model_id} ...")
        pipe = load_pipeline(model_id, dtype=dtype)

    # Build prompt describing desired next state
    prompt = (
        f"Generate the UI screenshot after the following action was performed: {action.strip()}. "
        "Pixel-accurate UI, crisp legible text, clean layout."
    )
    print(f"[prompt] {prompt}")
    
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        generator.manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=screenshot,           # <-- pass screenshot as input image
        height=h_adj,
        width=w_adj,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    return result.images[0]


def parse_args():
    p = argparse.ArgumentParser(description="Predict next UI screenshot after an action.")
    p.add_argument("--screenshot", required=True, help="Path to current screenshot (PNG/JPG).")
    p.add_argument("--action", required=True, help='e.g. "click on the Sign in Button and go to that page"')
    p.add_argument("--output", default="next_screenshot.png", help="Output path.")
    p.add_argument("--steps", type=int, default=8, help="Inference steps (default 8).") #added more steps
    p.add_argument("--guidance", type=float, default=4.0, help="Guidance scale (default 4.0).") # change guidance
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    next_img = predict_next_screenshot(
        screenshot=args.screenshot,
        action=args.action,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        model_id=args.model,
    )

    out_path = Path(args.output)
    next_img.save(out_path)
    print(f"[saved] {out_path.resolve()}")