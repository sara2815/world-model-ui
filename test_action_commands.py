"""
test_action_commands.py

Takes ONE input screenshot and runs it through FLUX.2-klein-4B with 4 different
action prompts, so you can compare how the model reacts to different actions on
the same starting state. Inference settings (resize, prompt template, steps,
guidance) are kept identical to batch_inference.py.

Usage:
  python test_action_commands.py --image /path/to/before.png --output_dir ./test_action_commands
"""

import argparse
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


def run(screenshot: Path, action: str, pipe: Flux2KleinPipeline,
        steps: int = 4, guidance: float = 1.0, seed: int | None = None) -> Image.Image:
    img = Image.open(screenshot).convert("RGB")
    w, h = img.size
    w = (w // 64) * 64
    h = (h // 64) * 64
    img = img.resize((w, h), Image.LANCZOS)

    prompt = (
        f"Generate a UI screenshot after the following action was performed: {action.strip()}. "
        "The predicted screen clearly shows the result of this action. "
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
    p.add_argument("--image", required=True, type=Path, help="Path to the single input screenshot.")
    p.add_argument("--output_dir", type=Path, default=Path("action_compare"))
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    args = p.parse_args()

    # --- Edit these 4 actions to whatever you want to compare ---
    actions = [
        "click on (424, 353) and then type 'Tableau Public Mortgage complaints' and then sleep 1.00s and then press enter",
        "click on item 60 (div \"Fig. 5: Commodity profile of imports through ICDs - All. Data Visualization. Bubble chart, Color applied to Product Cate\") sleep 1.00s, type 'Tableau Public Mortgage complaints, sleep 1.00s, press Enter, scroll down to the bottom of the page, close the current dialog",
        "make a black cross on (424, 353) and then type 'Tableau Public Mortgage complaints' and then press enter",
        "make a black cross on the Plastics, Rubber and their products section and type 'Tableau Public Mortgage complaints' and then press enter",
    ]
    # --------------------------------------------------------------

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] loading pipeline...")
    pipe = load_pipeline(args.model)

    # Save the shared input once
    Image.open(args.image).convert("RGB").save(args.output_dir / "input.png")

    for i, action in enumerate(actions):
        print(f"[{i+1}/{len(actions)}] action: {action}")
        out_dir = args.output_dir / f"action_{i+1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        pred = run(args.image, action, pipe,
                   steps=args.steps, guidance=args.guidance, seed=args.seed + i)
        pred.save(out_dir / "pred_after.png")
        (out_dir / "action.txt").write_text(action, encoding="utf-8")

        print(f"    saved to {out_dir}")

    print(f"\n[done] results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
    
    
 #image used: "C:\Users\adyes\Downloads\results_gemini_pro_25_screenshot\results_gemini_pro_25_screenshot\168\step_7_20250724@160236_not_annotated_with_cursor.png"
    