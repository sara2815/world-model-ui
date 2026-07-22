"""
test_prompt_variants.py

Takes ONE input screenshot and ONE action (pulled from the "action_new" field of a
triplets JSON file), and runs it through FLUX.2-klein-4B with 4 different PROMPT
TEMPLATES, so you can compare how the wording of the prompt affects the prediction
for a fixed image + action. All other inference settings (resize, steps, guidance)
match batch_inference.py / test_action_commands.py.

Usage:
  python test_prompt_variants.py \
      --image /path/to/before.png \
      --json /path/to/actions_triplets_new.json \
      --index 0 \
      --output_dir ./prompt_compare
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
print(f"[cuda check] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[cuda check] device_name={torch.cuda.get_device_name(0)}")
else:
    print("[cuda check] WARNING: running on CPU -- this will be extremely slow.")

PROMPT_TEMPLATES = [
    # Level 1 
    "{action}",

    # Level 2
    (
        "This is the current UI state. Predict the next UI state after this "
        "action: {action}"
    ),

    # Level 3 
    (
        "You are a world model for UI screenshots. Given the current screenshot "
        "and an action, generate the screenshot of the resulting next state. "
        "Action: {action}. Keep the layout, text, and unrelated elements "
        "unchanged; only reflect the effect of this action."
    ),

    # Level 4
    (
        "You are simulating a deterministic world model of a graphical user "
        "interface. You are given the current UI screenshot (state S_t) and a "
        "single user action. Your task is to render the screenshot of the next "
        "state S_t+1 that would result from applying this action to S_t.\n\n"
        f"Action to apply: {{action}}\n\n"
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
]
# ----------------------------------------------------------------------


def load_pipeline(model_id: str = "black-forest-labs/FLUX.2-klein-4B") -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(DEVICE)
    return pipe


def load_action_new(json_path: Path, index: int = None, screenshot_name: str = None) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    if screenshot_name is not None:
        matches = [
            (i, row) for i, row in enumerate(data)
            if Path(row.get("screenshot_before", "")).name == screenshot_name
        ]
        if not matches:
            available = [Path(row.get("screenshot_before", "")).name for row in data]
            raise ValueError(
                f"No entry with screenshot_before == '{screenshot_name}' found in {json_path}.\n"
                f"Available screenshot_before values in this file:\n  " + "\n  ".join(available)
            )
        if len(matches) > 1:
            print(f"[warn] {len(matches)} entries matched '{screenshot_name}', using the first one (index {matches[0][0]})")
        i, row = matches[0]
        print(f"[match] '{screenshot_name}' -> index {i}")
        action = row.get("action_new", "")
        if not action:
            raise ValueError(f"Matched entry (index {i}) has an empty 'action_new' field")
        return action

    if index is None:
        raise ValueError("Must provide either --index or --screenshot_name")
    if not (0 <= index < len(data)):
        raise IndexError(f"--index {index} out of range (json has {len(data)} steps)")
    action = data[index].get("action_new", "")
    if not action:
        raise ValueError(f"Step {index} has an empty 'action_new' field")
    return action


def run(screenshot: Path, action: str, pipe: Flux2KleinPipeline, prompt_template: str,
        steps: int = 4, guidance: float = 1.0, seed: int | None = None) -> Image.Image:
    img = Image.open(screenshot).convert("RGB")
    w, h = img.size
    w = (w // 64) * 64
    h = (h // 64) * 64
    img = img.resize((w, h), Image.LANCZOS)

    prompt = prompt_template.format(action=action.strip())

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
    p.add_argument("--json", required=True, type=Path, help="Path to actions_triplets(_new).json.")
    p.add_argument("--index", type=int, default=None,
                    help="Index into the JSON list to pull action_new from. "
                         "Ignored if --screenshot_name is given.")
    p.add_argument("--screenshot_name", default=None,
                    help="Filename (e.g. step_7_...png) to match against screenshot_before in the JSON. "
                         "If omitted, defaults to the basename of --image.")
    p.add_argument("--output_dir", type=Path, default=Path("prompt_compare"))
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    args = p.parse_args()

    if args.screenshot_name is None and args.index is None:
        # Default: auto-match against the input image's own filename
        args.screenshot_name = args.image.name
        print(f"[auto] no --index or --screenshot_name given, matching JSON by --image filename: {args.screenshot_name}")

    action = load_action_new(args.json, index=args.index, screenshot_name=args.screenshot_name)
    print(f"[action_new] {action}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] loading pipeline...")
    pipe = load_pipeline(args.model)

    # Save the shared input + action once
    Image.open(args.image).convert("RGB").save(args.output_dir / "input.png")
    (args.output_dir / "action_new.txt").write_text(action, encoding="utf-8")

    for i, template in enumerate(PROMPT_TEMPLATES):
        prompt = template.format(action=action.strip())
        print(f"[{i+1}/{len(PROMPT_TEMPLATES)}] prompt: {prompt}")
        out_dir = args.output_dir / f"prompt_{i+1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        pred = run(args.image, action, pipe, template,
                   steps=args.steps, guidance=args.guidance, seed=args.seed + i)
        pred.save(out_dir / "pred_after.png")
        (out_dir / "prompt_used.txt").write_text(prompt, encoding="utf-8")

        print(f"    saved to {out_dir}")

    print(f"\n[done] results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()