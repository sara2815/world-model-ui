import torch
from pathlib import Path
from PIL import Image
from diffusers import Flux2KleinPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

SCREENSHOT = "/home/adyesha7/dataset/271/step_4_20250723@075401_not_annotated_with_cursor.png"  #path on th server
ACTION = "click the white space to unselect the graph"
OUTPUT_DIR = Path("/scratch/adyesha7/guidance_test")
STEPS = 8
SEED = 42
GUIDANCE_VALUES = [1.0, 3.0, 5.0, 7.0]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[load] loading pipeline...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
).to(DEVICE)

img = Image.open(SCREENSHOT).convert("RGB")
w, h = img.size
w, h = (w // 64) * 64, (h // 64) * 64
img = img.resize((w, h), Image.LANCZOS)
img.save(OUTPUT_DIR / "input.png")

prompt = (
    f"A UI screenshot after the following action was performed: {ACTION.strip()}. "
    "The screen clearly shows the result of this action. "
    "Pixel-accurate UI, crisp text, clean layout."
)
print(f"[prompt] {prompt}")

for g in GUIDANCE_VALUES:
    print(f"[guidance={g}] generating...")
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    result = pipe(
        prompt=prompt, image=img, height=h, width=w,
        guidance_scale=g, num_inference_steps=STEPS, generator=gen,
    ).images[0]
    out_path = OUTPUT_DIR / f"guidance_{g}.png"
    result.save(out_path)
    print(f"    saved {out_path}")

print(f"\n[done] results in {OUTPUT_DIR}")