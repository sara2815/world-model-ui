import torch
from pathlib import Path
from PIL import Image
from diffusers import Flux2KleinPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}", flush=True)

SCREENSHOT = "/home/adyesha7/dataset/271/step_4_20250723@075401_not_annotated_with_cursor.png"
ACTION = "click the white space to unselect the graph"
OUTPUT_DIR = Path("/scratch/adyesha7/inference_steps_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GUIDANCE = 4.0
INFERENCE_STEPS = [4, 6, 8, 12, 16, 24]
SEED = 42

print("[load] loading pipeline...", flush=True)
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16
).to(DEVICE)
print("[load] done", flush=True)

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
print(f"[prompt] {prompt}", flush=True)

for steps in INFERENCE_STEPS:
    print(f"[steps={steps}] generating...", flush=True)

    gen = torch.Generator(device=DEVICE).manual_seed(SEED)

    result = pipe(
        prompt=prompt,
        image=img,
        height=h,
        width=w,
        guidance_scale=GUIDANCE,
        num_inference_steps=steps,
        generator=gen,
    ).images[0]

    out_path = OUTPUT_DIR / f"steps_{steps}.png"
    result.save(out_path)
    print(f"    saved {out_path}", flush=True)

print("[done]", flush=True)