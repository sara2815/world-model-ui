import os
import sys
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from world_model_ui.data.dataset import WorldModelDataset

NUM_EXAMPLES = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "cuda"
dtype = torch.bfloat16
#black-forest-labs/FLUX.2-klein-9B # FLUX.2-klein-4B # FLUX.2-klein-base-4B
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype #black-forest-labs/FLUX.2-klein-base-4B
)
pipe = pipe.to(device)

generator = torch.Generator(device=device).manual_seed(0)

ds = WorldModelDataset(use_annotated_image=True)

for i in range(min(NUM_EXAMPLES, len(ds))):
    sample = ds[i]
    input_image = sample["input_image"]
    action_text = sample["action_text"]
    #action_text = "Change the background to cyan and add a red circle in the center."
    print("action_text:", action_text)
    prompt = (
        "Given the following UI image and action, predict the resulting UI image after performing the action:\n"
        + action_text
    )
    # Build prefix from source folder, e.g. "results_cua_screenshot/0" -> "results_cua_screenshot_0"
    prefix = sample["source"].replace("/", "_").replace(os.sep, "_")
    gt_path = Path(ds.samples[i]["output_path"])
    base_name = f"{prefix}_{gt_path.stem}"

    input_path = Path(ds.samples[i]["input_path"])
    input_save_path = OUTPUT_DIR / f"{prefix}_{input_path.stem}_input.png"
    gt_save_path = OUTPUT_DIR / f"{base_name}_gt.png"
    pred_save_path = OUTPUT_DIR / f"{base_name}_predicted.png"

    # Save input and GT
    input_image.save(input_save_path)
    sample["output_image"].save(gt_save_path)

    w, h = input_image.size
    result = pipe(
        prompt=prompt,
        image=input_image,
        height=h,
        width=w,
        guidance_scale= 4.0,
        num_inference_steps=25,
        generator=generator,
    ).images[0]

    result.save(pred_save_path)
    print(f"[{i+1}/{NUM_EXAMPLES}] step={sample['step_number']} | {action_text}")
    print(f"  input     -> {input_save_path}")
    print(f"  gt        -> {gt_save_path}")
    print(f"  predicted -> {pred_save_path}")
