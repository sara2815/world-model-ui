import os
import sys
from pathlib import Path

import torch
from diffusers import QwenImageEditPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from world_model_ui.data.dataset import WorldModelDataset

NUM_EXAMPLES = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "cuda"
dtype = torch.bfloat16
pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipe.to(dtype)
pipe.to(device)
pipe.set_progress_bar_config(disable=None)

generator = torch.manual_seed(0)

ds = WorldModelDataset(use_annotated_image=False)

for i in range(min(NUM_EXAMPLES, len(ds))):
    sample = ds[i]
    input_image = sample["input_image"]
    action_text = sample["action_text"]
    print("action_text:", action_text)
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

    with torch.inference_mode():
        result = pipe(
            image=input_image,
            prompt=action_text,
            generator=generator,
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=50,
        ).images[0]

    result.save(pred_save_path)
    print(f"[{i+1}/{NUM_EXAMPLES}] step={sample['step_number']} | {action_text}")
    print(f"  input     -> {input_save_path}")
    print(f"  gt        -> {gt_save_path}")
    print(f"  predicted -> {pred_save_path}")
