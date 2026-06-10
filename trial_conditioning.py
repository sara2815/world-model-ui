"""
World-model inference: given a prev screenshot + action text, predict the next screenshot.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler


# ---------------------------------------------------------------------------
# Must match the patch used during training
# ---------------------------------------------------------------------------

def patch_unet_input_channels(unet, new_in_channels=8):
    old_conv = unet.conv_in
    old_in   = old_conv.in_channels
    out_ch   = old_conv.out_channels
    new_conv = nn.Conv2d(
        new_in_channels, out_ch,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight[:, :old_in] = old_conv.weight.clone()
        new_conv.weight[:, old_in:] = 0.0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    unet.config.in_channels = new_in_channels
    return unet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PREV_IMAGE_PATH = "step_0_loaded_annotated.png"
ACTION_TEXT     = "click create button"
WEIGHTS_PATH    = "world_model_unet.pth"
NOISE_AMOUNTS   = [0.0, 0.25, 0.5, 0.75, 1.0]   # visualise across noise levels


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading pipeline...")
pipeline = StableDiffusionPipeline.from_pretrained(
    "sd-dreambooth-library/mr-potato-head",
    torch_dtype=torch.float32,
)

net          = pipeline.unet
vae          = pipeline.vae
tokenizer    = pipeline.tokenizer
text_encoder = pipeline.text_encoder

net = patch_unet_input_channels(net, new_in_channels=8)

print("Loading trained weights...")
net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

net.to(device).eval()
vae.to(device).eval()
text_encoder.to(device).eval()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# ---------------------------------------------------------------------------
# Pre-process prev screenshot
# ---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

prev_img = Image.open(PREV_IMAGE_PATH).convert("RGB")
prev_tensor = transform(prev_img).unsqueeze(0).to(device)   # (1, 3, 512, 512)

# Encode prev frame once — reused as conditioning for every noise level
with torch.no_grad():
    prev_latents = vae.encode(prev_tensor).latent_dist.sample() * vae.config.scaling_factor

    # Text conditioning
    tok = tokenizer(
        [ACTION_TEXT],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(tok.input_ids.to(device))[0]


# ---------------------------------------------------------------------------
# Inference across noise amounts
# ---------------------------------------------------------------------------

fig, axs = plt.subplots(2, len(NOISE_AMOUNTS), figsize=(15, 6))

for i, amt in enumerate(NOISE_AMOUNTS):
    timestep = torch.tensor([int(amt * 999)], device=device, dtype=torch.long)

    with torch.no_grad():
        # Start from pure noise and denoise toward the predicted next frame
        noisy_next = torch.randn_like(prev_latents)
        if amt < 1.0:
            # For partial amounts, add noise to a zero latent baseline
            zero_latents = torch.zeros_like(prev_latents)
            noisy_next = noise_scheduler.add_noise(zero_latents, noisy_next, timestep)

        # Concatenate [noisy_next | prev] → (1, 8, H, W)
        model_input = torch.cat([noisy_next, prev_latents], dim=1)

        pred_noise = net(model_input, timestep, text_embeddings).sample

        scheduler_out = noise_scheduler.step(pred_noise, timestep.item(), noisy_next)
        denoised = scheduler_out.pred_original_sample / vae.config.scaling_factor

        predicted_img = vae.decode(denoised).sample
        predicted_img = (predicted_img / 2 + 0.5).clamp(0, 1)

        # Also decode the noisy input for the top row
        noisy_preview = vae.decode(noisy_next / vae.config.scaling_factor).sample
        noisy_preview = (noisy_preview / 2 + 0.5).clamp(0, 1)

    def to_np(t):
        return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    axs[0, i].imshow(to_np(noisy_preview))
    axs[0, i].set_title(f"Noisy input\n(amt={amt:.2f})")
    axs[0, i].axis("off")

    axs[1, i].imshow(to_np(predicted_img))
    axs[1, i].set_title("Predicted next frame")
    axs[1, i].axis("off")

plt.suptitle(f'Action: "{ACTION_TEXT}"', fontsize=12)
plt.tight_layout()
plt.savefig("world_model_prediction.png", dpi=150)
plt.show()
print("Saved → world_model_prediction.png")