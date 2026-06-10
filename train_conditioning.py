import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers import StableDiffusionPipeline, DDPMScheduler

from dataset import DashboardQAScreenshotTransitionDataset

# ---------------------------------------------------------------------------
# UNet input channel patch  (4 → 8: 4 noisy-next-latents + 4 prev-latents)
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

DATA_ROOT            = r"C:\Users\adyes\Downloads\results_gemini_pro_25_screenshot"
BATCH_SIZE           = 4
IMAGE_SIZE           = (512, 512)
EPOCHS               = 10
LR                   = 1e-5
SAVE_PATH            = "world_model_unet.pth"
ACTION_FIELD         = "action_full"
VISUAL_CHANGE_FILTER = "any"


# ---------------------------------------------------------------------------
# Setup 
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading base pipeline...")
pipeline = StableDiffusionPipeline.from_pretrained(
    "sd-dreambooth-library/mr-potato-head",
    torch_dtype=torch.float32,
)

net          = pipeline.unet
vae          = pipeline.vae
tokenizer    = pipeline.tokenizer
text_encoder = pipeline.text_encoder

net = patch_unet_input_channels(net, new_in_channels=8)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

net.to(device).train()
vae.to(device).eval()
text_encoder.to(device).eval()

dataset = DashboardQAScreenshotTransitionDataset(
    root=DATA_ROOT,
    action_field=ACTION_FIELD,
    image_size=IMAGE_SIZE,
    value_range="neg1_1",
    split="train",
    visual_change_filter=VISUAL_CHANGE_FILTER,
    skip_missing_files=True,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,    # must be 0 on Windows
    pin_memory=False, # only useful with CUDA
)

print(f"Dataset: {len(dataset)} transitions")
for k, v in sorted(dataset.build_stats.items()):
    print(f"  {k}: {v}")

optimizer       = AdamW(net.parameters(), lr=LR)
criterion       = nn.MSELoss()
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# ---------------------------------------------------------------------------
# Training  (guarded for Windows multiprocessing safety)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for batch in dataloader:
            prev_imgs  = batch["source"].to(device)
            next_imgs  = batch["target"].to(device)
            actions    = batch["action_text"]
            batch_size = prev_imgs.shape[0]

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                prev_latents = vae.encode(prev_imgs).latent_dist.sample() * vae.config.scaling_factor
                next_latents = vae.encode(next_imgs).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(next_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device, dtype=torch.long,
            )
            noisy_next  = noise_scheduler.add_noise(next_latents, noise, timesteps)
            model_input = torch.cat([noisy_next, prev_latents], dim=1)

            with torch.no_grad():
                tok = tokenizer(
                    list(actions),
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                text_embeddings = text_encoder(tok.input_ids.to(device))[0]

            pred_noise = net(model_input, timesteps, text_embeddings).sample
            loss       = criterion(pred_noise, noise)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {running_loss / len(dataloader):.5f}")

    torch.save(net.state_dict(), SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")