import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline

from diffusers import DDPMScheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading base pipeline components for standalone image training...")
pipeline = StableDiffusionPipeline.from_pretrained(
    "sd-dreambooth-library/mr-potato-head", 
    torch_dtype= torch.float32
)

net = pipeline.unet
vae = pipeline.vae

# Freeze the VAE
vae.requires_grad_(False)

# Put UNet into explicit training mode
net.to(device).train()
vae.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
     transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


dataset = datasets.ImageFolder(root="./data", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Use standard hyperparameters for fine-tuning a UNet
optimizer = AdamW(net.parameters(), lr=1e-5)
criterion = nn.MSELoss()


noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000
)

scaler = torch.cuda.amp.GradScaler()

epochs = 10

print(f"Beginning training on {len(dataset)} images...")


for epoch in range(epochs):

    running_loss = 0.0

    for imgs, _ in dataloader:

        imgs = imgs.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            latents = vae.encode(imgs).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        batch_size = latents.shape[0]

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long
        )

        noisy_latents = noise_scheduler.add_noise(
            latents,
            noise,
            timesteps
        )

        encoder_placeholder = torch.zeros(
            (batch_size, 77, 768),
            device=device,
            dtype=latents.dtype
        )

        with torch.cuda.amp.autocast():
            pred_noise = net(
                noisy_latents,
                timesteps,
                encoder_placeholder
            ).sample

            loss = criterion(pred_noise, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss / len(dataloader):.5f}"
    )

torch.save(
    net.state_dict(),
    "image_only_reconstruction_unet.pth"
)

print("Training finished!")
