import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading base pipeline components for standalone image training...")
pipeline = StableDiffusionPipeline.from_pretrained(
    "sd-dreambooth-library/mr-potato-head", 
    torch_dtype=torch.float32
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
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(root="./data", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Use standard hyperparameters for fine-tuning a UNet
optimizer = AdamW(net.parameters(), lr=1e-5)
criterion = nn.MSELoss()

def corrupt(latents, noise, timesteps):
    # Map 0-999 timesteps to a percentage scalar for noise blending
    amount = (timesteps.float() / 999.0).view(-1, 1, 1, 1)
    return latents * (1 - amount) + noise * amount

epochs = 10
print(f"Beginning training on {len(dataset)} individual images...")

for epoch in range(epochs):
    running_loss = 0.0
    
    for step, (imgs, _) in enumerate(dataloader):
        # imgs shape: [Batch, 3, 512, 512]
        imgs = imgs.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            # Compress your raw pixels down into 4-channel latent matrices
            latents = vae.encode(imgs).latent_dist.sample()
            latents = latents * 0.18215 # Shape: [Batch, 4, 64, 64]
            
        # Create random noise profiles and timestamps for the batch
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
        
        # Corrupt the latents based on the selected timesteps
        noisy_latents = corrupt(latents, noise, timesteps)
        
        # predicting 
        # Pass a blank placeholder for encoder_hidden_states to run unconditionally
        encoder_placeholder = torch.zeros((batch_size, 77, 768), device=device, dtype=latents.dtype)
        
        pred_output = net(noisy_latents, timesteps, encoder_placeholder)
        pred_noise = pred_output.sample
        
        # how close the model was to identifying the added noise
        loss = criterion(pred_noise, noise)
        
        # backprop
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(dataloader):.5f}")

# Save the updated image-only weights
torch.save(net.state_dict(), "image_only_reconstruction_unet.pth")
print("Training finished! Weights saved locally to 'image_only_reconstruction_unet.pth'")