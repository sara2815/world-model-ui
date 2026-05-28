import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Downloading/Loading Mr. Potato Head model from Hugging Face...")
# downloading example pipeline from HF tutotiral
pipeline = StableDiffusionPipeline.from_pretrained(
    "sd-dreambooth-library/mr-potato-head", 
    torch_dtype=torch.float32
)

net = pipeline.unet  # u-net
vae = pipeline.vae

#load trained weights
print("Loading your custom trained UNet weights...")
net.load_state_dict(torch.load("image_only_reconstruction_unet.pth", map_location=device))

net.to(device).eval()
vae.to(device).eval()  #vae encoder compresses image by 8

# Ensure square 512x512 dimension
image = Image.open("step_0_loaded_annotated.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

x = transform(image).unsqueeze(0).to(device) 

# corrpu on the basis of amount (0-1)
def corrupt(img_tensor, amount):
    noise = torch.rand_like(img_tensor)
    amount = amount.view(-1, 1, 1, 1) 
    return img_tensor * (1 - amount) + noise * amount

amounts = [0.0, 0.25, 0.5, 0.75, 1.0] # test different amount 
fig, axs = plt.subplots(2, len(amounts), figsize=(15, 6))

for i, amt in enumerate(amounts):
    amt_tensor = torch.tensor([amt], device=device)
    noisy_img = corrupt(x, amt_tensor)
    
    with torch.no_grad():
        # Stable Diffusion scales latents by a scaling factor constant (0.18215)
        latents = vae.encode(noisy_img).latent_dist.sample()
        latents = latents * 0.18215  # Shape becomes: [1, 4, 64, 64]
        
        # Setup mock diffusion conditions
        timestep = torch.tensor([int(amt * 999)], device=device, dtype=torch.long)
        encoder_hidden_states = torch.zeros((1, 77, 768), device=device, dtype=latents.dtype)
        
        # Predict 
        pred_output = net(latents, timestep, encoder_hidden_states)
        pred_latents = pred_output.sample   
        
        pred_latents = pred_latents / 0.18215
        decoded_img = vae.decode(pred_latents).sample
        # Clamp to expected RGB image range
        decoded_img = (decoded_img / 2 + 0.5).clamp(0, 1)

    # Re-arrange shapes for Matplotlib [Batch, Channels, H, W] -> [H, W, Channels]
    noisy_np = noisy_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_np = decoded_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Plot Noisy Input Row
    axs[0, i].imshow(noisy_np)
    axs[0, i].set_title(f"Noisy (Amt: {amt:.2f})")
    axs[0, i].axis("off")
    
    # Plot Model Prediction Row
    axs[1, i].imshow(pred_np)
    axs[1, i].set_title("Prediction")
    axs[1, i].axis("off")
    
plt.tight_layout()
plt.show()