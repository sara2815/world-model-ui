import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline, DDPMScheduler

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
vae.to(device).eval()  #vae encoder compresses image by 8 and converts image to latent space

noise_scheduler = DDPMScheduler(  #handles noising and denoising according to ddpm
    num_train_timesteps=1000
)

# Ensure square 512x512 dimension, pre-process etc
image = Image.open("step_0_loaded_annotated.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
      transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

x = transform(image).unsqueeze(0).to(device) #adds batch dim

# # corrpu on the basis of amount (0-1)
# def corrupt(img_tensor, amount):
#     noise = torch.randn_like(img_tensor)  #randn to add noise of normal
#     amount = amount.view(-1, 1, 1, 1) 
#     return img_tensor * (1 - amount) + noise * amount

amounts = [0.0, 0.25, 0.5, 0.75, 1.0] # test different amount 
fig, axs = plt.subplots(2, len(amounts), figsize=(15, 6))

with torch.no_grad():

    # Encode ONCE
    clean_latents = vae.encode(
        x
    ).latent_dist.sample()
    
    
     # Stable Diffusion scales latents by scaling factor instead of hard coded value
        
    clean_latents = (
        clean_latents *
        vae.config.scaling_factor
    )
    
for i, amt in enumerate(amounts):
    
    timestep = torch.tensor(  #noise convert to timsetep
        [int(amt * 999)],
        device=device,
        dtype=torch.long
    )
    
    with torch.no_grad():
        noise = torch.randn_like(clean_latents) #randn for normal noise

        noisy_latents = noise_scheduler.add_noise( #add diffusion noise
            clean_latents,
            noise,
            timestep
        )

        # encoder_hidden_states = torch.zeros(  #dummy for text conditioning
        #     (1, 77, 768),
        #     device=device,
        #     dtype=clean_latents.dtype
        # )
        
        action = "click create button"

        text_embeddings = text_encoder(
            tokenizer(
                action,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
        )[0]
        
         # UNet predicts noise
        pred_noise = net(
            noisy_latents,
            timestep,
            # encoder_hidden_states,
            text_embeddings
        ).sample

        # Scheduler converts prediction
        scheduler_output = noise_scheduler.step(
            pred_noise,
            timestep.item(),
            noisy_latents
        )

        denoised_latents = (
            scheduler_output.pred_original_sample
        )
        
        denoised_latents = (
            denoised_latents /
            vae.config.scaling_factor
        )

        decoded_img = vae.decode(
            denoised_latents
        ).sample

        decoded_img = (
            decoded_img / 2 + 0.5
        ).clamp(0, 1)

        noisy_preview = vae.decode(
            noisy_latents /
            vae.config.scaling_factor
        ).sample

        noisy_preview = (
            noisy_preview / 2 + 0.5
        ).clamp(0, 1)

    noisy_np = (
        noisy_preview
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    pred_np = (
        decoded_img
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    axs[0, i].imshow(noisy_np)
    axs[0, i].set_title(
        f"Noisy (Amt: {amt:.2f})"
    )
    axs[0, i].axis("off")

    axs[1, i].imshow(pred_np)
    axs[1, i].set_title("Prediction")
    axs[1, i].axis("off")

plt.tight_layout()
plt.show()
