import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import Flux2KleinPipeline
from diffusers.training_utils import compute_loss_weighting_for_sd3
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from PIL import Image

try:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import calculate_shift
except ImportError:
    def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from world_model_ui.data.dataset import WorldModelDataset

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID   = "black-forest-labs/FLUX.2-klein-base-4B"
OUTPUT_DIR          = Path(__file__).resolve().parent / "finetuned_model"
OUTPUT_TRAINING_DIR = Path(__file__).resolve().parent / "output_training"
VIS_EVERY = 2  # save a generated sample every N epochs (+ always at final)
IMG_H = 720   # 1280×720 preserves 16:9 from native 1920×1080; both divisible by 16
IMG_W = 1280
BATCH_SIZE = 4
NUM_EPOCHS = 15
LR         = 1e-4
GRAD_CLIP  = 1.0
LORA_RANK  = 16
SAVE_EVERY = 500  # steps

device = "cuda"
dtype  = torch.bfloat16

# ── Load pipeline (same as inference.py) ──────────────────────────────────
pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
pipe = pipe.to(device)
#breakpoint()
vae          = pipe.vae.eval()
text_encoder = pipe.text_encoder.eval()  # Qwen3ForCausalLM
tokenizer    = pipe.tokenizer            # Qwen2Tokenizer
transformer  = pipe.transformer
scheduler    = pipe.scheduler

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

TEXT_MAX_LEN = 512  # cap Qwen3's large context window to prompt-appropriate length

# Gradient checkpointing: slower per step but much lower GPU memory usage
transformer.enable_gradient_checkpointing()

# ── LoRA on transformer (consistent with dreambooth/train_dreambooth_lora_flux.py)
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    init_lora_weights="gaussian",
    target_modules=[
        # attention
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
        # feedforward
        "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
    ],
    lora_dropout=0.0,
    bias="none",
)
transformer = get_peft_model(transformer, lora_config)
transformer.print_trainable_parameters()

# ── Dataset / DataLoader ───────────────────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
])
ds = WorldModelDataset(
    use_annotated_image=True,
    transform=img_tf,
    target_transform=img_tf,
)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=2, pin_memory=True)

# ── Optimizer ──────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    [p for p in transformer.parameters() if p.requires_grad],
    lr=LR, weight_decay=1e-2,
)

# ── Helpers ────────────────────────────────────────────────────────────────
vae_shift = getattr(vae.config, "shift_factor",   0.0)
vae_scale = getattr(vae.config, "scaling_factor", 1.0)

VAE_PATCH = getattr(vae.config, "patch_size", [2, 2])
VAE_PATCH_H, VAE_PATCH_W = (VAE_PATCH if isinstance(VAE_PATCH, (list, tuple)) else (VAE_PATCH, VAE_PATCH))


def build_prompt(action_text: str) -> str:
    return (
        "Given the following UI image and action, predict the resulting UI image after performing the action:\n"
        f"{action_text}"
    )

def make_img_ids(H: int, W: int) -> torch.Tensor:
    """FLUX-style 2-D RoPE position IDs for a latent grid.
    H, W are latent spatial dims; returns [(H/ph)*(W/pw), 3] on CPU."""
    gh, gw = H // VAE_PATCH_H, W // VAE_PATCH_W
    ids = torch.zeros(gh * gw, 3)
    ids[:, 1] = torch.arange(gh).repeat_interleave(gw)
    ids[:, 2] = torch.arange(gw).repeat(gh)
    return ids


@torch.no_grad()
def encode_image(imgs: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [-1,1] → VAE latents [B,C,h,w]"""
    latents = vae.encode(imgs.to(dtype=dtype)).latent_dist.sample()
    return (latents - vae_shift) * vae_scale


@torch.no_grad()
def encode_text(prompts: list[str]) -> torch.Tensor:
    """Encode text with Qwen3. Returns prompt_embeds [B, L, D]."""
    tok = tokenizer(
        prompts,
        padding="max_length",
        max_length=TEXT_MAX_LEN,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    out = text_encoder(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        output_hidden_states=True,
    )
    return out.hidden_states[-1].to(dtype=dtype)  # [B, L, D]


_to_pil = transforms.ToPILImage()

def _unnorm(t: torch.Tensor) -> Image.Image:
    """Undo Normalize([0.5],[0.5]) and return PIL image."""
    return _to_pil((t.float().cpu() * 0.5 + 0.5).clamp(0, 1))


def concat_packed_conditioning(target_tokens: torch.Tensor, source_tokens: torch.Tensor) -> torch.Tensor:
    """Append source image tokens after target-noisy tokens for edit conditioning."""
    return torch.cat([target_tokens, source_tokens], dim=1)


def concat_img_ids(target_ids: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
    """Match concatenated image tokens with concatenated FLUX image position ids."""
    if target_ids.dim() == 2:
        return torch.cat([target_ids, source_ids], dim=0)
    if target_ids.dim() == 3:
        return torch.cat([target_ids, source_ids], dim=1)
    raise ValueError(f"Unexpected img_ids rank: {target_ids.dim()}")


@torch.no_grad()
def save_visual_sample(epoch: int):
    """Decode train[0] GT and run inference; save both under output_training/."""
    transformer.eval()
    sample = ds[0]
    input_pil = _unnorm(sample["input_image"])
    gt_pil    = _unnorm(sample["output_image"])
    prompt    = build_prompt(sample["action_text"])

    gen = pipe(
        prompt=prompt,
        image=input_pil,
        height=IMG_H,
        width=IMG_W,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]

    gen.save(OUTPUT_TRAINING_DIR / f"gen_epoch_{epoch:03d}.png")
    print(f"[vis] saved gen_epoch_{epoch:03d}.png  prompt: {prompt!r}")
    transformer.train()


# ── Training loop ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# Save GT once before any training
_gt_sample = ds[0]
_unnorm(_gt_sample["input_image"]).save(OUTPUT_TRAINING_DIR / "gt_input.png")
_unnorm(_gt_sample["output_image"]).save(OUTPUT_TRAINING_DIR / "gt_output.png")
print(f"[vis] GT images saved to {OUTPUT_TRAINING_DIR}")

transformer.train()
global_step = 0

# Set scheduler to training timesteps so scheduler.sigmas has num_train_timesteps+1
# entries. save_visual_sample() calls pipe() which overwrites this with inference
# timesteps (28 steps), so we must restore after each vis call.
_vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
_image_seq_len = (IMG_H // _vae_scale // VAE_PATCH_H) * (IMG_W // _vae_scale // VAE_PATCH_W)
_mu = calculate_shift(
    _image_seq_len,
    scheduler.config.get("base_image_seq_len", 256),
    scheduler.config.get("max_image_seq_len", 4096),
    scheduler.config.get("base_shift", 0.5),
    scheduler.config.get("max_shift", 1.16),
)
scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device, mu=_mu)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for step, batch in enumerate(dl):
        input_imgs  = batch["input_image"].to(device)   # [B,3,H,W]
        output_imgs = batch["output_image"].to(device)  # [B,3,H,W]  GT
        prompts     = [build_prompt(text) for text in batch["action_text"]]
        B = input_imgs.shape[0]
        
        input_imgs = input_imgs.to(
            device=pipe.vae.device,
            dtype=next(pipe.vae.parameters()).dtype,
        )
        target_imgs = output_imgs.to(device=pipe.vae.device,dtype=next(pipe.vae.parameters()).dtype,)
        
        # Encode images

        source_latents = pipe._encode_vae_image(image=input_imgs, generator=None)
        target_latents = pipe._encode_vae_image(image=target_imgs, generator=None)

        noise = torch.randn_like(target_latents)

        u = torch.normal(mean=0.0, std=1.0, size=(B,), device=device, dtype=dtype)
        u = torch.sigmoid(u)

        indices = (u * scheduler.config.num_train_timesteps).long().cpu()
        timesteps = scheduler.timesteps[indices].to(device=device)
        sigmas = scheduler.sigmas[indices].to(device=device, dtype=target_latents.dtype)

        sigmas_ = sigmas.view(B, 1, 1, 1)
        noisy = (1.0 - sigmas_) * target_latents + sigmas_ * noise
        #breakpoint()
        packed_noisy_target = pipe._pack_latents(noisy)
        packed_source = pipe._pack_latents(source_latents)
        packed_input = concat_packed_conditioning(packed_noisy_target, packed_source)

        target_img_ids = pipe._prepare_latent_ids(noisy).to(device)
        source_img_ids = pipe._prepare_latent_ids(source_latents).to(device)
        img_ids = concat_img_ids(target_img_ids, source_img_ids)

        velocity_target = noise - target_latents
        packed_target = pipe._pack_latents(velocity_target)

        prompt_embeds, txt_ids = pipe.encode_prompt(
            prompts,
            device=device,
            max_sequence_length=TEXT_MAX_LEN,
        )

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        txt_ids = txt_ids.to(device=device)

        pred = transformer(
            hidden_states=packed_input,
            timestep=timesteps / 1000,
            encoder_hidden_states=prompt_embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
        pred = pred[:, : packed_target.shape[1], :]

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="logit_normal",
            sigmas=sigmas,
        )

        loss = ((pred.float() - packed_target.float()) ** 2).reshape(B, -1)
        loss = (weighting.view(B, 1).float() * loss).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in transformer.parameters() if p.requires_grad], GRAD_CLIP
        )
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        global_step += 1

        if global_step % 10 == 0:
            print(f"epoch {epoch+1}/{NUM_EPOCHS}  step {global_step}  "
                  f"loss {loss.item():.4f}")

        if global_step % SAVE_EVERY == 0:
            ckpt = OUTPUT_DIR / f"checkpoint-{global_step}"
            Flux2KleinPipeline.save_lora_weights(
                save_directory=str(ckpt),
                transformer_lora_layers=get_peft_model_state_dict(transformer),
            )
            print(f"Checkpoint → {ckpt}")

    avg_loss = epoch_loss / len(dl)
    print(f"── epoch {epoch+1} done  avg_loss={avg_loss:.4f}")

    is_final = (epoch == NUM_EPOCHS - 1)
    if (epoch + 1) % VIS_EVERY == 0 or is_final:
        save_visual_sample(epoch + 1)
        scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device, mu=_mu)

# ── Save final LoRA weights ────────────────────────────────────────────────
Flux2KleinPipeline.save_lora_weights(
    save_directory=str(OUTPUT_DIR / "lora_weights"),
    transformer_lora_layers=get_peft_model_state_dict(transformer),
)
print(f"Saved → {OUTPUT_DIR / 'lora_weights'}")