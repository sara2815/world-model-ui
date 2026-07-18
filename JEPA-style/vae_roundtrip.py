"""
vae_roundtrip.py — does ANY latent representation preserve your chart text?

This is not part of the JEPA pipeline. It is a 20-minute experiment that answers
a question the JEPA pipeline structurally cannot: SigLIP is one-way, so you can
never see what its encoding threw away. FLUX's VAE has both halves, so we can
compress a real screenshot and decompress it and LOOK.

WHY THIS IS AN UPPER BOUND
--------------------------
FLUX's VAE is a *generous* stand-in for the general question "does compressing a
1280x720 chart to a small latent grid keep the text?":
  * it compresses ~8x spatially; SigLIP-2 at 512/patch16 effectively compresses
    ~16x relative to the native screenshot
  * its decoder was trained on billions of images specifically to reconstruct
    well; a from-scratch decoder over frozen SigLIP embeddings would be far worse
So if FLUX's VAE already smears your axis labels, no decoder over SigLIP
embeddings will recover them, and the whole pixel-prediction framing is bounded
regardless of RoPE ids, change masks, or architecture.

NO SCALING FACTORS NEEDED
-------------------------
The `shift_factor` / `scaling_factor` dance exists to normalize latents for the
*diffusion* model. A pure encode->decode round-trip does not need them: whatever
scaling you apply on the way in you undo on the way out, so it cancels. We call
vae.encode().latent_dist.mode() -> vae.decode() directly and sidestep the whole
question. `.mode()` rather than `.sample()` because we want the best case, not a
noisy draw.

CONTROL
-------
A plain bicubic downsample-then-upsample at the same spatial factor is included.
If the VAE is no better than naive resampling, it is contributing nothing beyond
resolution loss — which is itself worth knowing.

Hardware: forced to fp32. V100 (sm_70) has no bfloat16, and this is one image, so
speed is irrelevant.

Run:
    python vae_roundtrip.py --images output_training/gt_output.png
    python vae_roundtrip.py --images output_training/gt_input.png output_training/gt_output.png
    python vae_roundtrip.py --images shot.png --crops 300,200,160,90 900,400,160,90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "vae_roundtrip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_AUTO_CROPS = 3
CROP_SIZE = 96


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB on [0,1] tensors. >30 is good, >40 is near-lossless."""
    mse = F.mse_loss(a, b).item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def find_text_crops(img: torch.Tensor, n: int, size: int) -> list[tuple[int, int]]:
    """Locate the n highest-gradient-energy regions — a proxy for text.

    Chart text is the highest-frequency content in a UI screenshot, so gradient
    energy finds it without needing OCR. Returns (x, y) top-left corners, spread
    apart so we do not return n overlapping views of the same label.
    """
    g = img.mean(dim=0, keepdim=True)
    gx = (g[:, :, 1:] - g[:, :, :-1]).abs()
    gy = (g[:, 1:, :] - g[:, :-1, :]).abs()
    energy = torch.zeros_like(g)
    energy[:, :, :-1] += gx
    energy[:, :-1, :] += gy
    pooled = F.avg_pool2d(energy.unsqueeze(0), size, stride=size // 2)[0, 0]

    picks: list[tuple[int, int]] = []
    flat = pooled.flatten().argsort(descending=True)
    for idx in flat.tolist():
        r, c = divmod(idx, pooled.shape[1])
        y, x = r * (size // 2), c * (size // 2)
        if any(abs(x - px) < size and abs(y - py) < size for px, py in picks):
            continue
        if y + size > img.shape[1] or x + size > img.shape[2]:
            continue
        picks.append((x, y))
        if len(picks) == n:
            break
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--crops", type=str, nargs="*", default=None,
                    help="explicit crops as x,y,w,h (default: auto-find text)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from diffusers import Flux2KleinPipeline

    print(f"Loading VAE from {args.model} (fp32 — V100 has no bf16)...")
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    vae = pipe.vae.eval().to(DEVICE)
    vae.requires_grad_(False)
    spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    print(f"VAE spatial compression: {spatial}x  latent channels: "
          f"{vae.config.latent_channels}")

    to_tensor = transforms.ToTensor()

    for path in args.images:
        img01 = to_tensor(Image.open(path).convert("RGB"))
        _, H, W = img01.shape
        # crop to a multiple of the compression factor so nothing is padded
        H2, W2 = (H // spatial) * spatial, (W // spatial) * spatial
        img01 = img01[:, :H2, :W2]
        x = (img01 * 2 - 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            lat = vae.encode(x).latent_dist.mode()   # no scaling: it cancels
            rec = vae.decode(lat).sample
            # CONTROL: naive bicubic at the same spatial factor
            small = F.interpolate(x, scale_factor=1 / spatial, mode="bicubic",
                                  align_corners=False, antialias=True)
            bic = F.interpolate(small, size=(H2, W2), mode="bicubic",
                                align_corners=False)

        rec01 = (rec * 0.5 + 0.5).clamp(0, 1)[0].cpu()
        bic01 = (bic * 0.5 + 0.5).clamp(0, 1)[0].cpu()
        p_vae, p_bic = psnr(rec01, img01), psnr(bic01, img01)

        print(f"\n=== {path.name}  ({W2}x{H2} -> latent "
              f"{lat.shape[-2]}x{lat.shape[-1]}x{lat.shape[1]}) ===")
        print(f"  VAE round-trip PSNR : {p_vae:5.2f} dB")
        print(f"  bicubic control PSNR: {p_bic:5.2f} dB   "
              f"({'VAE better' if p_vae > p_bic else 'VAE NO BETTER than resampling'})")

        if args.crops:
            crops = []
            for c in args.crops:
                cx, cy, cw, ch = (int(v) for v in c.split(","))
                crops.append((cx, cy, cw, ch))
        else:
            crops = [(x0, y0, CROP_SIZE, CROP_SIZE)
                     for x0, y0 in find_text_crops(img01, N_AUTO_CROPS, CROP_SIZE)]
            print(f"  auto-selected {len(crops)} high-gradient (text-like) crops")

        for cx, cy, cw, ch in crops:
            o = img01[:, cy:cy+ch, cx:cx+cw]
            r = rec01[:, cy:cy+ch, cx:cx+cw]
            print(f"    crop ({cx},{cy}) {cw}x{ch}: PSNR {psnr(r, o):5.2f} dB")

        # ── figure ───────────────────────────────────────────────────────────
        n = len(crops)
        fig, ax = plt.subplots(1 + n, 3, figsize=(15, 5 * (1 + n)))
        ax = np.atleast_2d(ax)
        hwc = lambda t: t.permute(1, 2, 0).numpy()
        ax[0, 0].imshow(hwc(img01)); ax[0, 0].set_title("original")
        ax[0, 1].imshow(hwc(rec01)); ax[0, 1].set_title(f"VAE round-trip ({p_vae:.1f} dB)")
        ax[0, 2].imshow(hwc(bic01)); ax[0, 2].set_title(f"bicubic {spatial}x control ({p_bic:.1f} dB)")
        for i, (cx, cy, cw, ch) in enumerate(crops, start=1):
            ax[i, 0].imshow(hwc(img01[:, cy:cy+ch, cx:cx+cw]), interpolation="nearest")
            ax[i, 0].set_title(f"original crop ({cx},{cy})")
            ax[i, 1].imshow(hwc(rec01[:, cy:cy+ch, cx:cx+cw]), interpolation="nearest")
            ax[i, 1].set_title("VAE round-trip")
            ax[i, 2].imshow(hwc(bic01[:, cy:cy+ch, cx:cx+cw]), interpolation="nearest")
            ax[i, 2].set_title("bicubic control")
        for a in ax.flat:
            a.axis("off")
        fig.suptitle(f"{path.name} — can you still read the text after "
                     f"{spatial}x latent compression?", fontsize=12)
        fig.tight_layout()
        out = OUT_DIR / f"roundtrip_{path.stem}.png"
        fig.savefig(out, dpi=130)
        print(f"  saved -> {out}")

    print("\nRead the CROP rows, not the full images — full-frame reconstructions")
    print("always look fine. The question is whether an axis label is still")
    print("legible. If it is not, pixel-space prediction is bounded here no")
    print("matter which model you train, and that is a result worth reporting.")


if __name__ == "__main__":
    main()