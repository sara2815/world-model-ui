"""
decoder_roundtrip.py — test the decoder alone: encode an image, decode it back.

No predictor involved. This measures the CEILING of the whole rendering path:
whatever decode(pred) looks like later, it cannot be better than this. If a real
image does not survive encode->decode, then decode(pred) is uninterpretable and
there is no point looking at it.

This is the SigLIP analogue of vae_roundtrip.py, and the comparison between them
is the interesting part:

  * FLUX's VAE compresses 1280x720x3 (2.76M values) to 90x160x32 (461k) — about
    6x — and was trained specifically to reconstruct.
  * SigLIP-2 at 512/patch16 gives 32x32x1152 (1.18M floats) for a 512x512x3
    (786k) image. The embedding is LARGER than the image. There is no
    dimensional bottleneck at all.

So if SigLIP round-trips worse than the VAE, it is not a compression limit — it
is SigLIP's training objective. It learned to represent what matches captions,
and captions do not transcribe axis ticks. That distinction matters: a bottleneck
would mean "use a bigger latent"; an objective mismatch means "use an encoder
trained on text" (an OCR/document encoder) and no amount of decoder capacity will
substitute.

Run:
    python decoder_roundtrip.py --decoder checkpoints/decoder_siglip2_best.pt \
        --images output_training/gt_input.png output_training/gt_output.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wm_encoders import build_encoder, encoder_spec
from inspect_prediction import build_transform
from vae_roundtrip import find_text_crops, psnr
from train_decoder import Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "decoder_roundtrip"
N_AUTO_CROPS = 3
CROP_SIZE = 96


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--crops", type=str, nargs="*", default=None,
                    help="explicit crops as x,y,w,h (default: auto-find text)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dc = torch.load(args.decoder, map_location="cpu")
    # The decoder records which cache it was trained on; the cache records which
    # encoder built it. Resolving the encoder this way makes it impossible to
    # pair a decoder with the wrong encoder.
    cache_dir = Path(dc["encoder_cache"])
    enc_name = json.loads((cache_dir / "meta.json").read_text())["encoder"]
    spec = encoder_spec(enc_name)

    dec = Decoder(dc["d_model"], tuple(dc["grid"]), dc["image_size"],
                  dc["base_ch"]).to(DEVICE)
    dec.load_state_dict(dc["model"])
    dec.eval()
    encoder = build_encoder(enc_name, DEVICE)
    tf = build_transform(spec.image_size)

    n_emb = dc["grid"][0] * dc["grid"][1] * dc["d_model"]
    n_pix = spec.image_size ** 2 * 3
    print(f"Decoder : {args.decoder.name} (epoch {dc['epoch']}, "
          f"val L1 {dc['metrics']['l1']:.4f})")
    print(f"Encoder : {enc_name} | grid {tuple(dc['grid'])} | D={dc['d_model']}")
    print(f"Embedding {n_emb:,} floats vs image {n_pix:,} values "
          f"({n_emb / n_pix:.2f}x) — {'NO bottleneck' if n_emb >= n_pix else 'bottleneck'}")

    for path in args.images:
        img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z = encoder.encode(img)
            rec = dec(z)

        orig01 = (img[0].cpu() * 0.5 + 0.5).clamp(0, 1)
        rec01 = (rec[0].cpu() * 0.5 + 0.5).clamp(0, 1)
        p_full = psnr(rec01, orig01)
        print(f"\n=== {path.name} ===")
        print(f"  full-frame round-trip PSNR: {p_full:5.2f} dB")

        if args.crops:
            crops = [tuple(int(v) for v in c.split(",")) for c in args.crops]
        else:
            crops = [(x, y, CROP_SIZE, CROP_SIZE)
                     for x, y in find_text_crops(orig01, N_AUTO_CROPS, CROP_SIZE)]
            print(f"  auto-selected {len(crops)} high-gradient (text-like) crops")
        for cx, cy, cw, ch in crops:
            o = orig01[:, cy:cy+ch, cx:cx+cw]
            r = rec01[:, cy:cy+ch, cx:cx+cw]
            print(f"    crop ({cx},{cy}) {cw}x{ch}: PSNR {psnr(r, o):5.2f} dB")

        hwc = lambda t: t.permute(1, 2, 0).numpy()
        n = len(crops)
        fig, ax = plt.subplots(1 + n, 2, figsize=(11, 5.2 * (1 + n)))
        ax = np.atleast_2d(ax)
        ax[0, 0].imshow(hwc(orig01)); ax[0, 0].set_title("original (letterboxed)")
        ax[0, 1].imshow(hwc(rec01))
        ax[0, 1].set_title(f"encode -> decode ({p_full:.1f} dB)")
        for i, (cx, cy, cw, ch) in enumerate(crops, start=1):
            ax[i, 0].imshow(hwc(orig01[:, cy:cy+ch, cx:cx+cw]), interpolation="nearest")
            ax[i, 0].set_title(f"original crop ({cx},{cy})")
            ax[i, 1].imshow(hwc(rec01[:, cy:cy+ch, cx:cx+cw]), interpolation="nearest")
            ax[i, 1].set_title("round-trip crop")
        for a in ax.flat:
            a.axis("off")
        fig.suptitle(f"{path.name} — decoder ceiling: can a REAL image survive "
                     f"{enc_name} encode->decode?", fontsize=12)
        fig.tight_layout()
        out = OUT_DIR / f"decroundtrip_{path.stem}.png"
        fig.savefig(out, dpi=130)
        print(f"  saved -> {out}")

    print("\nRead the CROP rows. PSNR is dominated by flat background — a chart")
    print("can score 30 dB with every axis label illegible.")
    print("If the crops are unreadable here, decode(pred) will be unreadable too,")
    print("and that is a fact about the encoder, not about your world model.")


if __name__ == "__main__":
    main()