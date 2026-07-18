"""
train_decoder.py — invert the frozen encoder: z tokens -> pixels.

The world model predicts embeddings, so there is nothing to look at. This trains
a decoder on the cached embeddings so predictions can be rendered.

WHY THIS IS PLAUSIBLE (it was not obvious)
------------------------------------------
SigLIP-2 at 512/patch16 gives 32x32x1152 = 1,179,648 floats for a 512x512x3 =
786,432-value image. The embedding is LARGER than the image it encodes. There is
no dimensional bottleneck. Whether the text is recoverable depends entirely on
whether SigLIP's training objective (match images to captions) kept it — captions
do not transcribe axis ticks, so it may not have. That is an empirical question
and this decoder answers it.

L1, NOT GAN — DELIBERATELY
--------------------------
A perceptual/adversarial loss would produce sharper output, and that is exactly
the problem: a GAN hallucinates plausible-looking text. For a chart world model,
a confident wrong "4.7" is far worse than an honest blur, because blur correctly
communicates "this information is not in the embedding". L1 stays faithful to
what the representation actually contains.

WHAT IT IS TRAINED ON
---------------------
`z_src -> src_image`: a pure inversion task over every cached pair. At inference
the predictor's output `pred` is fed in, which is slightly off-distribution (a
predicted embedding, not a real encoder output). That is why the visualizer ALWAYS
decodes z_tgt alongside pred: decode(z_tgt) is the ceiling. If it is mush, the
pipeline is at its limit and decode(pred) says nothing about the predictor.

Run:
    python train_decoder.py --cache cache/siglip2
    python train_decoder.py --cache cache/siglip2 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import encoder_spec
from inspect_prediction import build_transform, dataset_index_map
from train_predictor import SEED, VAL_FRAC, EmbeddingBank

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CH = 256          # channels after the 1x1 reduction from d_model
BATCH_SIZE = 8
NUM_EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0
SAVE_EVERY = 5
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
EVAL_SUBSET = 256


class UpBlock(nn.Module):
    """2x nearest-neighbour upsample + two convs.

    Nearest+conv rather than ConvTranspose2d: transposed convolutions produce
    checkerboard artifacts at exactly the frequency of fine text, which is the
    one thing this decoder exists to be judged on.
    """

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(F.interpolate(x, scale_factor=2, mode="nearest"))


class Decoder(nn.Module):
    """[B, N, D] tokens -> [B, 3, S, S] image in [-1, 1].

    Tokens are unflattened row-major to (rows, cols) — the same ordering the
    change masks use and that precompute's alignment test verified against the
    encoder. Getting this wrong would transpose every output image.
    """

    def __init__(self, d_model: int, grid: tuple[int, int], image_size: int,
                 base_ch: int = BASE_CH) -> None:
        super().__init__()
        self.grid = grid
        n_up = int(math.log2(image_size // grid[0]))
        assert 2 ** n_up * grid[0] == image_size, (
            f"grid {grid[0]} must reach {image_size} by exact doublings"
        )
        self.stem = nn.Sequential(
            nn.Conv2d(d_model, base_ch, 1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )
        chans = [max(base_ch >> i, 32) for i in range(n_up + 1)]
        self.ups = nn.ModuleList(
            [UpBlock(chans[i], chans[i + 1]) for i in range(n_up)]
        )
        self.head = nn.Conv2d(chans[-1], 3, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        b, n, d = z.shape
        r, c = self.grid
        x = z.transpose(1, 2).reshape(b, d, r, c)   # row-major unflatten
        x = self.stem(x)
        for up in self.ups:
            x = up(x)
        return torch.tanh(self.head(x))              # [-1, 1], matches the data


class PixelBank:
    """Pairs cached z_src rows with their source images from the dataset."""

    def __init__(self, cache_dir: Path) -> None:
        self.bank = EmbeddingBank(cache_dir, "structured")
        meta = json.loads((cache_dir / "meta.json").read_text())
        self.spec = encoder_spec(meta["encoder"])
        tf = build_transform(self.spec.image_size)
        self.ds = WorldModelDataset(use_annotated_image=True, transform=tf,
                                    target_transform=tf)
        self.idx_map = dataset_index_map(self.bank, self.ds)
        self.m = self.bank.m

    def fetch(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """(z_src [B,N,D], src_image [B,3,S,S]) — both on DEVICE, fp32."""
        i = np.asarray(idx.cpu().numpy())
        z = torch.from_numpy(np.array(self.bank._z_src[i])).float().to(DEVICE)
        imgs = torch.stack([self.ds[self.idx_map[j]]["input_image"] for j in i.tolist()])
        return z, imgs.to(DEVICE)


@torch.no_grad()
def evaluate(dec: Decoder, pb: PixelBank, idx: Tensor) -> dict:
    """Chunked L1 + PSNR. PSNR is reported because dB is interpretable: >30 is
    good, but see the note in the final print — it is dominated by background."""
    dec.eval()
    l1_sum, mse_sum, n = 0.0, 0.0, 0
    for start in range(0, idx.numel(), BATCH_SIZE):
        b = idx[start : start + BATCH_SIZE]
        z, img = pb.fetch(b)
        rec = dec(z)
        l1_sum += F.l1_loss(rec, img, reduction="sum").item()
        mse_sum += F.mse_loss((rec + 1) / 2, (img + 1) / 2, reduction="sum").item()
        n += img.numel()
    mse = mse_sum / n
    return {"l1": l1_sum / n, "psnr": 10 * math.log10(1.0 / max(mse, 1e-12))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    pb = PixelBank(args.cache)
    spec = pb.spec
    grid = pb.bank.grid
    print(f"Cache: {pb.m} pairs | grid {grid} | D={pb.bank.d_model} | "
          f"image {spec.image_size}^2")
    print(f"Embedding is {grid[0]*grid[1]*pb.bank.d_model:,} floats vs "
          f"{spec.image_size**2*3:,} pixel values — no dimensional bottleneck.")

    # Same split as the predictor, so a decoder never sees the predictor's val
    # images during its own training.
    torch.manual_seed(SEED)
    perm = torch.randperm(pb.m)
    n_val = max(1, int(round(VAL_FRAC * pb.m)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tr_eval = train_idx[: min(EVAL_SUBSET, train_idx.numel())]
    va_eval = val_idx[: min(EVAL_SUBSET, val_idx.numel())]

    dec = Decoder(pb.bank.d_model, grid, spec.image_size).to(DEVICE)
    print(f"Decoder: {sum(p.numel() for p in dec.parameters())/1e6:.1f}M params")

    opt = torch.optim.AdamW(dec.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = math.ceil(train_idx.numel() / BATCH_SIZE) * args.epochs
    warm = int(WARMUP_RATIO * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        dec.train()
        order = train_idx[torch.randperm(train_idx.numel())]
        for start in range(0, order.numel(), BATCH_SIZE):
            b = order[start : start + BATCH_SIZE]
            z, img = pb.fetch(b)
            loss = F.l1_loss(dec(z), img)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()

        tr, va = evaluate(dec, pb, tr_eval), evaluate(dec, pb, va_eval)
        print(f"epoch {epoch:3d} | lr {sched.get_last_lr()[0]:.2e} | "
              f"train L1 {tr['l1']:.4f} psnr {tr['psnr']:5.2f} | "
              f"val L1 {va['l1']:.4f} psnr {va['psnr']:5.2f}", flush=True)

        payload = {"model": dec.state_dict(), "epoch": epoch, "metrics": va,
                   "d_model": pb.bank.d_model, "grid": list(grid),
                   "image_size": spec.image_size, "base_ch": BASE_CH,
                   "encoder_cache": str(args.cache)}
        if va["l1"] < best:
            best = va["l1"]
            torch.save(payload, CKPT_DIR / f"decoder_{args.cache.name}_best.pt")
            print(f"  new best val L1 -> decoder_{args.cache.name}_best.pt")
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == args.epochs - 1:
            torch.save(payload, CKPT_DIR / f"decoder_{args.cache.name}_ep{epoch:03d}.pt")

    print("\nDone. NOTE: PSNR here is dominated by flat background — a chart can")
    print("score 30 dB with every axis label illegible. Judge this decoder by")
    print("looking at decode(z_tgt) crops in visualize_decoded.py, not by dB.")


if __name__ == "__main__":
    main()