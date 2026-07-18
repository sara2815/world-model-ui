"""
visualize_decoded.py — render predicted screenshots, with the ceiling shown.

Feeds the predictor's output through the trained decoder so you can finally LOOK
at a prediction. Per sample, five panels:

    input | GT after | decode(z_src) | decode(z_tgt) | decode(pred)
                        ^ copy floor    ^ CEILING       ^ the result

WHY THE CEILING PANEL IS NOT OPTIONAL
-------------------------------------
decode(pred) can be blurry for three different reasons: the predictor is
uncertain, the decoder is lossy, or the encoder discarded the detail. One blurry
image cannot tell you which. decode(z_tgt) is the ground-truth embedding through
the same decoder, so it isolates encoder+decoder loss from predictor error:

  * decode(z_tgt) mush   -> the pipeline is at its limit; decode(pred) says
                            NOTHING about the predictor. Stop reading it.
  * decode(z_tgt) crisp
    and decode(pred) mush -> a real predictor result.

decode(z_src) is the floor: the identity/copy solution, rendered. If decode(pred)
is indistinguishable from it, the model is copying whatever the loss curve claims.

SINGLE-SAMPLE MODE ADDS A ZOOM ROW
----------------------------------
Full-frame panels always look acceptable at thumbnail size — that is exactly how
this failure mode goes unnoticed. With --index / --dataset-index we also crop to
the bounding box of the CHANGED tokens and show all five views at 1:1 there. That
crop is where legibility is actually decided.

Run:
    # the gt_input.png / gt_output.png pair (they came from dataset[0])
    python visualize_decoded.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt \
        --decoder checkpoints/decoder_siglip2_best.pt --dataset-index 0

    # a grid across edit sizes
    python visualize_decoded.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt \
        --decoder checkpoints/decoder_siglip2_best.pt --grid 6
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import encoder_spec
from inspect_prediction import build_transform, dataset_index_map
from train_decoder import Decoder
from train_predictor import (
    SEED, VAL_FRAC, EmbeddingBank, build_model, parse_action,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "decoded"
CROP_PAD = 2   # token-grid cells of padding around the changed region


def to_img(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss((a + 1) / 2, (b + 1) / 2).item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def changed_bbox(mask: np.ndarray, grid, patch: int, size: int):
    """Pixel bbox (x0, y0, x1, y1) of the changed tokens, padded and clamped.

    Returns None if nothing changed — the caller then skips the zoom row rather
    than cropping an arbitrary region and pretending it is the edit.
    """
    r, c = grid
    m2 = mask.reshape(r, c)
    hot = np.argwhere(m2 > 0.5)
    if hot.size == 0:
        return None
    y0, x0 = hot.min(axis=0) - CROP_PAD
    y1, x1 = hot.max(axis=0) + 1 + CROP_PAD
    to_px = lambda v: int(np.clip(v * patch, 0, size))
    return to_px(x0), to_px(y0), to_px(x1), to_px(y1)


def load_all(args):
    pc = torch.load(args.ckpt, map_location="cpu")
    bank = EmbeddingBank(args.cache, pc["mode"])
    model = build_model(bank, pc["mode"])
    model.load_state_dict(pc["model"])
    model.eval()

    dc = torch.load(args.decoder, map_location="cpu")
    assert dc["encoder_cache"] == str(args.cache), (
        f"decoder was trained on {dc['encoder_cache']} but you passed "
        f"{args.cache} — embeddings from different encoders are not comparable."
    )
    dec = Decoder(dc["d_model"], tuple(dc["grid"]), dc["image_size"],
                  dc["base_ch"]).to(DEVICE)
    dec.load_state_dict(dc["model"])
    dec.eval()
    print(f"Predictor: {args.ckpt.name} (epoch {pc['epoch']})")
    print(f"Decoder  : {args.decoder.name} (epoch {dc['epoch']}, "
          f"val L1 {dc['metrics']['l1']:.4f})")
    return bank, model, dec


def render_one(bank, model, dec, ds, idx_map, j, spec, out: Path) -> None:
    """Five panels + a zoom row on the changed region."""
    idx = torch.tensor([j])
    zs, zt, mk = bank._fetch(idx)
    with torch.no_grad():
        pred = model(zs, bank.action_batch(idx))
        d_src, d_tgt, d_pred = dec(zs)[0], dec(zt)[0], dec(pred)[0]

    s = ds[idx_map[j]]
    gt_after = s["output_image"].to(DEVICE)
    mask = mk[0].cpu().numpy()
    ceil_db, pred_db, copy_db = (psnr(d_tgt, gt_after), psnr(d_pred, gt_after),
                                 psnr(d_src, gt_after))
    verb, elem = parse_action(bank.action_text[j])

    print(f"\ncache index {j} (dataset index {idx_map[j]})")
    print(f"  action  : {bank.action_text[j][:100]}")
    print(f"  parsed  : {verb}/{elem}   changed tokens: {mask.mean():.1%}")
    print(f"  CEILING decode(z_tgt) vs GT : {ceil_db:6.2f} dB")
    print(f"  RESULT  decode(pred)  vs GT : {pred_db:6.2f} dB")
    print(f"  FLOOR   decode(z_src) vs GT : {copy_db:6.2f} dB")
    if pred_db <= copy_db:
        print("  !!  decode(pred) is no closer to the target than the copy floor.")

    views = [
        (to_img(s["input_image"]), "input (before)"),
        (to_img(gt_after), "GT after"),
        (to_img(d_src), f"decode(z_src) — copy floor {copy_db:.1f} dB"),
        (to_img(d_tgt), f"decode(z_tgt) — CEILING {ceil_db:.1f} dB"),
        (to_img(d_pred), f"decode(pred) — {pred_db:.1f} dB"),
    ]
    box = changed_bbox(mask, bank.grid, spec.patch_size, spec.image_size)
    nrows = 2 if box else 1
    fig, ax = plt.subplots(nrows, 5, figsize=(22, 4.6 * nrows))
    ax = np.atleast_2d(ax)
    for c, (im, t) in enumerate(views):
        ax[0, c].imshow(im); ax[0, c].set_title(t, fontsize=8); ax[0, c].axis("off")
    if box:
        x0, y0, x1, y1 = box
        for c, (im, t) in enumerate(views):
            ax[1, c].imshow(im[y0:y1, x0:x1], interpolation="nearest")
            ax[1, c].set_title(f"ZOOM: {t.split(' — ')[0]}", fontsize=8)
            ax[1, c].axis("off")
        print(f"  zoom crop: ({x0},{y0})-({x1},{y1})")
    else:
        print("  no changed tokens — zoom row skipped")

    fig.suptitle(f"[{j}] {bank.action_text[j][:110]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved -> {out}")
    print("Read the ZOOM row. Is the ceiling legible? If not, decode(pred) cannot")
    print("be. Is decode(pred) different from the copy floor in the right way?")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="predictor checkpoint")
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--index", type=int, default=None, help="cache index")
    ap.add_argument("--dataset-index", type=int, default=None,
                    help="dataset index (gt_input.png/gt_output.png came from 0)")
    ap.add_argument("--grid", type=int, default=0, help="render N samples instead")
    ap.add_argument("--split", choices=["train", "val"], default="val",
                    help="for --grid; val is honest, train shows memorization")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bank, model, dec = load_all(args)

    enc = json.loads((args.cache / "meta.json").read_text())["encoder"]
    spec = encoder_spec(enc)
    tf = build_transform(spec.image_size)
    ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)
    idx_map = dataset_index_map(bank, ds)

    # ── single sample ────────────────────────────────────────────────────────
    if args.index is not None or args.dataset_index is not None:
        if args.dataset_index is not None:
            try:
                j = idx_map.index(args.dataset_index)
            except ValueError:
                raise SystemExit(
                    f"dataset index {args.dataset_index} is not in the cache — it "
                    f"was almost certainly a standalone `sleep` row, which "
                    f"precompute filtered out. Pick another."
                )
        else:
            j = args.index
        render_one(bank, model, dec, ds, idx_map, j, spec,
                   OUT_DIR / f"{args.ckpt.stem}_one{j:05d}.png")
        return

    # ── grid ─────────────────────────────────────────────────────────────────
    n = args.grid or 6
    torch.manual_seed(SEED)
    perm = torch.randperm(bank.m)
    n_val = max(1, int(round(VAL_FRAC * bank.m)))
    pool = (perm[:n_val] if args.split == "val" else perm[n_val:]).numpy()

    # Spread across edit sizes rather than taking the biggest — large layout
    # changes are the easy case and would flatter the model.
    frac = np.asarray(bank._mask[pool]).mean(axis=1)
    order = pool[np.argsort(frac)]
    order = order[np.sort(frac) > 0]
    picks = [int(order[int(round(q * (len(order) - 1)))])
             for q in np.linspace(0.2, 0.99, n)]

    fig, ax = plt.subplots(n, 5, figsize=(22, 4.5 * n))
    ax = np.atleast_2d(ax)
    print(f"\n{'idx':>6} {'ceiling dB':>11} {'pred dB':>9} {'copy dB':>9}  action")
    for r, j in enumerate(picks):
        idx = torch.tensor([j])
        zs, zt, mk = bank._fetch(idx)
        with torch.no_grad():
            pred = model(zs, bank.action_batch(idx))
            d_src, d_tgt, d_pred = dec(zs)[0], dec(zt)[0], dec(pred)[0]
        s = ds[idx_map[j]]
        gt_after = s["output_image"].to(DEVICE)
        ceil_db, pred_db, copy_db = (psnr(d_tgt, gt_after), psnr(d_pred, gt_after),
                                     psnr(d_src, gt_after))
        verb, elem = parse_action(bank.action_text[j])
        print(f"{j:6d} {ceil_db:11.2f} {pred_db:9.2f} {copy_db:9.2f}  "
              f"{verb}/{elem} ({float(mk.mean()):.1%} changed)")
        for c, (im, t) in enumerate([
            (to_img(s["input_image"]), "input (before)"),
            (to_img(gt_after), "GT after"),
            (to_img(d_src), f"decode(z_src) — copy {copy_db:.1f} dB"),
            (to_img(d_tgt), f"decode(z_tgt) — CEILING {ceil_db:.1f} dB"),
            (to_img(d_pred), f"decode(pred) — {pred_db:.1f} dB"),
        ]):
            ax[r, c].imshow(im); ax[r, c].set_title(t, fontsize=8); ax[r, c].axis("off")

    fig.suptitle(f"decoded predictions ({args.split} split, small edits at top)",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"{args.ckpt.stem}_decoded_{args.split}{n}.png"
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")
    print("\nHow to read it:")
    print("  1. CEILING first. If decode(z_tgt) is mush, decode(pred) tells you")
    print("     nothing about the predictor.")
    print("  2. Then decode(pred) vs the copy floor. Same = the model is copying.")
    print("  3. pred dB above copy dB = genuinely closer to the true after-state.")


if __name__ == "__main__":
    main()