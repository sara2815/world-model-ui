"""
visualize_predictions.py — see what the latent world model predicts, as pictures.

There is no decoder, so we cannot render a predicted screenshot. Two things we CAN
render, both more honest than a blurry reconstruction would be:

GRID MODE (--grid N)
    N samples, one per row: input | true change | predicted change, as heatmaps on
    the encoder token grid painted over the screenshot. One example can mislead;
    a grid shows whether localization holds across action types or only works for
    the big obvious edits. This is the primary answer to "what parts of the image
    are getting changed".

NEIGHBOUR MODE (--neighbors)
    Delta-space nearest neighbours. The naive version of this idea — rank
    candidate z_tgt by distance to pred — is a trap: pred ~= z_src ~= z_tgt, so it
    retrieves the sample's own ground truth ~always and looks perfect while
    proving nothing. Instead we rank candidate (z_tgt - z_src) by distance to
    (pred - z_src): "which known CHANGE is the predicted change most like?" We
    then show that neighbour's real before/after images. The identity baseline
    predicts a zero delta for every sample and cannot rank at all, so anything
    above chance here is real.

Run:
    python visualize_predictions.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt --grid 8
    python visualize_predictions.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt --neighbors --index 986
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import encoder_spec
from inspect_prediction import build_transform, dataset_index_map
from train_predictor import (
    SEED, VAL_FRAC, EmbeddingBank, build_model, parse_action,
)

OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "predictions"
NEIGHBOR_POOL = 1000   # training deltas to search; 1000 x 1.18M fp16 ~= 2.4 GB
N_NEIGHBORS = 3


def load(cache: Path, ckpt_path: Path):
    """Load checkpoint + bank + model, asserting they belong together."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    bank = EmbeddingBank(cache, ckpt["mode"])
    assert len(bank.elem_vocab) == len(ckpt["elem_vocab"]), (
        f"element vocab {len(bank.elem_vocab)} != checkpoint's "
        f"{len(ckpt['elem_vocab'])} — cache and checkpoint disagree."
    )
    model = build_model(bank, ckpt["mode"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint {ckpt_path.name} | epoch {ckpt['epoch']} | mode {ckpt['mode']}")
    return ckpt, bank, model


def splits(bank: EmbeddingBank):
    """Reproduce train_predictor's split exactly (same seed, same call order)."""
    torch.manual_seed(SEED)
    perm = torch.randperm(bank.m)
    n_val = max(1, int(round(VAL_FRAC * bank.m)))
    return perm[n_val:], perm[:n_val]


def images_for(ds, idx_map, j: int):
    """(before, after) as HWC float arrays in [0,1] for cache index j."""
    s = ds[idx_map[j]]
    to_img = lambda t: (t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    return to_img(s["input_image"]), to_img(s["output_image"])


def deltas_for(bank: EmbeddingBank, model, j: int):
    """(true_delta_map, pred_delta_map, corr) for cache index j."""
    idx = torch.tensor([j])
    zs, zt, _ = bank._fetch(idx)
    with torch.no_grad():
        pred = model(zs, bank.action_batch(idx))
    td = (zt - zs).norm(dim=-1)[0].cpu().numpy()
    pd = (pred - zs).norm(dim=-1)[0].cpu().numpy()
    corr = float(np.corrcoef(td, pd)[0, 1])
    rows, cols = bank.grid
    return td.reshape(rows, cols), pd.reshape(rows, cols), corr


# ── Grid mode ────────────────────────────────────────────────────────────────
def grid_mode(bank, model, ds, idx_map, n: int, out: Path) -> None:
    """Spread the sample choice across change magnitudes, not just the big edits.

    Picking the top-N changed pairs would flatter the model: large layout changes
    are the easy case. We take an even spread over the changed-token fraction so
    the small text/value edits — the ones that matter and that the probe showed
    are hardest — are represented.
    """
    train_idx, _ = splits(bank)
    tn = train_idx.numpy()
    frac = np.asarray(bank._mask[tn]).mean(axis=1)
    order = tn[np.argsort(frac)]
    order = order[frac[np.argsort(frac)] > 0]           # skip degenerate pairs
    picks = [int(order[int(round(q * (len(order) - 1)))]) 
             for q in np.linspace(0.15, 0.99, n)]

    size = encoder_spec(json.loads((bank.cache_dir / "meta.json").read_text())["encoder"]).image_size
    ext = [0, size, size, 0]
    fig, ax = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    ax = np.atleast_2d(ax)
    for r, j in enumerate(picks):
        before, _ = images_for(ds, idx_map, j)
        tm, pm, corr = deltas_for(bank, model, j)
        mfrac = float(np.asarray(bank._mask[j]).mean())
        verb, elem = parse_action(bank.action_text[j])

        ax[r, 0].imshow(before)
        ax[r, 0].set_title(f"[{j}] {verb}/{elem}  ({mfrac:.1%} tokens changed)",
                           fontsize=8, loc="left")
        ax[r, 1].imshow(before); ax[r, 1].imshow(tm, extent=ext, alpha=0.6, cmap="hot")
        ax[r, 1].set_title("TRUE change", fontsize=8)
        ax[r, 2].imshow(before); ax[r, 2].imshow(pm, extent=ext, alpha=0.6, cmap="hot")
        ax[r, 2].set_title(f"PREDICTED change  (corr {corr:+.2f})", fontsize=8)
        for a in ax[r]:
            a.axis("off")
    fig.suptitle("true vs predicted change, spread across edit sizes "
                 "(small edits at top)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Saved grid -> {out}")
    print("Read top-to-bottom: localization on the SMALL edits (top rows) is the")
    print("real test. Good corr only on the bottom rows means the model handles")
    print("big layout changes and misses the fine ones.")


# ── Neighbour mode ───────────────────────────────────────────────────────────
def neighbor_mode(bank, model, ds, idx_map, j: int, out: Path) -> None:
    """Delta-space NN: which known change is the predicted change most like?"""
    train_idx, _ = splits(bank)
    pool = train_idx[:NEIGHBOR_POOL]
    pool = pool[pool != j]

    idx = torch.tensor([j])
    zs, zt, _ = bank._fetch(idx)
    with torch.no_grad():
        pred = model(zs, bank.action_batch(idx))
    q = (pred - zs).reshape(1, -1).half()               # predicted delta

    cands, keep = [], []
    for start in range(0, pool.numel(), 8):
        b = pool[start : start + 8]
        bzs, bzt, _ = bank._fetch(b)
        cands.append((bzt - bzs).reshape(b.numel(), -1).half())
        keep.extend(b.tolist())
    cands = torch.cat(cands)

    d = ((q.float() ** 2).sum(1, keepdim=True)
         + (cands.float() ** 2).sum(1, keepdim=True).t()
         - 2.0 * (q.float() @ cands.float().t()))[0]
    nn = [keep[i] for i in d.argsort()[:N_NEIGHBORS].tolist()]

    print(f"\nQuery [{j}]: {bank.action_text[j][:90]}")
    print("Nearest known CHANGES in delta space:")
    for r, k in enumerate(nn):
        print(f"  {r+1}. [{k}] {bank.action_text[k][:80]}")

    fig, ax = plt.subplots(N_NEIGHBORS + 1, 2, figsize=(11, 4.2 * (N_NEIGHBORS + 1)))
    qb, qa = images_for(ds, idx_map, j)
    ax[0, 0].imshow(qb); ax[0, 0].set_title(f"QUERY [{j}] before", fontsize=9)
    ax[0, 1].imshow(qa); ax[0, 1].set_title("QUERY after (ground truth)", fontsize=9)
    for r, k in enumerate(nn, start=1):
        nb, na = images_for(ds, idx_map, k)
        ax[r, 0].imshow(nb)
        ax[r, 0].set_title(f"NN{r} [{k}] before — {bank.action_text[k][:55]}",
                           fontsize=8, loc="left")
        ax[r, 1].imshow(na); ax[r, 1].set_title(f"NN{r} after", fontsize=8)
    for a in ax.flat:
        a.axis("off")
    fig.suptitle("The predicted change is most similar to these known changes",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")
    print("If the neighbours share the query's ACTION TYPE and change REGION, the")
    print("model has learned an action->change mapping. If they look arbitrary, it")
    print("is predicting a generic 'something changed' direction.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--grid", type=int, default=0, help="render N samples as a grid")
    ap.add_argument("--neighbors", action="store_true")
    ap.add_argument("--index", type=int, default=None, help="cache index for --neighbors")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt, bank, model = load(args.cache, args.ckpt)

    enc = json.loads((bank.cache_dir / "meta.json").read_text())["encoder"]
    ds = WorldModelDataset(use_annotated_image=True,
                           transform=build_transform(encoder_spec(enc).image_size),
                           target_transform=build_transform(encoder_spec(enc).image_size))
    idx_map = dataset_index_map(bank, ds)

    if args.grid:
        grid_mode(bank, model, ds, idx_map, args.grid,
                  OUT_DIR / f"{args.ckpt.stem}_grid{args.grid}.png")
    if args.neighbors:
        train_idx, _ = splits(bank)
        j = args.index
        if j is None:
            tn = train_idx.numpy()
            j = int(tn[int(np.argmax(np.asarray(bank._mask[tn]).mean(axis=1)))])
        assert j in set(train_idx.tolist()), f"index {j} is not in the train split"
        neighbor_mode(bank, model, ds, idx_map, j,
                      OUT_DIR / f"{args.ckpt.stem}_nn{j:05d}.png")
    if not args.grid and not args.neighbors:
        print("Nothing to do — pass --grid N and/or --neighbors")


if __name__ == "__main__":
    main()