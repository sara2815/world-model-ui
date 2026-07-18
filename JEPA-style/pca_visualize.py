"""
pca_visualize.py — V-JEPA-style PCA-to-RGB view of the latent tokens.

Standard practice in JEPA work (and what the V-JEPA 2.1 paper does): compute PCA
on the patch features and map the top three components to RGB. It needs no
decoder, no training, and runs instantly — which is exactly why the JEPA family
uses it. It shows the *semantic layout* the encoder sees: regions that share a
colour share a representation.

Here it is extended to the world-model setting. We render three token grids:

    PCA(z_src)   what the encoder sees before the action
    PCA(z_tgt)   what it sees after            <- the target
    PCA(pred)    what the model predicts it will see

THE ONE DETAIL THAT MAKES OR BREAKS THIS
----------------------------------------
The PCA basis MUST be fit jointly on all three token sets, not separately on
each. PCA components are defined only up to sign and rotation, so three separate
fits give three arbitrary colour schemes and the panels become incomparable —
you would be reading colour differences that are pure basis noise. With a shared
basis, "PCA(pred) looks like PCA(z_tgt) and not like PCA(z_src)" is a real
observation about the prediction.

Normalization is by joint 2nd/98th percentile across all three, for the same
reason: per-panel min/max would rescale each independently and manufacture
contrast where there is none.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
Shows: does the predicted representation reorganize the right REGION in the right
DIRECTION. Three panels, same colours, one visibly different patch.
Does not show: whether an axis label reads 4.2 or 4.7. Three PCA components out
of 1152 dims is a drastic projection. This is a semantic-layout view, not a
fidelity check — for fidelity you need the decoder.

Run:
    python pca_visualize.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt --dataset-index 0
    python pca_visualize.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt --grid 6 --split val
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
from data.dataset import WorldModelDataset  # noqa: E402

from wm_encoders import encoder_spec  # noqa: E402
from inspect_prediction import build_transform, dataset_index_map  # noqa: E402
from train_predictor import (  # noqa: E402
    SEED, VAL_FRAC, EmbeddingBank, build_model, parse_action,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "pca"


def pca_rgb(token_sets: list[torch.Tensor], grid) -> list[np.ndarray]:
    """Project several [N, D] token sets through ONE shared PCA basis -> RGB maps.

    Args:
        token_sets: list of [N, D] tensors (e.g. z_src, z_tgt, pred).
        grid: (rows, cols) to reshape each [N, 3] projection into.

    Returns:
        list of [rows, cols, 3] float arrays in [0, 1], one per input set.

    The basis is fit on the concatenation so the three renders are directly
    comparable. Fitting per-set would give each an arbitrary sign/rotation and
    the comparison would be meaningless.
    """
    r, c = grid
    X = torch.cat(token_sets, dim=0).float()       # [k*N, D]
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    # Randomized SVD: D is 1152 and k*N ~ 3072, so full SVD is unnecessary.
    _, _, V = torch.pca_lowrank(Xc, q=3, center=False)
    proj = Xc @ V[:, :3]                           # [k*N, 3]

    # Joint robust normalization — percentiles, not min/max, so one outlier token
    # cannot flatten the rest of the map.
    lo = torch.quantile(proj, 0.02, dim=0, keepdim=True)
    hi = torch.quantile(proj, 0.98, dim=0, keepdim=True)
    rgb = ((proj - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)

    n = token_sets[0].shape[0]
    return [rgb[i * n : (i + 1) * n].reshape(r, c, 3).cpu().numpy()
            for i in range(len(token_sets))]


def render(bank, model, ds, idx_map, j, spec, ax_row) -> dict:
    """One sample: fill a 5-panel axis row, return stats."""
    idx = torch.tensor([j])
    zs, zt, mk = bank._fetch(idx)
    with torch.no_grad():
        pred = model(zs, bank.action_batch(idx))

    p_src, p_tgt, p_pred = pca_rgb([zs[0], zt[0], pred[0]], bank.grid)
    s = ds[idx_map[j]]
    to_img = lambda t: (t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

    # How close is the predicted PCA map to the target's, vs the source's?
    # Same logic as the dB floor/ceiling in the decoder view, in PCA space.
    d_pred_tgt = float(np.abs(p_pred - p_tgt).mean())
    d_src_tgt = float(np.abs(p_src - p_tgt).mean())
    verb, elem = parse_action(bank.action_text[j])

    for c, (im, t) in enumerate([
        (to_img(s["input_image"]), "input (before)"),
        (to_img(s["output_image"]), "GT after"),
        (p_src, "PCA(z_src)"),
        (p_tgt, "PCA(z_tgt) — target"),
        (p_pred, f"PCA(pred)  |pred-tgt| {d_pred_tgt:.3f}"),
    ]):
        ax_row[c].imshow(im, interpolation="nearest")
        ax_row[c].set_title(t, fontsize=8)
        ax_row[c].axis("off")

    return {"j": j, "verb": verb, "elem": elem, "changed": float(mk.mean()),
            "d_pred_tgt": d_pred_tgt, "d_src_tgt": d_src_tgt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--index", type=int, default=None, help="cache index")
    ap.add_argument("--dataset-index", type=int, default=None,
                    help="dataset index (gt_input.png came from 0)")
    ap.add_argument("--grid", type=int, default=0, help="render N samples instead")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    bank = EmbeddingBank(args.cache, ckpt["mode"])
    model = build_model(bank, ckpt["mode"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Predictor: {args.ckpt.name} (epoch {ckpt['epoch']}, mode {ckpt['mode']})")

    enc = json.loads((args.cache / "meta.json").read_text())["encoder"]
    spec = encoder_spec(enc)
    tf = build_transform(spec.image_size)
    ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)
    idx_map = dataset_index_map(bank, ds)

    # ── which samples ────────────────────────────────────────────────────────
    if args.dataset_index is not None:
        try:
            picks = [idx_map.index(args.dataset_index)]
        except ValueError:
            raise SystemExit(
                f"dataset index {args.dataset_index} is not in the cache — it was "
                f"almost certainly a standalone `sleep` row that precompute "
                f"filtered out."
            )
        tag = f"one{picks[0]:05d}"
    elif args.index is not None:
        picks, tag = [args.index], f"one{args.index:05d}"
    else:
        n = args.grid or 6
        torch.manual_seed(SEED)
        perm = torch.randperm(bank.m)
        n_val = max(1, int(round(VAL_FRAC * bank.m)))
        pool = (perm[:n_val] if args.split == "val" else perm[n_val:]).numpy()
        # Spread across edit sizes; the biggest edits are the easy case.
        frac = np.asarray(bank._mask[pool]).mean(axis=1)
        order = pool[np.argsort(frac)]
        order = order[np.sort(frac) > 0]
        picks = [int(order[int(round(q * (len(order) - 1)))])
                 for q in np.linspace(0.2, 0.99, n)]
        tag = f"{args.split}{n}"

    # ── render ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(len(picks), 5, figsize=(21, 4.4 * len(picks)))
    ax = np.atleast_2d(ax)
    print(f"\n{'idx':>6} {'changed':>8} {'|pred-tgt|':>11} {'|src-tgt|':>10}  action")
    for r, j in enumerate(picks):
        st = render(bank, model, ds, idx_map, j, spec, ax[r])
        better = "closer" if st["d_pred_tgt"] < st["d_src_tgt"] else "NOT closer"
        print(f"{st['j']:6d} {st['changed']:8.1%} {st['d_pred_tgt']:11.4f} "
              f"{st['d_src_tgt']:10.4f}  {st['verb']}/{st['elem']}  ({better})")

    fig.suptitle("PCA of patch features -> RGB (shared basis across all three "
                 "token sets)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"{args.ckpt.stem}_pca_{tag}.png"
    fig.savefig(out, dpi=115)
    print(f"\nSaved -> {out}")
    print("\nHow to read it: the last three panels share one PCA basis, so colour")
    print("means the same thing in all of them. Does PCA(pred) reorganize the same")
    print("region as PCA(z_tgt), and in the same direction? |pred-tgt| below")
    print("|src-tgt| means the prediction moved toward the target rather than")
    print("sitting at the copy solution.")


if __name__ == "__main__":
    main()