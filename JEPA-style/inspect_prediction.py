"""
inspect_prediction.py — visualize what a trained predictor actually predicts.

THERE IS NO DECODER. The world model predicts embeddings, not pixels, so there is
no "output image" to render — that was the deliberate tradeoff in choosing a
JEPA-style latent predictor over a diffusion model.

What we CAN see is more diagnostic than a picture anyway: WHERE the model expects
change, versus where change actually happened. Per token we compute

    true_delta[n]  = || z_tgt[n] - z_src[n] ||       what really changed
    pred_delta[n]  = || pred[n]  - z_src[n] ||       what the model expected
    error[n]       = || pred[n]  - z_tgt[n] ||       what it got wrong

and paint each over the input screenshot on the encoder's token grid. If
pred_delta lights up on the element the action targeted, the model has grounded
the action spatially. If it is diffuse, or hot in the wrong place, that is the
mislocalization failure — the same one the diffusion model showed at epoch 15,
now visible in latent space where it is cheap to diagnose.

The headline number is the SPATIAL CORRELATION between true_delta and pred_delta.
The identity baseline predicts a zero delta everywhere, so its correlation is
undefined/zero — any positive correlation is signal the copy solution cannot get.

CACHE INDEX != DATASET INDEX: precompute dropped standalone `sleep` rows, so the
j-th cache entry is not the j-th dataset item. We rebuild the mapping once by
replaying the same filter, verify it against the cached action strings, and save
it next to the cache.

Run:
    python inspect_prediction.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt
    python inspect_prediction.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt --index 986
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
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import encoder_spec, letterbox_tensor
from precompute_embeddings import is_sleep
from train_predictor import (
    SEED, VAL_FRAC, EmbeddingBank, build_model, parse_action, parse_continuous,
)

OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "predictions"


def build_transform(image_size: int) -> transforms.Compose:
    """Identical to probe/precompute: the overlay must align with the tokens."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: letterbox_tensor(t, image_size)),
        transforms.Normalize([0.5], [0.5]),
    ])


def dataset_index_map(bank: EmbeddingBank, ds) -> list[int]:
    """Map cache index -> dataset index by replaying precompute's sleep filter.

    Cached to `<cache>/dataset_index_map.json` because building it requires
    touching every dataset item (the action text only comes back with the images).
    Verified against the cached action strings so a silent misalignment is
    impossible.
    """
    path = bank.cache_dir / "dataset_index_map.json"
    if path.exists():
        idx_map = json.loads(path.read_text())
        if len(idx_map) == bank.m:
            return idx_map

    print("Building cache->dataset index map (one-time; touches every sample)...")
    idx_map: list[int] = []
    for i in range(len(ds)):
        if not is_sleep(ds[i]["action_text"]):
            idx_map.append(i)
        if i % 1000 == 0:
            print(f"  {i}/{len(ds)}", flush=True)

    assert len(idx_map) == bank.m, (
        f"replayed filter kept {len(idx_map)} items but the cache has {bank.m}. "
        f"The cache was built from a different dataset or a different filter."
    )
    # Spot-check the alignment rather than trusting the count alone.
    for j in (0, bank.m // 2, bank.m - 1):
        got = ds[idx_map[j]]["action_text"]
        assert got == bank.action_text[j], (
            f"index map misaligned at {j}: dataset says {got!r}, cache says "
            f"{bank.action_text[j]!r}"
        )
    path.write_text(json.dumps(idx_map))
    print(f"Index map saved -> {path}")
    return idx_map


def pick_training_sample(bank: EmbeddingBank, index: int | None) -> int:
    """Deterministically choose a cache index that is in the TRAIN split.

    Reproduces train_predictor's split exactly (same seed, same call order).
    Default pick: the training sample with the most changed tokens — a big,
    unambiguous edit, so a failure to localize it is obviously a failure.
    """
    torch.manual_seed(SEED)
    perm = torch.randperm(bank.m)
    n_val = max(1, int(round(VAL_FRAC * bank.m)))
    train_idx = perm[n_val:]

    if index is not None:
        assert index in set(train_idx.tolist()), (
            f"index {index} is in the VAL split — pass a training index, or use "
            f"the default pick."
        )
        return index

    train_np = train_idx.numpy()
    frac = np.asarray(bank._mask[train_np]).mean(axis=1)
    return int(train_np[int(np.argmax(frac))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--index", type=int, default=None,
                    help="cache index (must be in the train split); default = "
                         "the training pair with the most changed tokens")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    mode = ckpt["mode"]
    print(f"Checkpoint: {args.ckpt.name} | epoch {ckpt['epoch']} | mode={mode}")
    print(f"  val changed-token loss at save: {ckpt['metrics']['model_changed']:.4f} "
          f"(identity {ckpt['metrics']['identity_changed']:.4f})")

    bank = EmbeddingBank(args.cache, mode)
    assert len(bank.elem_vocab) == len(ckpt["elem_vocab"]), (
        f"element vocab size {len(bank.elem_vocab)} != checkpoint's "
        f"{len(ckpt['elem_vocab'])} — cache and checkpoint disagree."
    )
    model = build_model(bank, mode)
    model.load_state_dict(ckpt["model"])
    model.eval()

    j = pick_training_sample(bank, args.index)
    action = bank.action_text[j]
    print(f"\nCache index {j} (TRAIN split)")
    print(f"  action : {action}")
    print(f"  parsed : {parse_action(action)}  cont="
          f"{[round(x, 3) for x in parse_continuous(action)]}")

    # ── forward ──────────────────────────────────────────────────────────────
    idx = torch.tensor([j])
    zs, zt, mk = bank._fetch(idx)
    with torch.no_grad():
        pred = model(zs, bank.action_batch(idx))

    true_delta = (zt - zs).norm(dim=-1)[0].cpu().numpy()      # [N]
    pred_delta = (pred - zs).norm(dim=-1)[0].cpu().numpy()
    err = (pred - zt).norm(dim=-1)[0].cpu().numpy()
    mask = mk[0].cpu().numpy()

    rows, cols = bank.grid
    tm, pm, em, mm = (a.reshape(rows, cols) for a in (true_delta, pred_delta, err, mask))

    # ── the headline number ──────────────────────────────────────────────────
    corr = float(np.corrcoef(true_delta, pred_delta)[0, 1])
    k = max(1, int(mask.sum()))
    top_true = set(np.argsort(-true_delta)[:k].tolist())
    top_pred = set(np.argsort(-pred_delta)[:k].tolist())
    iou = len(top_true & top_pred) / len(top_true | top_pred)
    print(f"\n  spatial correlation (true vs predicted delta): {corr:+.3f}")
    print(f"  top-{k} token IoU                              : {iou:.3f}")
    print(f"  mean delta magnitude — true {true_delta.mean():.3f} / "
          f"predicted {pred_delta.mean():.3f}")
    if pred_delta.mean() < 0.2 * true_delta.mean():
        print("  !!  Predicted delta is much smaller than the true delta — the "
              "model is hedging toward the identity/copy solution.")

    # ── images ───────────────────────────────────────────────────────────────
    spec = encoder_spec(bank.cache_dir.name if bank.cache_dir.name in ("siglip2", "dinov2")
                        else json.loads((bank.cache_dir / "meta.json").read_text())["encoder"])
    tf = build_transform(spec.image_size)
    ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)
    ds_i = dataset_index_map(bank, ds)[j]
    sample = ds[ds_i]
    src_img = (sample["input_image"] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    tgt_img = (sample["output_image"] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

    ext = [0, spec.image_size, spec.image_size, 0]
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax[0, 0].imshow(src_img); ax[0, 0].set_title("input (before)")
    ax[0, 1].imshow(tgt_img); ax[0, 1].set_title("output (after, ground truth)")
    ax[0, 2].imshow(src_img); ax[0, 2].imshow(mm, extent=ext, alpha=0.55, cmap="Blues")
    ax[0, 2].set_title(f"pixel change mask ({mask.mean():.1%} of tokens)")

    ax[1, 0].imshow(src_img); ax[1, 0].imshow(tm, extent=ext, alpha=0.6, cmap="hot")
    ax[1, 0].set_title("TRUE delta ||z_tgt - z_src||")
    ax[1, 1].imshow(src_img); ax[1, 1].imshow(pm, extent=ext, alpha=0.6, cmap="hot")
    ax[1, 1].set_title(f"PREDICTED delta ||pred - z_src||  (corr {corr:+.2f})")
    ax[1, 2].imshow(src_img); ax[1, 2].imshow(em, extent=ext, alpha=0.6, cmap="viridis")
    ax[1, 2].set_title("error ||pred - z_tgt||")
    for a in ax.flat:
        a.axis("off")

    fig.suptitle(f"[{j}] {action[:110]}", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / f"{args.ckpt.stem}_idx{j:05d}.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved -> {out}")
    print("Read the bottom row: do TRUE and PREDICTED light up in the same place?")
    print("Same place = the model grounded the action. Diffuse or displaced =")
    print("it knows something changed but not where — the mislocalization failure.")


if __name__ == "__main__":
    main()