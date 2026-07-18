"""
probe_encoder.py — does a frozen encoder even *see* the UI changes we care about?

Before training any world model we must answer a prior question: given a frozen
vision encoder, is the embedding of the *after* screenshot measurably different
from the *before* screenshot, and is that difference concentrated where the UI
actually changed? If the encoder collapses the small edit (a dropdown opening, a
value changing) into embedding noise, no predictor can recover it and the whole
approach is dead on arrival.

THE METRIC THAT MATTERS
-----------------------
We compare, within each pair, the encoder's response on tokens where pixels
changed against its response on tokens where they did not:

    ratio = mean(d_pair[mask == 1]) / mean(d_pair[mask == 0])

``d_unchanged`` is the honest noise floor. It is nonzero even for a deterministic
encoder because ViT tokens have global receptive fields — an edit anywhere
perturbs every token a little. If changed tokens do not move meaningfully more
than unchanged ones, the encoder is not localizing the edit.

An earlier version of this probe compared ``d_pair`` against ``d_random`` (the
distance to an unrelated screenshot's embedding) and required d_pair to be
*larger*. That criterion is inverted: two frames of the same UI are far more
similar than two unrelated UIs, so d_pair/d_random is expected to be well below
1 even when the encoder works perfectly. d_random is still reported as context —
it bounds the scale of the embedding space — but it does not gate PASS/FAIL.

STRATIFICATION
--------------
The mean over all pairs is misleading: it is dominated by large layout changes
(a dropdown opens) while the cases we actually care about (a value changes, an
axis rescales) touch <1% of tokens. We therefore bucket pairs by the fraction of
tokens that changed and report the ratio per bucket. The smallest bucket is the
one that decides whether this approach is viable.

Run:
    python probe_encoder.py --encoder siglip2 --max-samples 400
    python probe_encoder.py --encoder dinov2  --max-samples 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, never open a window
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import build_encoder, encoder_spec, letterbox_tensor

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda"
BATCH_SIZE = 4        # so400m @ 512^2 fp16, two images per sample — keep small
NUM_WORKERS = 2
OUT_DIR = Path(__file__).resolve().parent / "probe_out"

MASK_DIFF_THRESH = 0.05   # on the [-1,1] scale, so ~2.5% of dynamic range

# PASS/FAIL thresholds.
RATIO_MIN = 3.0           # d_changed must be >= 3x d_unchanged (the noise floor)
LOCALIZATION_MIN = 0.25   # top-decile tokens must carry >= 25% of total change
#                           (a perfectly diffuse change would carry only ~10%)

# Buckets over "fraction of tokens changed". The first row is the decisive one.
BUCKETS: list[tuple[float, float]] = [
    (0.00, 0.01),
    (0.01, 0.05),
    (0.05, 0.20),
    (0.20, 1.01),
]


def build_transform(image_size: int) -> transforms.Compose:
    """Letterbox to the encoder's square resolution, then normalize to [-1,1].

    Screenshots are 1280x720. A direct Resize((512,512)) would squash them 2.5x
    horizontally against 1.4x vertically, turning already-small text into
    unreadable smears *before* the encoder sees it — which would make this probe
    measure the transform's damage rather than the encoder's capability.
    letterbox_tensor preserves aspect ratio and pads with white.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),  # -> [0,1]
            transforms.Lambda(lambda t: letterbox_tensor(t, image_size)),
            transforms.Normalize([0.5], [0.5]),  # -> [-1,1]
        ]
    )


def per_token_delta(a: Tensor, b: Tensor) -> Tensor:
    """L2 distance per token: ``[B, N, D] -> [B, N]``."""
    return (a.float() - b.float()).norm(dim=-1)


@torch.inference_mode()
def collect_stats(encoder, loader: DataLoader) -> tuple[Tensor, Tensor, Tensor]:
    """Encode batch-by-batch, keeping only the [M, N] reductions.

    Never materializes the full [M, N, D] embedding tensors: at 8413 pairs x 1024
    tokens x 1152 dims those are ~20 GB each in fp16, and the fp32 subtraction in
    per_token_delta would transiently need ~80 GB. Reducing inside the loop keeps
    the whole probe in a few MB.

    Returns (d_pair, d_random, masks), each [M, N] on CPU.
    """
    d_pair_chunks: list[Tensor] = []
    d_rand_chunks: list[Tensor] = []
    mask_chunks: list[Tensor] = []
    grid = encoder.grid

    for batch in loader:
        src = batch["input_image"].to(DEVICE, non_blocking=True)
        tgt = batch["output_image"].to(DEVICE, non_blocking=True)
        if src.shape[0] < 2:
            continue  # random baseline needs >=2 samples to pair against

        z_src = encoder.encode(src)
        z_tgt = encoder.encode(tgt)

        # SIGNAL: the true (input -> output) transition.
        d_pair_chunks.append(per_token_delta(z_src, z_tgt).cpu())

        # CONTEXT: each src paired with a *different* sample's tgt. Reported for
        # scale only; see module docstring for why it does not gate PASS/FAIL.
        d_rand_chunks.append(per_token_delta(z_src, z_tgt.roll(1, dims=0)).cpu())

        # Pixel diff -> token grid. precompute_embeddings.py MUST use this exact
        # recipe (same threshold, same max-pool, same flatten order) or the
        # training loss will upweight the wrong tokens.
        diff = (tgt - src).abs().amax(dim=1, keepdim=True)      # [B,1,H,W]
        m = (diff > MASK_DIFF_THRESH).float()
        m = F.adaptive_max_pool2d(m, grid).flatten(1)           # [B,N]
        mask_chunks.append(m.cpu())

    return torch.cat(d_pair_chunks), torch.cat(d_rand_chunks), torch.cat(mask_chunks)


def signal_ratio(d: Tensor, mask: Tensor) -> tuple[float, float, float]:
    """(d_changed, d_unchanged, ratio) over a selection of pairs.

    Both tensors are [K, N]; they are flattened together so the statistic is
    token-weighted across the selection.
    """
    changed = d[mask > 0.5]
    unchanged = d[mask <= 0.5]
    if changed.numel() == 0 or unchanged.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    dc = changed.mean().item()
    du = unchanged.mean().item()
    return dc, du, dc / max(du, 1e-8)


def localization_stats(d_pair: Tensor) -> tuple[float, float]:
    """(mean frac_above_median, mean top_decile_mass) over pairs.

    ``top_decile_mass`` is the informative one: the share of total change
    magnitude carried by the top 10% of tokens. A localized edit concentrates
    change into few tokens (well above 0.10); a diffuse whole-frame shift spreads
    it evenly (near 0.10). ``frac_above_median`` sits near 0.5 for any
    distribution and is reported only for transparency.
    """
    median = d_pair.median(dim=1, keepdim=True).values
    frac_above = (d_pair > median).float().mean().item()

    n = d_pair.shape[1]
    k = max(1, n // 10)
    top_sum = torch.topk(d_pair, k, dim=1).values.sum(dim=1)
    total = d_pair.sum(dim=1).clamp_min(1e-8)
    top_mass = (top_sum / total).mean().item()
    return frac_above, top_mass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default="siglip2", help="registry key")
    parser.add_argument("--max-samples", type=int, default=400, help="0 = all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = encoder_spec(args.encoder)
    tf = build_transform(spec.image_size)
    ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)
    if args.max_samples and args.max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(args.max_samples)))

    side = spec.image_size // spec.patch_size
    print(f"Dataset: {len(ds)} pairs | encoder={args.encoder} ({spec.hf_name})")
    print(f"  grid={side}x{side} ({side * side} tokens)  D={spec.d_model}  "
          f"letterboxed to {spec.image_size}^2")

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    encoder = build_encoder(args.encoder, DEVICE)
    d_pair, d_rand, masks = collect_stats(encoder, loader)
    m, n_tokens = d_pair.shape

    frac_changed = masks.mean(dim=1)  # [M] fraction of tokens touched per pair

    # ── Overall ──────────────────────────────────────────────────────────────
    dc_all, du_all, ratio_all = signal_ratio(d_pair, masks)
    frac_above, top_mass = localization_stats(d_pair)

    print("\n===== PROBE RESULTS =====")
    print(f"pairs analysed          : {m}")
    print(f"mean frac tokens changed: {frac_changed.mean().item():.4f}")
    print(f"pairs with empty mask   : {(frac_changed == 0).sum().item()}")
    print()
    print(f"d_changed               : {dc_all:.4f}")
    print(f"d_unchanged (noise floor): {du_all:.4f}")
    print(f"ratio changed/unchanged : {ratio_all:.3f}   (threshold >= {RATIO_MIN})")
    print(f"d_random (context only)  : {d_rand.mean().item():.4f}   "
          f"<- unrelated screenshots; expect d_changed << this")
    print()
    print(f"frac tokens > pair-median: {frac_above:.3f}  (~0.5 by construction)")
    print(f"top-decile change mass   : {top_mass:.3f}   "
          f"(threshold >= {LOCALIZATION_MIN}; diffuse baseline ~0.10)")

    # ── Stratified by change size — THE decisive table ────────────────────────
    print("\n----- stratified by fraction of tokens changed -----")
    print(f"{'bucket':>12} {'n':>5} {'d_changed':>10} {'d_unchanged':>12} {'ratio':>7}")
    for lo, hi in BUCKETS:
        sel = (frac_changed >= lo) & (frac_changed < hi)
        k = int(sel.sum())
        if k == 0:
            continue
        dc, du, r = signal_ratio(d_pair[sel], masks[sel])
        label = f"{lo:.0%}-{hi:.0%}"
        if dc != dc:  # nan
            print(f"{label:>12} {k:5d} {'—':>10} {'—':>12} {'—':>7}")
        else:
            print(f"{label:>12} {k:5d} {dc:10.4f} {du:12.4f} {r:7.2f}")
    print("The <1% row is the one that matters: those are the small text/value")
    print("edits. If its ratio is ~1, the encoder cannot see them.")

    signal_ok = ratio_all >= RATIO_MIN
    local_ok = top_mass >= LOCALIZATION_MIN
    passed = signal_ok and local_ok
    print("\nsignal above noise floor:", "PASS" if signal_ok else "FAIL")
    print("change is localized     :", "PASS" if local_ok else "FAIL")
    print("=========================")
    print("OVERALL:", "PASS  encoder represents localized UI edits"
          if passed else
          "FAIL  encoder does NOT usefully represent the edits")
    print("(Advisory only — read the stratified table before believing this.)")

    # ── Histogram: changed vs unchanged tokens ───────────────────────────────
    hist_path = OUT_DIR / f"probe_hist_{args.encoder}.png"
    plt.figure(figsize=(9, 5))
    changed_vals = d_pair[masks > 0.5].numpy()
    unchanged_vals = d_pair[masks <= 0.5].numpy()
    if changed_vals.size:
        plt.hist(changed_vals, bins=120, alpha=0.6, density=True,
                 label=f"changed tokens (mean={dc_all:.3f})")
    plt.hist(unchanged_vals, bins=120, alpha=0.6, density=True,
             label=f"unchanged tokens (mean={du_all:.3f})")
    plt.xlabel("per-token L2 distance between encode(input) and encode(output)")
    plt.ylabel("density")
    plt.title(f"{args.encoder}: token deltas where pixels changed vs where they "
              f"did not (ratio={ratio_all:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=130)
    print(f"\nHistogram saved -> {hist_path}")
    print("Read it as: how much do the two distributions overlap? Heavy overlap")
    print("means the encoder's response to a real edit is indistinguishable from")
    print("its response to untouched background.")


if __name__ == "__main__":
    main()