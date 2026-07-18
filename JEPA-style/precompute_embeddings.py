"""
precompute_embeddings.py — cache frozen encoder outputs + change masks once.

The encoder is frozen and never trained, so re-running it every epoch is pure
waste. We encode each (input, output) pair a single time and cache:
    z_src        [N, D]  encoder tokens of the before-image
    z_tgt        [N, D]  encoder tokens of the after-image
    change_mask  [N]     which tokens actually changed
    action_text  str

WHY MEMMAP AND NOT ONE .pt
--------------------------
At ~8k pairs x 1024 tokens x 1152 dims x 2 bytes, z_src alone is ~20 GB and
z_tgt another ~20 GB. `torch.cat` on accumulated chunks needs the chunks *and*
the result resident at once (~80 GB peak), and `torch.load` in the trainer would
need the whole thing in RAM again. We stream straight into .npy memmaps instead;
the trainer opens them with `np.load(..., mmap_mode='r')` and pages in only the
rows a batch touches.

WHY LETTERBOX AND NOT RESIZE
----------------------------
Screenshots are 1280x720. `Resize((512, 512))` squashes 2.5x horizontally against
1.4x vertically, turning small text into smears. This must match probe_encoder.py
*exactly* or the cache describes different data than the probe measured.

TOKEN-ORDER CORRECTNESS IS THE #1 SILENT FAILURE MODE
-----------------------------------------------------
If the mask's flatten order does not match the encoder's token order, every mask
is scrambled and the weighted loss upweights the wrong tokens — no crash, just
quietly worse results.

Two things that do NOT work as tests, both learned the hard way:
  * A pure-tensor test (paint a corner, reshape, assert the corner is there)
    always passes: reshape is row-major by definition, so it verifies PyTorch
    rather than the encoder.
  * An absolute overlap threshold ("mask must capture >50% of the encoder's
    most-changed tokens") fails on a *correct* encoder. ViT attention is global,
    so an edit perturbs every token; on a synthetic uniform field the delta is
    almost entirely diffuse and overlap lands near 27%.

What works is a COMPARATIVE test: score the row-major mask against the plausible
wrong orderings (transpose, flips) and require row-major to win. That is immune
to how diffuse the delta is, because every candidate is scored on the same delta.

Run:
    python precompute_embeddings.py --encoder siglip2 --test-only
    python precompute_embeddings.py --encoder siglip2 --out cache/siglip2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import build_encoder, encoder_spec, letterbox_tensor

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda"
BATCH_SIZE = 4           # so400m @ 512^2 fp16, two images per sample
NUM_WORKERS = 2
MASK_DIFF_THRESH = 0.05  # on the [-1,1] scale => ~2.5% of full pixel range
SLEEP_PREFIX = "sleep"   # `sleep 1.00s` is a wait, not an action — see below
DEFAULT_OUT = Path(__file__).resolve().parent / "cache" / "embeddings"

# Minimum inside/outside delta ratio for the winning ordering. A quarter-image
# edit should be *clearly* visible; if even the best ordering is near 1.0, the
# encoder is not localizing at all and nothing downstream will work.
MIN_ALIGNMENT_RATIO = 1.5


def build_transform(image_size: int) -> transforms.Compose:
    """MUST stay identical to probe_encoder.build_transform."""
    return transforms.Compose(
        [
            transforms.ToTensor(),  # -> [0,1]
            transforms.Lambda(lambda t: letterbox_tensor(t, image_size)),
            transforms.Normalize([0.5], [0.5]),  # -> [-1,1]
        ]
    )


def is_sleep(action_text: str) -> bool:
    """`sleep N.NNs` is the recorder's own wait event, not a user action.

    A sleep's before/after frames are bit-identical in ~96% of cases, so these
    pairs teach the predictor `z_tgt = z_src` — exactly the identity-copy
    attractor the change-mask weighting exists to fight. Dropping them removes
    ~14% of the dataset.
    """
    return action_text.strip().lower().startswith(SLEEP_PREFIX)


def compute_change_mask(
    input_img: Tensor,
    output_img: Tensor,
    grid: tuple[int, int],
    threshold: float = MASK_DIFF_THRESH,
) -> Tensor:
    """Binary per-token change mask from the pixel difference.

    Args:
        input_img/output_img: ``[B, 3, H, W]`` on the ``[-1, 1]`` scale.
        grid: ``(rows, cols)`` of the encoder token grid.
        threshold: pixel-diff cutoff on the ``[-1, 1]`` scale.

    Returns:
        ``[B, N]`` float mask (1.0 = changed), ``N = rows * cols``, flattened
        row-major to match HF vision towers' ``conv -> flatten(2)`` token order.

    Derived from pixels rather than embeddings because pixels are an unambiguous,
    encoder-independent ground truth of what visually changed.
    """
    rows, cols = grid
    diff = (input_img - output_img).abs().amax(dim=1, keepdim=True)  # [B,1,H,W]
    pooled = F.adaptive_max_pool2d(diff, output_size=(rows, cols))    # [B,1,rows,cols]
    mask = (pooled > threshold).float()
    return mask.reshape(mask.shape[0], rows * cols)


@torch.inference_mode()
def test_mask_matches_encoder_tokens(encoder, ds=None) -> None:
    """Assert row-major flatten beats the plausible alternative orderings.

    Scores each candidate ordering by the mean token delta *inside* the mask over
    the mean delta *outside* it — the same magnitude-aware statistic the probe
    uses, rather than a top-k set overlap that discards magnitude. Row-major must
    win outright.

    Uses a real dataset image as the base when available. A uniform synthetic
    field is the worst case: every patch is identical, so the tokens are
    degenerate and there is no local structure for the delta to localize against.
    """
    size = encoder.image_size
    rows, cols = encoder.grid

    if ds is not None:
        base = ds[0]["input_image"].unsqueeze(0).to(DEVICE)
        src_desc = "real dataset image"
    else:
        base = torch.full((1, 3, size, size), -1.0, device=DEVICE)
        src_desc = "synthetic uniform field (degenerate — expect weak ratios)"
    edit = base.clone()
    q = size // 4
    edit[:, :, 0:q, size - q:size] = 1.0  # top-right corner block

    print(f"[unit test] base = {src_desc}")
    delta = (encoder.encode(edit) - encoder.encode(base)).float().norm(dim=-1)[0].cpu()
    mask2d = compute_change_mask(base.cpu(), edit.cpu(), (rows, cols))[0].reshape(rows, cols)

    def score(m2d: Tensor) -> float:
        m = m2d.reshape(-1).bool()
        if m.sum() == 0 or (~m).sum() == 0:
            return float("nan")
        return (delta[m].mean() / delta[~m].mean().clamp_min(1e-8)).item()

    candidates = {
        "row-major (expected)": mask2d,
        "column-major":         mask2d.t().contiguous(),
        "flip-rows":            mask2d.flip(0),
        "flip-cols":            mask2d.flip(1),
    }
    scores = {k: score(v) for k, v in candidates.items()}
    for k, v in scores.items():
        print(f"    {k:22s} inside/outside delta ratio = {v:6.2f}")

    best = max(scores, key=lambda k: scores[k])
    assert best == "row-major (expected)", (
        f"token order mismatch: '{best}' scores higher than row-major "
        f"({scores[best]:.2f} vs {scores['row-major (expected)']:.2f}). "
        f"Do NOT build the cache."
    )
    assert scores[best] > MIN_ALIGNMENT_RATIO, (
        f"row-major wins but its ratio is only {scores[best]:.2f} (need "
        f">{MIN_ALIGNMENT_RATIO}) — the encoder barely localizes even a "
        f"quarter-image edit. Something is off."
    )
    print(f"[unit test] row-major confirmed (ratio {scores[best]:.2f}): OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default="siglip2", help="registry key")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output DIRECTORY (memmap shards + meta.json)")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--keep-sleeps", action="store_true",
                        help="do not filter `sleep` pseudo-actions")
    parser.add_argument("--test-only", action="store_true",
                        help="run the encoder-alignment test and exit")
    args = parser.parse_args()

    spec = encoder_spec(args.encoder)
    grid = (spec.image_size // spec.patch_size, spec.image_size // spec.patch_size)
    n_tokens = grid[0] * grid[1]

    # Dataset is built before the test so the test can use a real image.
    tf = build_transform(spec.image_size)
    ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)
    if args.max_samples and args.max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(args.max_samples)))
    m_max = len(ds)

    encoder = build_encoder(args.encoder, DEVICE)
    # CRITICAL: verify token ordering before doing any expensive encoding.
    test_mask_matches_encoder_tokens(encoder, ds)
    if args.test_only:
        return

    print(f"\nDataset: {m_max} pairs | encoder={args.encoder} grid={grid} "
          f"D={spec.d_model} | letterboxed to {spec.image_size}^2")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allocate memmaps at the upper bound; we record how many rows we actually
    # wrote and the loader slices to that. Cheaper than a pre-pass over the data
    # just to count non-sleep samples.
    z_src_mm = np.lib.format.open_memmap(
        out_dir / "z_src.npy", mode="w+", dtype=np.float16,
        shape=(m_max, n_tokens, spec.d_model))
    z_tgt_mm = np.lib.format.open_memmap(
        out_dir / "z_tgt.npy", mode="w+", dtype=np.float16,
        shape=(m_max, n_tokens, spec.d_model))
    mask_mm = np.lib.format.open_memmap(
        out_dir / "change_mask.npy", mode="w+", dtype=np.uint8,
        shape=(m_max, n_tokens))

    actions: list[str] = []
    cursor = 0
    n_skipped = 0

    with torch.inference_mode():
        for batch in loader:
            texts = list(batch["action_text"])

            if not args.keep_sleeps:
                keep = [i for i, t in enumerate(texts) if not is_sleep(t)]
                n_skipped += len(texts) - len(keep)
                if not keep:
                    continue
                idx = torch.tensor(keep)
                inp = batch["input_image"][idx]
                out = batch["output_image"][idx]
                texts = [texts[i] for i in keep]
            else:
                inp, out = batch["input_image"], batch["output_image"]

            inp_d = inp.to(DEVICE, non_blocking=True)
            out_d = out.to(DEVICE, non_blocking=True)

            z_src = encoder.encode(inp_d).half().cpu().numpy()
            z_tgt = encoder.encode(out_d).half().cpu().numpy()
            # Mask from the same [-1,1] tensors the encoder saw.
            mask = compute_change_mask(inp, out, grid).to(torch.uint8).numpy()

            b = z_src.shape[0]
            z_src_mm[cursor:cursor + b] = z_src
            z_tgt_mm[cursor:cursor + b] = z_tgt
            mask_mm[cursor:cursor + b] = mask
            actions.extend(texts)
            cursor += b

            if cursor % 500 < b:
                print(f"  {cursor}/{m_max - n_skipped} written...", flush=True)

    z_src_mm.flush(); z_tgt_mm.flush(); mask_mm.flush()
    del z_src_mm, z_tgt_mm, mask_mm

    meta = {
        "n_pairs": cursor,
        "n_tokens": n_tokens,
        "d_model": spec.d_model,
        "grid": list(grid),
        "encoder": args.encoder,
        "image_size": spec.image_size,
        "mask_diff_thresh": MASK_DIFF_THRESH,
        "sleeps_filtered": not args.keep_sleeps,
        "n_sleeps_skipped": n_skipped,
        "action_text": actions,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))

    changed = float(np.load(out_dir / "change_mask.npy", mmap_mode="r")[:cursor].mean())
    total_gb = sum(f.stat().st_size for f in out_dir.glob("*.npy")) / 1e9
    print(f"\nWrote {cursor} pairs ({n_skipped} sleep pseudo-actions skipped)")
    print(f"tokens/pair={n_tokens}  D={spec.d_model}")
    print(f"mean changed-token fraction: {changed:.4f}  (~{changed*100:.1f}% of tokens)")
    print(f"cache -> {out_dir}  ({total_gb:.1f} GB across memmaps)")
    print("\nNOTE: memmaps are allocated at the pre-filter upper bound, so rows")
    print(f"[{cursor}:{m_max}] are zero padding. The trainer must slice to")
    print("meta['n_pairs']. Files are sparse on most filesystems, so the")
    print("on-disk size is smaller than the figure above.")


if __name__ == "__main__":
    main()