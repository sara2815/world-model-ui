"""
diagnose_empty_masks.py — why do ~19% of pairs show no pixel change?

The probe reported 77/400 pairs whose change mask is entirely zero at
MASK_DIFF_THRESH=0.05. Those pairs are actively harmful: a model that perfectly
fits them must learn z_tgt = z_src, which is precisely the identity-copy attractor
we are trying to escape. They were in the FLUX training set too.

There are three possible explanations and they need different fixes:

  (a) THRESHOLD TOO HIGH — the edit is real but subtle (antialiased text, a 1px
      border, a faint hover state). diff_max sits just under 0.05. Fix: lower the
      threshold, keep the data.
  (b) CAPTURE BUG — diff_max ~= 0. The pipeline recorded the same frame twice, or
      the action was issued before the UI settled. Fix: drop or re-capture.
  (c) GENUINELY INVISIBLE ACTION — the action had no visual effect (clicking a
      disabled control, a no-op toggle, scrolling at the end of a list). Fix: drop,
      or keep deliberately as "null transition" training signal — but that is a
      decision to make consciously, not by accident.

This script tells you which. Run:
    python diagnose_empty_masks.py --encoder siglip2 --max-samples 400 --dump 12
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import encoder_spec, letterbox_tensor

OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "empty_masks"
THRESHOLDS = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]
PROBE_THRESH = 0.05  # the value the probe used; defines "empty" here


def build_transform(image_size: int) -> transforms.Compose:
    """Identical to probe_encoder.build_transform — the diagnosis must see exactly
    the pixels the probe saw, including letterbox padding."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: letterbox_tensor(t, image_size)),
            transforms.Normalize([0.5], [0.5]),  # -> [-1,1]
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="siglip2")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--dump", type=int, default=12, help="empty-mask pairs to save as PNG")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = encoder_spec(args.encoder)
    grid = (spec.image_size // spec.patch_size,) * 2
    tf = build_transform(spec.image_size)
    ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)

    n = min(args.max_samples, len(ds)) if args.max_samples else len(ds)
    print(f"Scanning {n} pairs at {spec.image_size}^2, token grid {grid}\n")

    diff_max: list[float] = []
    frac_at: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
    actions: list[str] = []

    for i in range(n):
        s = ds[i]
        src, tgt = s["input_image"], s["output_image"]
        d = (tgt - src).abs().amax(dim=0, keepdim=True)  # [1,H,W] on [-1,1] scale
        diff_max.append(d.max().item())
        actions.append(s["action_text"])
        for t in THRESHOLDS:
            m = F.adaptive_max_pool2d((d > t).float().unsqueeze(0), grid)
            frac_at[t].append(m.mean().item())

    diff_max_t = torch.tensor(diff_max)
    empty = torch.tensor(frac_at[PROBE_THRESH]) == 0.0
    n_empty = int(empty.sum())
    print(f"empty at thresh {PROBE_THRESH}: {n_empty}/{n} ({n_empty / n:.1%})\n")

    # ── (a) vs (b): how big is the residual signal in the "empty" pairs? ──────
    print("--- diff_max distribution among EMPTY-mask pairs ---")
    print("If these cluster near 0     -> capture bug (case b), drop them.")
    print("If these cluster at .01-.05 -> threshold too high (case a), lower it.\n")
    e = diff_max_t[empty]
    if n_empty:
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            print(f"  q{q:<5.2f} diff_max = {e.quantile(q).item():.5f}")
        print(f"\n  exactly zero (bit-identical frames): "
              f"{int((e < 1e-6).sum())}/{n_empty}")
        print(f"  below 0.005 (invisible)            : {int((e < 0.005).sum())}/{n_empty}")
        print(f"  in [0.005, 0.05) (real but subtle) : "
              f"{int(((e >= 0.005) & (e < 0.05)).sum())}/{n_empty}")

    # ── threshold sweep: how many pairs survive at each cutoff? ───────────────
    print("\n--- threshold sweep (all pairs) ---")
    print(f"{'thresh':>8} {'n_empty':>8} {'%empty':>8} {'mean frac tokens':>18}")
    for t in THRESHOLDS:
        f = torch.tensor(frac_at[t])
        ne = int((f == 0).sum())
        print(f"{t:8.3f} {ne:8d} {ne / n:8.1%} {f.mean().item():18.4f}")
    print("Pick the largest threshold that recovers most pairs WITHOUT the mean")
    print("token fraction ballooning — that inflation means you're masking in")
    print("compression noise and antialiasing rather than real edits.")

    # ── (c): do the empty pairs cluster on particular actions? ────────────────
    print("\n--- action_text among EMPTY-mask pairs (top 15) ---")
    empty_actions = [a for a, flag in zip(actions, empty.tolist()) if flag]
    for text, cnt in Counter(empty_actions).most_common(15):
        print(f"  {cnt:4d}  {text[:100]}")

    print("\n--- same actions among NON-empty pairs, for comparison ---")
    ok_actions = Counter(a for a, flag in zip(actions, empty.tolist()) if not flag)
    for text, _ in Counter(empty_actions).most_common(8):
        print(f"  empty={Counter(empty_actions)[text]:4d}  ok={ok_actions[text]:4d}  {text[:80]}")
    print("An action type that is ~always empty is a no-op (case c). One that is")
    print("sometimes empty and sometimes not is more likely a capture race (case b).")

    # ── visual dump: look at them ────────────────────────────────────────────
    idx = torch.nonzero(empty).flatten()[: args.dump]
    for j in idx.tolist():
        s = ds[j]
        src, tgt = s["input_image"], s["output_image"]
        d = (tgt - src).abs().amax(dim=0, keepdim=True).repeat(3, 1, 1)
        d_amp = (d * 20).clamp(0, 1)  # 20x gain: makes sub-threshold edits visible
        row = torch.stack([src * 0.5 + 0.5, tgt * 0.5 + 0.5, d_amp])
        save_image(row, OUT_DIR / f"empty_{j:05d}.png", nrow=3)
    print(f"\nSaved {len(idx)} triptychs (input | output | 20x diff) -> {OUT_DIR}")
    print("Look at the third panel. If it is pure black, nothing changed. If you")
    print("can see a faint ghost of a real UI element, the threshold is the problem.")


if __name__ == "__main__":
    main()