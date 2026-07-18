"""
find_changing_sleeps.py — locate `sleep` actions whose screenshot DID change.

A sleep is a wait, not an action: nothing should change during one. When the
frame changes anyway, something asynchronous landed — a chart re-render, a fetch
resolving, an animation settling. The change got recorded as the effect of
*sleeping*.

The reason this is worth cataloguing: if an effect landed during the sleep, then
the action that *caused* it is the previous event, and that event's own pair
would show a stale "after". That is action-effect misattribution, and it teaches
the model that the real action does nothing.

CAVEAT ON HOW MUCH THIS PROVES: in a 400-sample scan only 2 of ~53 sleeps showed
a change (~4%). That is a handful, not a demonstrated systematic problem. This
script exists to size the effect over the full dataset and to let you eyeball the
cases — not to prejudge them.

Outputs:
  - a printed list of dataset indices, sorted by change magnitude
  - a CSV of the same, for filtering later
  - triptych PNGs (input | output | 20x diff) for the top-K changers

Run:
    python find_changing_sleeps.py --encoder siglip2
    python find_changing_sleeps.py --encoder siglip2 --context   # also dump pair i-1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data.dataset import WorldModelDataset

from wm_encoders import encoder_spec, letterbox_tensor

OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "changing_sleeps"
MASK_DIFF_THRESH = 0.05
SLEEP_PREFIX = "sleep"


def build_transform(image_size: int) -> transforms.Compose:
    """Same transform as the probe, so magnitudes are comparable across scripts."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: letterbox_tensor(t, image_size)),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def _triptych(sample, path: Path, gain: float = 20.0) -> None:
    """Save input | output | amplified-diff side by side."""
    src, tgt = sample["input_image"], sample["output_image"]
    d = (tgt - src).abs().amax(dim=0, keepdim=True).repeat(3, 1, 1)
    row = torch.stack([src * 0.5 + 0.5, tgt * 0.5 + 0.5, (d * gain).clamp(0, 1)])
    save_image(row, path, nrow=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="siglip2", help="only for grid/resolution")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--dump", type=int, default=20, help="top-K changers to save as PNG")
    ap.add_argument("--context", action="store_true",
                    help="also dump pair i-1 (ASSUMES dataset order == trajectory order)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = encoder_spec(args.encoder)
    grid = (spec.image_size // spec.patch_size,) * 2
    tf = build_transform(spec.image_size)
    ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)

    n = min(args.max_samples, len(ds)) if args.max_samples else len(ds)
    print(f"Scanning {n} pairs for actions starting with '{SLEEP_PREFIX}'...\n")

    rows: list[dict] = []
    n_sleep = 0
    for i in range(n):
        s = ds[i]
        action = s["action_text"]
        if not action.strip().lower().startswith(SLEEP_PREFIX):
            continue
        n_sleep += 1

        d = (s["output_image"] - s["input_image"]).abs().amax(dim=0, keepdim=True)
        dmax = d.max().item()
        frac = F.adaptive_max_pool2d((d > MASK_DIFF_THRESH).float().unsqueeze(0),
                                     grid).mean().item()
        if frac > 0.0:
            rows.append({"index": i, "action": action,
                         "diff_max": round(dmax, 5),
                         "frac_tokens_changed": round(frac, 5)})

    rows.sort(key=lambda r: -r["frac_tokens_changed"])

    print(f"sleep pairs found      : {n_sleep}")
    print(f"...of which CHANGED    : {len(rows)}  "
          f"({len(rows) / max(n_sleep, 1):.1%} of sleeps)")
    print(f"...clean no-ops (drop) : {n_sleep - len(rows)}\n")

    if rows:
        print(f"{'idx':>7} {'frac_tok':>9} {'diff_max':>9}  action")
        for r in rows[:40]:
            print(f"{r['index']:7d} {r['frac_tokens_changed']:9.4f} "
                  f"{r['diff_max']:9.4f}  {r['action'][:60]}")
        if len(rows) > 40:
            print(f"  ... and {len(rows) - 40} more (see CSV)")

    csv_path = OUT_DIR / "changing_sleeps.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "action", "diff_max",
                                          "frac_tokens_changed"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    for r in rows[: args.dump]:
        i = r["index"]
        _triptych(ds[i], OUT_DIR / f"sleep_{i:05d}_frac{r['frac_tokens_changed']:.3f}.png")
        if args.context and i > 0:
            prev = ds[i - 1]
            _triptych(prev, OUT_DIR / f"sleep_{i:05d}_PREV_{i-1:05d}.png")
            print(f"  [{i}] prev action was: {prev['action_text'][:80]}")
    print(f"\nSaved {min(args.dump, len(rows))} triptychs -> {OUT_DIR}")

    if args.context:
        print("\nWhat to look for in the *_PREV_* files: if the previous pair's diff")
        print("panel is black while the sleep's is not, the previous action's effect")
        print("landed during the sleep — misattribution. If the previous pair also")
        print("shows a change, the UI was simply still animating; harmless.")
    else:
        print("\nRe-run with --context to also dump the preceding pair, which is the")
        print("suspected true cause of each change. Note this assumes dataset order")
        print("follows trajectory order — verify that before trusting it.")


if __name__ == "__main__":
    main()