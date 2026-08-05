#!/usr/bin/env python3
"""
embedding_metric.py

JEPA-native adaptation of change_metric.py: instead of diffing RGB pixels,
diffs frozen-encoder PATCH EMBEDDINGS. Preserves the same change-region
decomposition (localized correctness vs. background hallucination), scored
in representation space.

Core quantities (patch-level, all in embedding L2 distance):
  - change_embed_score:  score in patches where GT changed (did right thing happen)
  - localization_iou:    IoU between GT changed patches and pred changed patches
  - background_embed_score: score in patches GT did NOT change (penalizes hallucinations)
  - composite_score:     weighted combination [0.0, 1.0]

USAGE
-----
Single comparison with pre-computed numpy tensors [H, W, D]:
    result = compare_tensors(z_before, z_gt, z_pred, thresh=0.15)

Batch / CLI mode (using disk images or saved numpy files):
    python embedding_metric.py --scan-root ./predictions_out --out results.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# ENCODER & FILE LOADERS
# --------------------------------------------------------------------------

def load_latent(path):
    """Loads a latent tensor or encodes an image from disk.
    
    Supports .npy / .npz files directly for JEPA outputs [H, W, D],
    or falls back to an image encoder function.
    """
    path = Path(path)
    if path.suffix in [".npy", ".npz"]:
        data = np.load(path)
        return data["arr_0"] if isinstance(data, np.lib.npyio.NpzFile) else data
    else:
        return encode_image(path)


def encode_image(image_path, model_name="vjepa2"):
    """Placeholder -- replace with your actual frozen encoder call if using images.
    Returns an L2-normalized [H, W, D] numpy array.
    """
    raise NotImplementedError(
        "Wire this up to your frozen JEPA encoder (V-JEPA 2, DINO-WM, etc.) "
        "or pass .npy / .npz files directly."
    )


# --------------------------------------------------------------------------
# METRIC FUNCTIONS (TENSOR-NATIVE)
# --------------------------------------------------------------------------

def compute_embed_change_mask(z_a, z_b, thresh=0.15):
    """Boolean [H, W] mask of patches whose embeddings differ meaningfully."""
    dist = np.linalg.norm(z_a - z_b, axis=-1)  # [H, W]
    return dist > thresh, dist


def iou(mask_a, mask_b):
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


DEFAULT_WEIGHTS = {
    "change_embed_score": 0.40,      # correctness where change should occur
    "localization_iou": 0.30,        # did change land in the right patches
    "background_embed_score": 0.30,  # avoided changing static patches
}


def compare_tensors(z_before, z_gt, z_pred, thresh=0.15, weights=None):
    """Evaluates prediction quality directly on [H, W, D] numpy tensors."""
    weights = weights or DEFAULT_WEIGHTS

    gt_change_mask, _ = compute_embed_change_mask(z_before, z_gt, thresh)
    pred_change_mask, _ = compute_embed_change_mask(z_before, z_pred, thresh)
    static_mask = ~gt_change_mask

    change_area_fraction = float(gt_change_mask.mean())

    # Within the "should have changed" region: how close is pred to gt?
    if gt_change_mask.any():
        change_dist = np.linalg.norm(z_pred - z_gt, axis=-1)[gt_change_mask]
        change_embed_score = float(max(0.0, 1.0 - change_dist.mean() / 2.0))
    else:
        change_embed_score = None

    # Within the "should stay the same" region: how close is pred to before?
    if static_mask.any():
        bg_dist = np.linalg.norm(z_pred - z_before, axis=-1)[static_mask]
        background_embed_score = float(max(0.0, 1.0 - bg_dist.mean() / 2.0))
    else:
        background_embed_score = None

    loc_iou = iou(gt_change_mask, pred_change_mask)

    terms = {
        "change_embed_score": change_embed_score,
        "localization_iou": loc_iou,
        "background_embed_score": background_embed_score,
    }
    valid = {k: v for k, v in terms.items() if v is not None}
    composite = (sum(weights[k] * v for k, v in valid.items()) / sum(weights[k] for k in valid)
                 if valid else None)

    return {
        "change_area_fraction": change_area_fraction,
        "change_embed_score": change_embed_score,
        "background_embed_score": background_embed_score,
        "localization_iou": loc_iou,
        "composite_score": composite,
    }


def calibrate_threshold(val_triplets, quantile=0.90):
    """Calibrates a fixed static threshold based on distances in unchanged regions."""
    diffs = []
    for t in val_triplets:
        z_b = load_latent(t["before"])
        z_g = load_latent(t["gt"])
        dist = np.linalg.norm(z_b - z_g, axis=-1)
        diffs.append(dist.ravel())
    if not diffs:
        return 0.15
    return float(np.quantile(np.concatenate(diffs), quantile))


# --------------------------------------------------------------------------
# BATCH DISCOVERY & CLI
# --------------------------------------------------------------------------

def discover_triplets(root, before_pattern="*before*", gt_pattern="*gt*", pred_pattern="*pred*"):
    root = Path(root)
    triplets = []
    for folder in [p for p in root.rglob("*") if p.is_dir()]:
        befores = list(folder.glob(before_pattern))
        gts = list(folder.glob(gt_pattern))
        preds = list(folder.glob(pred_pattern))
        if befores and gts and preds:
            triplets.append({
                "id": str(folder.relative_to(root)),
                "before": befores[0],
                "gt": gts[0],
                "pred": preds[0],
            })
    return triplets


def run_batch(args):
    triplets = discover_triplets(args.scan_root)
    print(f"Found {len(triplets)} triplets under {args.scan_root}")
    
    thresh = args.thresh
    if args.calibrate and triplets:
        print("Calibrating threshold on dataset noise floor...")
        thresh = calibrate_threshold(triplets[:20])  # Calibrate using up to 20 samples
        print(f"Calibrated threshold: {thresh:.4f}")

    rows = []
    for t in triplets:
        try:
            z_before = load_latent(t["before"])
            z_gt = load_latent(t["gt"])
            z_pred = load_latent(t["pred"])
            result = compare_tensors(z_before, z_gt, z_pred, thresh=thresh)
            row = {"id": t["id"], **result}
        except Exception as e:
            row = {"id": t["id"], "error": str(e)}
        rows.append(row)
        print(f"{t['id']}: composite={row.get('composite_score')}")

    fieldnames = ["id", "change_area_fraction", "change_embed_score",
                  "localization_iou", "background_embed_score", "composite_score", "error"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    valid = [r["composite_score"] for r in rows if r.get("composite_score") is not None]
    if valid:
        print(f"\nMean composite_score over {len(valid)} triplets: {np.mean(valid):.4f}")
    print(f"Wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", help="Path to before image or .npy/.npz tensor")
    ap.add_argument("--gt", help="Path to ground truth image or .npy/.npz tensor")
    ap.add_argument("--pred", help="Path to predicted image or .npy/.npz tensor")
    ap.add_argument("--scan-root", help="Root directory for batch evaluation")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--thresh", type=float, default=0.15, help="Embedding L2-distance threshold")
    ap.add_argument("--calibrate", action="store_true", help="Calibrate threshold on validation data")
    args = ap.parse_args()

    if args.scan_root:
        run_batch(args)
    elif args.before and args.gt and args.pred:
        z_before = load_latent(args.before)
        z_gt = load_latent(args.gt)
        z_pred = load_latent(args.pred)
        result = compare_tensors(z_before, z_gt, z_pred, thresh=args.thresh)
        print(json.dumps(result, indent=2))
    else:
        ap.error("Provide --scan-root, or --before/--gt/--pred")


if __name__ == "__main__":
    main()