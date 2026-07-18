#!/usr/bin/env python3
"""

-----
1. Diff the BEFORE image against the GT (after) image -> this tells us where
   change was supposed to happen. That region is left unmasked, everythinge 
   else gets masked out (down-weighted) because
   matching the static background exactly is not the interesting part.

2. Within that "should have changed" region, compare PRED to GT directly
   (SSIM + RMSE) -> did the right thing happen, with the right content.

3. Within the inverse region ("should have stayed the same"), compare PRED
   to BEFORE -> penalizes a model that hallucinates changes where nothing
   was supposed to change.

4. Independently of pixel content, also diff BEFORE vs PRED the same way
   GT vs BEFORE was diffed, and compute the difference between the two change
   masks -> assign different score combinations. 

A single composite score combines all of these, with configurable weights.

USAGE
-----
Single triplet:
    python change_metric.py --before before.png --gt gt.png --pred pred.png \
        [--viz viz_out.png] [--json out.json]

Batch (many triplets from a JSON list):
    python change_metric.py --triplets triplets.json --out results.csv

    triplets.json format:
    [
      {"id": "17_step0", "before": "path/before.png", "gt": "path/gt.png", "pred": "path/pred.png"},
      ...
    ]
"""

import argparse
import json
import csv
import sys
from pathlib import Path

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


# --------------------------------------------------------------------------

def load_image(path, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def compute_change_mask(img_a, img_b, thresh=25, dilate_px=5, min_area=40):
    """Binary mask (uint8, 0/255) of regions where img_a and img_b differ.

    thresh:     per-pixel grayscale-diff threshold to count as "changed"
    dilate_px:  grow the mask by this many pixels (tolerance for anti-
                aliasing / 1-2px rendering jitter between runs)
    min_area:   drop connected components smaller than this many pixels
                (removes compression-noise speckle)
    """
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)

    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # remove small noise
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        mask = clean

    # give some spatial tolerance around the true change region
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
        mask = cv2.dilate(mask, k)

    return mask  # uint8, 0 or 255


def masked_rmse(img_a, img_b, mask):
    """RMSE between two RGB images, restricted to mask==255. Returns value
    in [0, 255] (or None if the mask is empty)."""
    m = mask.astype(bool)
    if not m.any():
        return None
    a = img_a[m].astype(np.float64)
    b = img_b[m].astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def masked_ssim(img_a, img_b, mask):
    """Mean SSIM between two RGB images, restricted to mask==255. Returns
    value in [-1, 1] (or None if the mask is empty)."""
    m = mask.astype(bool)
    if not m.any():
        return None
    _, smap = ssim(img_a, img_b, channel_axis=-1, full=True)
    smap = smap.mean(axis=-1)  # average the 3 channel SSIM maps
    return float(smap[m].mean())


def iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0  # both agree nothing changed
    return float((a & b).sum() / union)


# --------------------------------------------------------------------------
# Composite metric
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "change_ssim": 0.35,     # correctness of content within the changed region
    "change_rmse": 0.20,     # pixel accuracy within the changed region
    "localization_iou": 0.30,  # did the change happen in the right place at all
    "background_ssim": 0.15,  # did it avoid hallucinating changes elsewhere
}


def compare(before, gt, pred, thresh=25, dilate_px=5, min_area=40, weights=None):
    """Compute the full change-aware metric set for one (before, gt, pred) triplet.

    Images must be numpy RGB arrays of matching shape (pred/gt are resized
    to match `before` if needed).
    """
    weights = weights or DEFAULT_WEIGHTS
    h, w = before.shape[:2]
    if gt.shape[:2] != (h, w):
        gt = cv2.resize(gt, (w, h))
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred, (w, h))

    gt_change_mask = compute_change_mask(before, gt, thresh, dilate_px, min_area)
    pred_change_mask = compute_change_mask(before, pred, thresh, dilate_px, min_area)
    static_mask = cv2.bitwise_not(gt_change_mask)

    change_area_fraction = float((gt_change_mask > 0).mean())

    c_ssim = masked_ssim(pred, gt, gt_change_mask)
    c_rmse = masked_rmse(pred, gt, gt_change_mask)
    bg_ssim = masked_ssim(pred, before, static_mask)
    loc_iou = iou(gt_change_mask, pred_change_mask)

    # normalize rmse (0-255) into a similarity-style [0,1] score, higher=better
    c_rmse_score = None if c_rmse is None else max(0.0, 1.0 - c_rmse / 255.0)

    # composite: only include terms we could actually compute (mask non-empty)
    terms = {
        "change_ssim": c_ssim,
        "change_rmse": c_rmse_score,
        "localization_iou": loc_iou,
        "background_ssim": bg_ssim,
    }
    valid = {k: v for k, v in terms.items() if v is not None}
    if valid:
        wsum = sum(weights[k] for k in valid)
        composite = sum(weights[k] * v for k, v in valid.items()) / wsum
    else:
        composite = None

    return {
        "change_area_fraction": change_area_fraction,
        "change_ssim": c_ssim,
        "change_rmse": c_rmse,
        "localization_iou": loc_iou,
        "background_ssim": bg_ssim,
        "composite_score": composite,
        "_gt_change_mask": gt_change_mask,
        "_pred_change_mask": pred_change_mask,
    }


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def save_visualization(before, gt, pred, result, out_path):
    gt_mask = result["_gt_change_mask"]
    pred_mask = result["_pred_change_mask"]

    def overlay(img, mask, color):
        out = img.copy()
        colored = np.zeros_like(img)
        colored[:] = color
        m = mask.astype(bool)
        out[m] = cv2.addWeighted(img, 0.4, colored, 0.6, 0)[m]
        return out

    gt_overlay = overlay(gt, gt_mask, (255, 0, 0))       # red = should-change region
    pred_overlay = overlay(pred, pred_mask, (0, 128, 255))  # orange = predicted-change region

    h, w = before.shape[:2]
    pad = 10
    canvas = np.full((h + 30, w * 4 + pad * 3, 3), 255, dtype=np.uint8)
    imgs = [before, gt_overlay, pred, pred_overlay]
    labels = ["before", "gt (change mask)", "pred", "pred (change mask)"]
    for i, (im, label) in enumerate(zip(imgs, labels)):
        x = i * (w + pad)
        canvas[30:30 + h, x:x + w] = im
        cv2.putText(canvas, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _strip_internal(result):
    return {k: v for k, v in result.items() if not k.startswith("_")}


def run_single(args):
    before = load_image(args.before)
    gt = load_image(args.gt)
    pred = load_image(args.pred)

    result = compare(before, gt, pred, thresh=args.thresh, dilate_px=args.dilate, min_area=args.min_area)

    if args.viz:
        save_visualization(before, gt, pred, result, args.viz)
        print(f"Saved visualization to {args.viz}")

    clean = _strip_internal(result)
    print(json.dumps(clean, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(clean, indent=2))
        print(f"Saved metrics to {args.json}")


def discover_triplets(root, before_name="before.png", gt_name="gt_after.png", pred_name="pred_after.png"):
    """Recursively find every folder under root containing all three of
    before_name/gt_name/pred_name and return them as a triplet list."""
    root = Path(root)
    triplets = []
    for before_path in root.rglob(before_name):
        folder = before_path.parent
        gt_path = folder / gt_name
        pred_path = folder / pred_name
        if gt_path.exists() and pred_path.exists():
            triplets.append({
                "id": str(folder.relative_to(root)),
                "before": str(before_path),
                "gt": str(gt_path),
                "pred": str(pred_path),
            })
    return triplets


def run_batch(args):
    if args.scan_root:
        triplets = discover_triplets(args.scan_root, args.before_name, args.gt_name, args.pred_name)
        print(f"Found {len(triplets)} triplets under {args.scan_root}")
        if not triplets:
            print("No matching folders found - check --before-name/--gt-name/--pred-name "
                  "match your actual filenames.")
            return
    else:
        triplets = json.loads(Path(args.triplets).read_text())
    rows = []
    for t in triplets:
        tid = t.get("id", t.get("pred", "?"))
        try:
            before = load_image(t["before"])
            gt = load_image(t["gt"])
            pred = load_image(t["pred"])
            result = compare(before, gt, pred, thresh=args.thresh, dilate_px=args.dilate, min_area=args.min_area)
            row = {"id": tid, **_strip_internal(result)}
        except Exception as e:
            row = {"id": tid, "error": str(e)}
        rows.append(row)
        print(f"{tid}: composite={row.get('composite_score')}")

    fieldnames = ["id", "change_area_fraction", "change_ssim", "change_rmse",
                  "localization_iou", "background_ssim", "composite_score", "error"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    valid_scores = [r["composite_score"] for r in rows if r.get("composite_score") is not None]
    if valid_scores:
        print(f"\nMean composite_score over {len(valid_scores)} triplets: {np.mean(valid_scores):.4f}")
    print(f"Wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before")
    ap.add_argument("--gt")
    ap.add_argument("--pred")
    ap.add_argument("--viz", help="Save a side-by-side visualization PNG")
    ap.add_argument("--json", help="Save single-triplet metrics to this JSON path")
    ap.add_argument("--triplets", help="JSON list of {id, before, gt, pred} for batch mode")
    ap.add_argument("--scan-root", help="Recursively scan this folder for subfolders containing "
                                         "before/gt_after/pred_after images (batch mode)")
    ap.add_argument("--before-name", default="before.png", help="Filename to look for as the 'before' image")
    ap.add_argument("--gt-name", default="gt_after.png", help="Filename to look for as the ground-truth image")
    ap.add_argument("--pred-name", default="pred_after.png", help="Filename to look for as the predicted image")
    ap.add_argument("--out", default="results.csv", help="CSV output path for batch mode")
    ap.add_argument("--thresh", type=int, default=25, help="Per-pixel diff threshold (0-255)")
    ap.add_argument("--dilate", type=int, default=5, help="Mask dilation radius in px")
    ap.add_argument("--min-area", type=int, default=40, help="Drop change blobs smaller than this (px)")
    args = ap.parse_args()

    if args.triplets or args.scan_root:
        run_batch(args)
    elif args.before and args.gt and args.pred:
        run_single(args)
    else:
        ap.error("Provide --scan-root or --triplets for batch mode, or --before/--gt/--pred for a single comparison")


if __name__ == "__main__":
    main()