"""
predict_from_images.py — run the trained predictor on image FILES, not cache rows.

Everything else in this pipeline reads precomputed embeddings. This script takes
raw PNGs (e.g. output_training/gt_input.png), encodes them live with the same
frozen encoder the cache was built with, and runs the predictor. Useful for
spot-checking arbitrary screenshots that may not be in the cache at all.

YOU MUST SUPPLY AN ACTION. The model predicts f(z_src, action) -> z_tgt. A
screenshot on its own is not a valid input; without an action there is nothing to
predict. Pass --action "click on item 10 (button ...)", or --dataset-index N to
look the string up from the dataset.

STILL NO DECODER. We cannot render a predicted screenshot — the encoder is
one-way. What we render is WHERE the model expects change:

    true change      = || z_tgt - z_src ||     what actually happened
    predicted change = || pred  - z_src ||     what the model expected

The true-change panel is included alongside the predicted one because the
predicted heatmap is uninterpretable on its own — "hot in the corner" only means
something relative to where the change really was.

Run:
    python predict_from_images.py --cache cache/siglip2 \
        --ckpt checkpoints/siglip2_structured_best.pt \
        --input output_training/gt_input.png \
        --output output_training/gt_output.png \
        --dataset-index 0
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wm_encoders import build_encoder, encoder_spec
from inspect_prediction import build_transform
from train_predictor import (
    SENTENCE_MODEL, EmbeddingBank, build_model, parse_action, parse_continuous,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent / "probe_out" / "predictions"


def encode_action(bank: EmbeddingBank, text: str) -> dict:
    """Build the action conditioning dict for an ARBITRARY action string.

    bank.action_batch() only serves cached indices; here the action may be new.
    We reuse the *training* vocabularies so ids mean the same thing they did
    during training. Unknown verbs/elements fall to <unk> (index 0) — flagged
    loudly, because an <unk> element means the model has no idea which UI element
    is being acted on and the prediction is close to meaningless.
    """
    verb, elem = parse_action(text)
    cont = parse_continuous(text)
    print(f"  parsed : verb={verb!r} element={elem!r}")
    print(f"  cont   : {[round(x, 3) for x in cont]}")

    t_id = bank.type_vocab.get(verb, 0)
    e_id = bank.elem_vocab.get(elem, 0)
    if verb is not None and t_id == 0:
        print(f"  !!  verb {verb!r} is OUT OF VOCABULARY -> <unk>")
    if elem is not None and e_id == 0:
        print(f"  !!  element {elem!r} is OUT OF VOCABULARY -> <unk>. The model "
              f"never saw this element in training; the prediction is unreliable.")
    parsed_ok = verb is not None and elem is not None

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(SENTENCE_MODEL, device=DEVICE)
    temb = st.encode([text], convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=False).float().to(DEVICE)

    return {
        "type_id": torch.tensor([t_id], device=DEVICE),
        "elem_id": torch.tensor([e_id], device=DEVICE),
        "parsed": torch.tensor([parsed_ok], device=DEVICE),
        "cont": torch.tensor([cont], dtype=torch.float32, device=DEVICE),
        "text_emb": temb,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True,
                    help="cache dir — used for the vocabularies and encoder name")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="before screenshot")
    ap.add_argument("--output", type=Path, default=None,
                    help="after screenshot (ground truth). Optional, but without "
                         "it there is no true-change panel to compare against.")
    ap.add_argument("--action", type=str, default=None, help="action text")
    ap.add_argument("--dataset-index", type=int, default=None,
                    help="look the action string up from the dataset instead")
    args = ap.parse_args()

    if args.action is None and args.dataset_index is None:
        ap.error("supply --action \"...\" or --dataset-index N. The model needs an "
                 "action; a screenshot alone is not a valid input.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    bank = EmbeddingBank(args.cache, ckpt["mode"])
    assert len(bank.elem_vocab) == len(ckpt["elem_vocab"]), (
        f"element vocab {len(bank.elem_vocab)} != checkpoint's "
        f"{len(ckpt['elem_vocab'])} — cache and checkpoint disagree."
    )
    model = build_model(bank, ckpt["mode"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint {args.ckpt.name} | epoch {ckpt['epoch']} | mode {ckpt['mode']}")

    enc_name = json.loads((args.cache / "meta.json").read_text())["encoder"]
    spec = encoder_spec(enc_name)
    encoder = build_encoder(enc_name, DEVICE)
    tf = build_transform(spec.image_size)

    # ── action ───────────────────────────────────────────────────────────────
    action = args.action
    if action is None:
        from data.dataset import WorldModelDataset
        ds = WorldModelDataset(use_annotated_image=True, transform=tf, target_transform=tf)
        action = ds[args.dataset_index]["action_text"]
        print(f"\nAction from dataset[{args.dataset_index}]:")
    else:
        print("\nAction (supplied):")
    print(f"  {action}")
    act = encode_action(bank, action)

    # ── encode the images ────────────────────────────────────────────────────
    src = tf(Image.open(args.input).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_src = encoder.encode(src)
        pred = model(z_src, act)
    pred_delta = (pred - z_src).norm(dim=-1)[0].cpu().numpy()

    z_tgt = true_delta = None
    if args.output is not None and args.output.exists():
        tgt = tf(Image.open(args.output).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z_tgt = encoder.encode(tgt)
        true_delta = (z_tgt - z_src).norm(dim=-1)[0].cpu().numpy()

    rows, cols = spec.image_size // spec.patch_size, spec.image_size // spec.patch_size
    pm = pred_delta.reshape(rows, cols)

    print(f"\n  mean predicted delta : {pred_delta.mean():.3f}")
    if true_delta is not None:
        corr = float(np.corrcoef(true_delta, pred_delta)[0, 1])
        print(f"  mean true delta      : {true_delta.mean():.3f}")
        print(f"  spatial correlation  : {corr:+.3f}")
        if pred_delta.mean() < 0.2 * true_delta.mean():
            print("  !!  Predicted delta is far smaller than the true delta — the "
                  "model is hedging toward the copy solution.")
    else:
        corr = float("nan")

    # ── figure ───────────────────────────────────────────────────────────────
    src_img = (src[0].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    ext = [0, spec.image_size, spec.image_size, 0]
    n_panels = 4 if true_delta is not None else 2
    fig, ax = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.4))

    ax[0].imshow(src_img); ax[0].set_title("input (before)")
    if true_delta is not None:
        tgt_img = (tgt[0].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        ax[1].imshow(tgt_img); ax[1].set_title("output (after, ground truth)")
        ax[2].imshow(src_img)
        ax[2].imshow(true_delta.reshape(rows, cols), extent=ext, alpha=0.6, cmap="hot")
        ax[2].set_title("TRUE change")
        ax[3].imshow(src_img); ax[3].imshow(pm, extent=ext, alpha=0.6, cmap="hot")
        ax[3].set_title(f"PREDICTED change (corr {corr:+.2f})")
    else:
        ax[1].imshow(src_img); ax[1].imshow(pm, extent=ext, alpha=0.6, cmap="hot")
        ax[1].set_title("PREDICTED change")
    for a in ax:
        a.axis("off")

    fig.suptitle(action[:120], fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / f"{args.ckpt.stem}_{args.input.stem}.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()