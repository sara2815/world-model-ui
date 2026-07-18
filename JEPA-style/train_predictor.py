"""
train_predictor.py — the action-conditioned latent predictor (the "world model").

Loads ONLY the cached embeddings from precompute_embeddings.py. The vision
encoder is never imported here; everything it produced (z_src, z_tgt, change
masks) is on disk. Given z_src tokens and an action, we predict z_tgt tokens.

The four ideas that make or break this model
--------------------------------------------
1. MASK-WEIGHTED LOSS. ~93% of tokens are identical before/after, so an
   unweighted loss is minimized by the trivial "copy z_src" solution. We upweight
   the tokens that actually changed (from the cached pixel-derived mask).

2. RESIDUAL PREDICTION. `pred = z_src + delta`, delta head zero-initialized, so
   the model starts *exactly at* the identity baseline and only has to learn the
   change.

3. THE SPATIAL POINTER. A purely symbolic action is a global conditioning signal
   with no spatial anchor: nothing tells the model *where* item 10 is, so it must
   ground the reference itself from the annotated screenshot. That missing
   information is what produces correctly-shaped but mislocalized edits. Where
   the action text carries explicit pixel coordinates (`click on (269, 528)`,
   `drag ... to (1147, 279)`), we feed them in directly as continuous features.

4. THE RIGHT RETRIEVAL METRIC. Ranking candidate z_tgt by distance to the
   prediction is nearly free: because pred ~= z_src and different samples are
   different screenshots, even the identity baseline scores near-perfect top-1.
   The informative version ranks on the *delta*: match (pred - z_src) against
   (z_tgt - z_src). Identity predicts a zero delta for everything, so it sits at
   chance — which is what a null baseline should do.

Run:
    python train_predictor.py --cache cache/siglip2 --overfit-one
    python train_predictor.py --cache cache/siglip2
    python train_predictor.py --cache cache/siglip2 --action-encoding text
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ── Config constants ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

N_LAYERS = 6
N_HEADS = 8
MLP_RATIO = 4

ACTION_ENCODING = "structured"          # "structured" | "text"
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MASK_WEIGHT = 8.0                        # changed tokens get weight (1 + this)
VAL_FRAC = 0.15                          # held-out *samples* (not tokens)

BATCH_SIZE = 16
EVAL_BATCH = 8                           # eval must be chunked; see evaluate()
NUM_EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0

SAVE_EVERY = 5                           # epochs
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
EVAL_SUBSET = 512                        # cap samples used for per-epoch metrics
RETRIEVAL_POOL = 512                     # candidates in the retrieval ranking

OVERFIT_STEPS = 400
OVERFIT_TOL = 1e-2                       # weighted loss must fall below this

SCREEN_W, SCREEN_H = 1280, 720           # native capture size, pre-letterbox
SCROLL_SCALE = 1000.0                    # rough max scroll magnitude
N_CONT_FEATS = 5                         # [x, y, scroll, has_coord, has_scroll]


# ── Action parsing ───────────────────────────────────────────────────────────
# The recorder emits compound events: `move to item 10 (...); click on item 10
# (...)`, `click on item 2 (button "Create"); sleep 1.00s`, `click on (269, 528)`.
# Three things follow from that format:
#
#   * The ACTION IS THE LAST non-sleep clause. `move to X; click on X` is a
#     click; the move is just mouse travel. Taking the first verb mislabels it.
#   * TRAILING SLEEPS ARE SETTLE TIME, not events. The recorder already bundles
#     each action with its wait — which is exactly the pairing we want. Only
#     standalone `sleep N.NNs` rows are pseudo-actions, and precompute drops
#     those.
#   * RAW INTEGERS ARE NOT ELEMENT IDS. An earlier parser matched the first
#     integer anywhere, which turned sleep durations ("1" from `sleep 1.00s`) and
#     pixel coordinates ("269" from `click on (269, 528)`) into one-off
#     vocabulary entries — inflating the element vocab to 764 for 7933 samples,
#     most seen exactly once. Coordinates and scroll distances are continuous and
#     belong in parse_continuous(), not in an embedding table where 269 and 270
#     are unrelated symbols.
_SLEEP_CLAUSE = re.compile(r"^\s*sleep\s+[\d.]+\s*s?\s*$", re.I)
_VERB_RE = re.compile(r"^\s*([a-zA-Z_]+)")
_ITEM_RE = re.compile(r"\bitem\s+(\d+)\b", re.I)
_KEY_RE = re.compile(r"^\s*(?:press|hotkey)\s+(\S+)", re.I)
_COORD_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
_SCROLL_RE = re.compile(r"scroll\s+\w+\s+by\s+(-?\d+)", re.I)
_QUOTED_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def primary_clause(text: str) -> str:
    """The action is the LAST non-sleep clause of a compound event."""
    parts = [c for c in (text or "").split(";") if not _SLEEP_CLAUSE.match(c)]
    return parts[-1].strip() if parts else (text or "").strip()


def parse_action(text: str) -> tuple[Optional[str], Optional[str]]:
    """Parse into ``(action_type, element_id)`` as strings, or ``(None, ...)``.

    Element handles resolve in priority order: `item N` (the annotated
    bounding-box id) first, then key names, then quoted literals, then the
    *categories* `coord` / `scroll`. A sample counts as parsed only if both a
    verb and a handle are found; otherwise the caller falls back to the text
    encoder for that sample.
    """
    clause = primary_clause(text)
    m_verb = _VERB_RE.match(clause)
    action_type = m_verb.group(1).lower() if m_verb else None

    element_id = None
    if (m := _ITEM_RE.search(clause)):
        element_id = f"item{m.group(1)}"
    elif (m := _KEY_RE.match(clause)):
        element_id = f"key:{m.group(1).lower().strip('.,;')}"
    elif (m := _QUOTED_RE.search(clause)):
        element_id = "text:" + (m.group(1) or m.group(2) or "").lower()[:32]
    elif _COORD_RE.search(clause):
        element_id = "coord"
    elif _SCROLL_RE.search(clause):
        element_id = "scroll"
    return action_type, element_id


def parse_continuous(text: str) -> list[float]:
    """Continuous action features: ``[x, y, scroll, has_coord, has_scroll]``.

    This is the spatial pointer described in the module docstring. Coordinates
    are normalized by the native capture size; the has_* flags let the model tell
    "coordinate at the origin" from "no coordinate given".
    """
    clause = primary_clause(text)
    f = [0.0] * N_CONT_FEATS
    if (m := _COORD_RE.search(clause)):
        f[0] = int(m.group(1)) / SCREEN_W
        f[1] = int(m.group(2)) / SCREEN_H
        f[3] = 1.0
    if (m := _SCROLL_RE.search(clause)):
        f[2] = int(m.group(1)) / SCROLL_SCALE
        f[4] = 1.0
    return f


class ActionEncoder(nn.Module):
    """Turn a batch of actions into a single conditioning token ``[B, 1, D]``.

      * "text": project a frozen sentence embedding. Deliberately gets NO
        continuous features — otherwise the text-vs-structured comparison would
        not isolate the effect of the structured encoding.
      * "structured": embed (action_type, element_id), add the projected
        continuous features, and project; samples that failed to parse fall back
        to the text projection so none are dropped.
    """

    def __init__(self, mode: str, d_model: int, n_types: int, n_elems: int,
                 text_dim: int) -> None:
        super().__init__()
        self.mode = mode
        half = d_model // 2
        self.type_emb = nn.Embedding(n_types, half)
        self.elem_emb = nn.Embedding(n_elems, d_model - half)
        self.struct_proj = nn.Linear(d_model, d_model)
        self.cont_proj = nn.Linear(N_CONT_FEATS, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)

    def forward(self, batch: dict) -> Tensor:
        text_tok = self.text_proj(batch["text_emb"])  # [B, D]
        if self.mode == "text":
            return text_tok.unsqueeze(1)

        struct = torch.cat(
            [self.type_emb(batch["type_id"]), self.elem_emb(batch["elem_id"])], dim=-1
        )
        struct_tok = self.struct_proj(struct) + self.cont_proj(batch["cont"])
        parsed = batch["parsed"].unsqueeze(-1)  # [B, 1]
        return torch.where(parsed, struct_tok, text_tok).unsqueeze(1)


# ── Predictor transformer ────────────────────────────────────────────────────
class Block(nn.Module):
    """Pre-norm transformer block using SDPA (FlashAttention-2-free, sm_70 safe)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, l, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, l, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each [B, heads, L, hd]
        attn = F.scaled_dot_product_attention(q, k, v)    # SDPA, no FlashAttn-2
        attn = attn.transpose(1, 2).reshape(b, l, d)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


class Predictor(nn.Module):
    """[action] + z_src tokens -> predicted z_tgt tokens.

    Predicts a residual over z_src with a zero-initialized delta head, so the
    network's initial output is *exactly* the identity baseline. Training then
    only has to move the tokens that change, and the ~93% unchanged tokens stay
    correct for free.
    """

    def __init__(self, d_model: int, n_tokens: int, n_layers: int, n_heads: int,
                 action_encoder: ActionEncoder) -> None:
        super().__init__()
        self.action_encoder = action_encoder
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.action_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, MLP_RATIO) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, z_src: Tensor, action_batch: dict) -> Tensor:
        a = self.action_encoder(action_batch) + self.action_pos  # [B, 1, D]
        x = z_src + self.pos_emb                                  # [B, N, D]
        seq = torch.cat([a, x], dim=1)                            # [B, N+1, D]
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        return z_src + self.delta_head(seq[:, 1:, :])             # residual


# ── Loss & metrics ───────────────────────────────────────────────────────────
def per_token_smooth_l1(pred: Tensor, tgt: Tensor) -> Tensor:
    """Smooth-L1 averaged over the feature dim -> ``[B, N]`` per-token loss."""
    return F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=-1)


def weighted_loss(pred: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
    """Mask-weighted smooth-L1. ``w = 1 + MASK_WEIGHT * mask``.

    An all-zero mask yields uniform w = 1, the safe fallback, so no division by
    zero is possible here.
    """
    ptl = per_token_smooth_l1(pred, tgt)
    w = 1.0 + MASK_WEIGHT * mask
    return (w * ptl).sum() / w.sum().clamp_min(1e-8)


# ── Data container ───────────────────────────────────────────────────────────
class EmbeddingBank:
    """Serves cached embeddings from disk-backed memmaps.

    The cache is ~40 GB (8k pairs x 1024 tokens x 1152 dims x 2 bytes x 2). It is
    never loaded into RAM: `np.load(mmap_mode='r')` pages in only the rows a batch
    touches, and rows are converted to fp32 torch tensors per batch.
    """

    def __init__(self, cache_dir: Path, mode: str) -> None:
        meta = json.loads((cache_dir / "meta.json").read_text())
        self.m = int(meta["n_pairs"])
        self.n_tokens = int(meta["n_tokens"])
        self.d_model = int(meta["d_model"])
        self.grid = tuple(meta["grid"])
        self.action_text: list[str] = meta["action_text"]
        self.cache_dir = cache_dir

        # Memmaps are allocated at the pre-filter upper bound; slice to n_pairs.
        self._z_src = np.load(cache_dir / "z_src.npy", mmap_mode="r")[: self.m]
        self._z_tgt = np.load(cache_dir / "z_tgt.npy", mmap_mode="r")[: self.m]
        self._mask = np.load(cache_dir / "change_mask.npy", mmap_mode="r")[: self.m]
        assert len(self.action_text) == self.m, (
            f"meta action_text has {len(self.action_text)} entries but n_pairs="
            f"{self.m} — cache is inconsistent"
        )

        self.mode = mode
        self._build_text_cache()
        self._build_structured(mode)

    def _fetch(self, idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Page in one batch: (z_src, z_tgt, mask) as fp32 CUDA tensors."""
        i = np.asarray(idx.cpu().numpy())
        zs = torch.from_numpy(np.array(self._z_src[i])).float().to(DEVICE)
        zt = torch.from_numpy(np.array(self._z_tgt[i])).float().to(DEVICE)
        mk = torch.from_numpy(np.array(self._mask[i])).float().to(DEVICE)
        return zs, zt, mk

    def _build_text_cache(self) -> None:
        """Encode action texts with a frozen sentence encoder once, then cache.

        Needed for "text" mode and as the structured-mode fallback, so always
        computed. The action vocabulary is small, so this is cheap.
        """
        text_path = self.cache_dir / "action_text_emb.pt"
        if text_path.exists():
            cached = torch.load(text_path, map_location="cpu")
            if cached.get("model") == SENTENCE_MODEL and cached["emb"].shape[0] == self.m:
                self.text_emb = cached["emb"].float()
                self.text_dim = self.text_emb.shape[1]
                return
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(SENTENCE_MODEL, device=DEVICE)
        emb = st.encode(
            self.action_text, convert_to_tensor=True, show_progress_bar=False,
            normalize_embeddings=True,
        ).cpu().float()
        self.text_emb = emb
        self.text_dim = emb.shape[1]
        torch.save({"model": SENTENCE_MODEL, "emb": emb}, text_path)

    def _build_structured(self, mode: str) -> None:
        """Parse actions; build (type, element) vocabularies, flags, and feats."""
        self.type_vocab = {"<unk>": 0}
        self.elem_vocab = {"<unk>": 0}
        type_ids, elem_ids, parsed, cont = [], [], [], []
        n_ok = 0
        for text in self.action_text:
            at, el = parse_action(text)
            cont.append(parse_continuous(text))
            ok = at is not None and el is not None
            parsed.append(ok)
            if ok:
                n_ok += 1
                self.type_vocab.setdefault(at, len(self.type_vocab))
                self.elem_vocab.setdefault(el, len(self.elem_vocab))
                type_ids.append(self.type_vocab[at])
                elem_ids.append(self.elem_vocab[el])
            else:
                type_ids.append(0)
                elem_ids.append(0)
        self.type_ids = torch.tensor(type_ids, dtype=torch.long)
        self.elem_ids = torch.tensor(elem_ids, dtype=torch.long)
        self.parsed = torch.tensor(parsed, dtype=torch.bool)
        self.cont = torch.tensor(cont, dtype=torch.float32)       # [M, 5]
        self.parse_rate = n_ok / max(self.m, 1)

        if mode == "structured":
            n_coord = int(self.cont[:, 3].sum())
            n_scroll = int(self.cont[:, 4].sum())
            print(f"[action] parse success rate: {self.parse_rate:.1%} "
                  f"({n_ok}/{self.m}) | types={len(self.type_vocab)} "
                  f"elements={len(self.elem_vocab)}")
            print(f"[action] continuous features: {n_coord} with coordinates, "
                  f"{n_scroll} with scroll distance")
            if len(self.elem_vocab) > 0.2 * self.m:
                warnings.warn(
                    f"element vocab ({len(self.elem_vocab)}) is large relative to "
                    f"the dataset ({self.m}) — most element embeddings would be "
                    f"seen only a handful of times and cannot generalize. Check "
                    f"the parser for numeric leakage."
                )
            n_fallback = self.m - n_ok
            if n_fallback:
                warnings.warn(f"{n_fallback} sample(s) failed structured parsing; "
                              f"falling back to the text encoder for those.")

    def action_batch(self, idx: Tensor, action_source: Optional[Tensor] = None) -> dict:
        """Assemble action conditioning for indices ``idx``.

        ``action_source`` lets us pair z_src of one sample with the action of
        another (used by the action-sensitivity probe). Defaults to ``idx``.
        """
        src = idx if action_source is None else action_source
        return {
            "type_id": self.type_ids[src].to(DEVICE),
            "elem_id": self.elem_ids[src].to(DEVICE),
            "parsed": self.parsed[src].to(DEVICE),
            "cont": self.cont[src].to(DEVICE),
            "text_emb": self.text_emb[src].to(DEVICE),
        }


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: Predictor, bank: EmbeddingBank, idx: Tensor) -> dict:
    """Chunked metrics for a split: model vs identity, changed vs unchanged.

    Chunked because a full-batch forward over even 1k samples would be
    1000 x 1024 x 1152 x 4 bytes ~= 4.7 GB *per tensor*, and we need several.
    """
    model.eval()
    acc = {k: 0.0 for k in ("m_w_num", "m_w_den", "i_w_num", "i_w_den",
                            "m_ch", "i_ch", "n_ch", "m_un", "i_un", "n_un")}
    for start in range(0, idx.numel(), EVAL_BATCH):
        b = idx[start : start + EVAL_BATCH]
        zs, zt, mk = bank._fetch(b)
        pred = model(zs, bank.action_batch(b))

        w = 1.0 + MASK_WEIGHT * mk
        m_ptl = per_token_smooth_l1(pred, zt)
        i_ptl = per_token_smooth_l1(zs, zt)        # identity: pred = z_src
        acc["m_w_num"] += (w * m_ptl).sum().item(); acc["m_w_den"] += w.sum().item()
        acc["i_w_num"] += (w * i_ptl).sum().item(); acc["i_w_den"] += w.sum().item()

        ch = mk > 0.5
        acc["m_ch"] += m_ptl[ch].sum().item(); acc["i_ch"] += i_ptl[ch].sum().item()
        acc["n_ch"] += int(ch.sum())
        acc["m_un"] += m_ptl[~ch].sum().item(); acc["i_un"] += i_ptl[~ch].sum().item()
        acc["n_un"] += int((~ch).sum())

    nan = float("nan")
    return {
        "model": acc["m_w_num"] / max(acc["m_w_den"], 1e-8),
        "identity": acc["i_w_num"] / max(acc["i_w_den"], 1e-8),
        "model_changed": acc["m_ch"] / acc["n_ch"] if acc["n_ch"] else nan,
        "identity_changed": acc["i_ch"] / acc["n_ch"] if acc["n_ch"] else nan,
        "model_unchanged": acc["m_un"] / acc["n_un"] if acc["n_un"] else nan,
        "identity_unchanged": acc["i_un"] / acc["n_un"] if acc["n_un"] else nan,
    }


def _pairwise_sq_dists(a: Tensor, b: Tensor) -> Tensor:
    """||a-b||^2 via the mm expansion. torch.cdist on [512, 1.2M] vectors is a
    memory hazard; the expansion needs only the [512, 512] gram matrix."""
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True).t()
    return (a2 + b2 - 2.0 * (a @ b.t())).clamp_min(0)


@torch.no_grad()
def _predict_pool(model: Predictor, bank: EmbeddingBank, idx: Tensor):
    """Forward a whole index set in chunks; return flattened pred/src/tgt."""
    model.eval()
    preds, srcs, tgts = [], [], []
    for start in range(0, idx.numel(), EVAL_BATCH):
        b = idx[start : start + EVAL_BATCH]
        zs, zt, _ = bank._fetch(b)
        p = model(zs, bank.action_batch(b))
        preds.append(p.reshape(b.numel(), -1).half())
        srcs.append(zs.reshape(b.numel(), -1).half())
        tgts.append(zt.reshape(b.numel(), -1).half())
    return torch.cat(preds), torch.cat(srcs), torch.cat(tgts)


@torch.no_grad()
def retrieval(model: Predictor, bank: EmbeddingBank, val_idx: Tensor) -> dict:
    """Two retrieval metrics. The delta one is the headline.

    ABSOLUTE: rank candidate z_tgt by distance to pred. Reported with an identity
    baseline, which will be near-perfect — different samples are different
    screenshots, and pred ~= z_src ~= z_tgt. Useful only as a sanity floor.

    DELTA: rank candidate (z_tgt - z_src) by distance to (pred - z_src). This
    asks "did you predict the right *change*". Identity predicts a zero delta for
    every sample, so it cannot rank at all and sits at chance.
    """
    pool = val_idx[: min(RETRIEVAL_POOL, val_idx.numel())]
    v = pool.numel()
    pred, src, tgt = _predict_pool(model, bank, pool)
    correct = torch.arange(v, device=pred.device).unsqueeze(1)

    def top_k(d: Tensor) -> tuple[float, float]:
        ranks = d.argsort(dim=1)
        t1 = (ranks[:, :1] == correct).any(dim=1).float().mean().item()
        t5 = (ranks[:, : min(5, v)] == correct).any(dim=1).float().mean().item()
        return t1, t5

    a1, a5 = top_k(_pairwise_sq_dists(pred.float(), tgt.float()))
    ia1, ia5 = top_k(_pairwise_sq_dists(src.float(), tgt.float()))
    d1, d5 = top_k(_pairwise_sq_dists((pred - src).float(), (tgt - src).float()))
    return {"abs_top1": a1, "abs_top5": a5, "id_abs_top1": ia1, "id_abs_top5": ia5,
            "delta_top1": d1, "delta_top5": d5,
            "rand_top1": 1.0 / v, "rand_top5": min(5, v) / v, "n": v}


@torch.no_grad()
def action_sensitivity(model: Predictor, bank: EmbeddingBank, val_idx: Tensor) -> dict:
    """Is the model using the action, or ignoring it?

    Feed the same z_src with (a) its correct action and (b) another sample's
    action. Measured against the norm of the predicted delta itself, so the ~93%
    static background does not drown the comparison.
    """
    model.eval()
    pool = val_idx[: min(RETRIEVAL_POOL, val_idx.numel())]
    v = pool.numel()
    perm = torch.roll(torch.arange(v), shifts=1)   # derangement for v > 1
    swapped = pool[perm]

    corr, swap, srcs = [], [], []
    for start in range(0, v, EVAL_BATCH):
        b = pool[start : start + EVAL_BATCH]
        sb = swapped[start : start + EVAL_BATCH]
        zs, _, _ = bank._fetch(b)
        corr.append(model(zs, bank.action_batch(b)).reshape(b.numel(), -1))
        swap.append(model(zs, bank.action_batch(b, action_source=sb)).reshape(b.numel(), -1))
        srcs.append(zs.reshape(b.numel(), -1))
    corr, swap, srcs = torch.cat(corr), torch.cat(swap), torch.cat(srcs)

    swap_l2 = (corr - swap).norm(dim=1).mean().item()
    delta_norm = (corr - srcs).norm(dim=1).mean().item()
    return {"swap_l2": swap_l2, "delta_norm": delta_norm,
            "ratio": swap_l2 / max(delta_norm, 1e-8)}


# ── Schedule / build ─────────────────────────────────────────────────────────
def cosine_warmup(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 0. Returns an LR multiplier."""
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def build_model(bank: EmbeddingBank, mode: str) -> Predictor:
    action_encoder = ActionEncoder(
        mode=mode, d_model=bank.d_model, n_types=len(bank.type_vocab),
        n_elems=len(bank.elem_vocab), text_dim=bank.text_dim,
    )
    return Predictor(d_model=bank.d_model, n_tokens=bank.n_tokens,
                     n_layers=N_LAYERS, n_heads=N_HEADS,
                     action_encoder=action_encoder).to(DEVICE)


def save_ckpt(model: Predictor, bank: EmbeddingBank, mode: str, epoch: int,
              metrics: dict, tag: str) -> Path:
    """Checkpoint weights + everything needed to rebuild the model."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{bank.cache_dir.name}_{mode}_{tag}.pt"
    torch.save({
        "model": model.state_dict(), "epoch": epoch, "metrics": metrics,
        "mode": mode, "d_model": bank.d_model, "n_tokens": bank.n_tokens,
        "n_layers": N_LAYERS, "n_heads": N_HEADS, "mask_weight": MASK_WEIGHT,
        "type_vocab": bank.type_vocab, "elem_vocab": bank.elem_vocab,
        "text_dim": bank.text_dim, "encoder_cache": str(bank.cache_dir),
    }, path)
    return path


# ── Overfit-one sanity check ─────────────────────────────────────────────────
def overfit_one(bank: EmbeddingBank, mode: str) -> None:
    """Memorize a single pair. If it can't, there's a bug — fail fast and loud.

    Picks the pair with the MOST changed tokens. Memorizing a near-static pair is
    trivial for a residual model (delta = 0 already) and would prove nothing.
    """
    changed_frac = np.asarray(bank._mask[: min(2000, bank.m)]).mean(axis=1)
    pick = int(np.argmax(changed_frac))
    print(f"\n=== --overfit-one: memorizing pair {pick} "
          f"({changed_frac[pick]:.1%} tokens changed) for {OVERFIT_STEPS} steps ===")
    print(f"    action: {bank.action_text[pick][:90]}")
    print(f"    parsed: {parse_action(bank.action_text[pick])} "
          f"cont={[round(x, 3) for x in parse_continuous(bank.action_text[pick])]}")

    model = build_model(bank, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    idx = torch.tensor([pick])
    zs, zt, mk = bank._fetch(idx)
    act = bank.action_batch(idx)

    model.train()
    loss = torch.tensor(float("nan"))
    for step in range(OVERFIT_STEPS):
        pred = model(zs, act)
        loss = weighted_loss(pred, zt, mk)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if step % 50 == 0 or step == OVERFIT_STEPS - 1:
            print(f"  step {step:4d} | loss {loss.item():.6e}")
    assert loss.item() < OVERFIT_TOL, (
        f"overfit-one FAILED: loss {loss.item():.3e} >= tol {OVERFIT_TOL:.1e}. "
        f"The model cannot memorize a single pair — there is a bug."
    )
    print(f"overfit-one PASSED: final loss {loss.item():.3e} < {OVERFIT_TOL:.1e}")


# ── Train ────────────────────────────────────────────────────────────────────
def train(bank: EmbeddingBank, mode: str, epochs: int) -> None:
    torch.manual_seed(SEED)
    perm = torch.randperm(bank.m)
    n_val = max(1, int(round(VAL_FRAC * bank.m)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    # Per-epoch metrics use a fixed subset; full-split eval every epoch would
    # cost more than the training itself.
    tr_eval = train_idx[: min(EVAL_SUBSET, train_idx.numel())]
    va_eval = val_idx[: min(EVAL_SUBSET, val_idx.numel())]
    print(f"Split: {train_idx.numel()} train / {val_idx.numel()} val samples "
          f"(held-out samples, not tokens)")

    model = build_model(bank, mode)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Predictor: {n_params/1e6:.1f}M params | d_model={bank.d_model} "
          f"tokens={bank.n_tokens} layers={N_LAYERS} heads={N_HEADS}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(train_idx.numel() / BATCH_SIZE)
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, int(WARMUP_RATIO * total_steps), total_steps)
    )

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(train_idx.numel())]
        for start in range(0, order.numel(), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            zs, zt, mk = bank._fetch(idx)
            loss = weighted_loss(model(zs, bank.action_batch(idx)), zt, mk)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()

        tr = evaluate(model, bank, tr_eval)
        va = evaluate(model, bank, va_eval)
        print(f"epoch {epoch:3d} | lr {sched.get_last_lr()[0]:.2e} | "
              f"train m/id {tr['model']:.4f}/{tr['identity']:.4f} "
              f"(chg {tr['model_changed']:.4f}/{tr['identity_changed']:.4f}) | "
              f"val m/id {va['model']:.4f}/{va['identity']:.4f} "
              f"(chg {va['model_changed']:.4f}/{va['identity_changed']:.4f})")
        if not (va["model_changed"] < va["identity_changed"]):
            print("  !!  MODEL DOES NOT BEAT IDENTITY ON CHANGED TOKENS (val) — "
                  "it is learning nothing useful about the edit.")

        if va["model_changed"] < best:
            best = va["model_changed"]
            p = save_ckpt(model, bank, mode, epoch, va, "best")
            print(f"  new best val changed-token loss -> {p.name}")
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == epochs - 1:
            save_ckpt(model, bank, mode, epoch, va, f"ep{epoch:03d}")

    # ── Final headline evaluation ────────────────────────────────────────────
    print("\n===== FINAL EVALUATION (val) =====")
    ret = retrieval(model, bank, val_idx)
    print(f"pool size: {ret['n']}  (chance top-1 {ret['rand_top1']:.4f})")
    print(f"  absolute top-1: {ret['abs_top1']:.3f}   "
          f"[identity baseline {ret['id_abs_top1']:.3f}]  <- expect both ~1.0; "
          f"uninformative")
    print(f"  DELTA    top-1: {ret['delta_top1']:.3f}  top-5: {ret['delta_top5']:.3f}  "
          f"(identity == chance {ret['rand_top1']:.4f})   <- THE metric")
    sens = action_sensitivity(model, bank, val_idx)
    print(f"action sensitivity: swap L2 {sens['swap_l2']:.4f} vs predicted-delta "
          f"norm {sens['delta_norm']:.4f}  (ratio {sens['ratio']:.3f})")
    if sens["ratio"] < 0.1:
        print("  !!  Predictions barely change with the action — the model is "
              "likely ignoring it.")
    va = evaluate(model, bank, val_idx)
    print(f"beats identity on changed tokens (val): "
          f"{'YES' if va['model_changed'] < va['identity_changed'] else 'NO'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True,
                        help="cache DIRECTORY from precompute_embeddings.py")
    parser.add_argument("--action-encoding", choices=["structured", "text"],
                        default=ACTION_ENCODING)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--overfit-one", action="store_true",
                        help="memorize a single pair and assert near-zero loss")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    bank = EmbeddingBank(args.cache, args.action_encoding)
    print(f"Loaded cache: {bank.m} pairs | tokens={bank.n_tokens} "
          f"D={bank.d_model} | action-encoding={args.action_encoding}")

    if args.overfit_one:
        overfit_one(bank, args.action_encoding)
        return
    train(bank, args.action_encoding, args.epochs)


if __name__ == "__main__":
    main()