"""
Swappable frozen vision-encoder backends for the JEPA-style UI world model.

The world model never decodes pixels; it predicts the *embedding* of the next
screenshot. That embedding is produced by a frozen vision encoder. This module
hides the concrete encoder behind a tiny interface so the rest of the pipeline
(probe / precompute / train) is agnostic to which encoder we use.

Design contract
----------------
- ``Encoder.encode(imgs) -> [B, N_tokens, D]`` returns *patch* tokens (no pooled
  CLS/summary vector), because the world model reasons over the spatial token
  grid, not a single global vector.
- Every backend declares its own ``image_mean`` / ``image_std``. This matters:
  the dataset transform normalizes screenshots to ``[-1, 1]`` (``Normalize([0.5],
  [0.5])``), which is exactly SigLIP's convention but NOT ImageNet's. Feeding
  ``[-1, 1]`` tensors into an ImageNet-normalized encoder (e.g. DINOv2) silently
  corrupts the input. ``prepare_pixels`` therefore converts the dataset's
  ``[-1, 1]`` tensor back to ``[0, 1]`` and re-normalizes with the *backend's*
  stats explicitly, instead of assuming the dataset already matched.
- Adding a new backend is a one-class change: subclass nothing, just register a
  factory in ``ENCODER_REGISTRY``.

Hardware note (6x V100 / sm_70): no bfloat16 and no FlashAttention-2. Weights are
loaded in fp16 and inference runs under ``torch.inference_mode`` with fp16
autocast; HF encoders internally use ``scaled_dot_product_attention`` which maps
to a memory-efficient/math kernel on sm_70.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@runtime_checkable
class Encoder(Protocol):
    """Structural interface every backend must satisfy.

    Attributes are used by downstream scripts to size caches and build the
    change-mask token grid without knowing the concrete class.
    """

    name: str
    image_size: int          # square input resolution fed to the model
    patch_size: int          # pixels per patch edge
    grid: tuple[int, int]    # (rows, cols) of the token grid = image_size / patch_size
    d_model: int             # token embedding width
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    def encode(self, imgs: Tensor) -> Tensor:
        """Map a batch of images ``[B, 3, H, W]`` to tokens ``[B, N_tokens, D]``."""
        ...


@dataclass
class _EncoderSpec:
    """Static description of a backend, resolved before weights are loaded."""

    hf_name: str
    image_size: int
    patch_size: int
    d_model: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    drop_prefix_tokens: int  # e.g. DINOv2 emits a leading CLS token to strip


class _HFVisionEncoder(nn.Module):
    """Generic wrapper over a HuggingFace vision tower returning patch tokens.

    Both SigLIP-2 and DINOv2 expose ``vision_model(pixel_values).last_hidden_state``
    as ``[B, N, D]``. SigLIP-2 has no prepended global token, DINOv2 prepends a
    single CLS token; ``drop_prefix_tokens`` removes any such non-spatial tokens
    so the returned grid is purely patches in row-major order.
    """

    def __init__(self, key: str, spec: _EncoderSpec, device: str = "cuda") -> None:
        super().__init__()
        from transformers import AutoModel  # local import: heavy, only when needed

        self.name = key
        self.spec = spec
        self.image_size = spec.image_size
        self.patch_size = spec.patch_size
        self.d_model = spec.d_model
        self.image_mean = spec.image_mean
        self.image_std = spec.image_std
        side = spec.image_size // spec.patch_size
        self.grid = (side, side)
        self._device = device

        model = AutoModel.from_pretrained(spec.hf_name, torch_dtype=torch.float16)
        vision = getattr(model, "vision_model", model)
        self.vision = vision.eval()          # <- drop .to(device) here
        self.vision.requires_grad_(False)

        self.register_buffer(
            "_mean", torch.tensor(spec.image_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(spec.image_std).view(1, 3, 1, 1), persistent=False
        )

        self.to(device)

    @property
    def num_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    def prepare_pixels(self, imgs: Tensor) -> Tensor:
        """Convert dataset ``[-1, 1]`` tensors to this backend's normalization.

        The dataset uses ``Normalize([0.5], [0.5])`` -> ``[-1, 1]``. We undo that
        to recover ``[0, 1]`` and then apply the backend's own mean/std. This is
        deliberately explicit so that swapping to an ImageNet-normalized backend
        does not silently feed it SigLIP-scaled inputs.
        """
        imgs = imgs.to(self._device)
        if imgs.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"{self.name} expects {self.image_size}x{self.image_size}, got "
                f"{tuple(imgs.shape[-2:])}. Use letterbox_tensor in the transform."
            )
        x01 = imgs.mul(0.5).add(0.5).clamp_(0.0, 1.0)  # [-1,1] -> [0,1]
        return (x01 - self._mean) / self._std

    @torch.inference_mode()
    def encode(self, imgs: Tensor) -> Tensor:
        pixels = self.prepare_pixels(imgs).to(torch.float16)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.vision(pixel_values=pixels)
        tokens = out.last_hidden_state  # [B, N(+prefix), D]
        if self.spec.drop_prefix_tokens:
            tokens = tokens[:, self.spec.drop_prefix_tokens :, :]
        return tokens.float()


# ---------------------------------------------------------------------------
# Registry. Adding a backend = add one _EncoderSpec entry here.
# ---------------------------------------------------------------------------
_SPECS: dict[str, _EncoderSpec] = {
    # SigLIP-2 so400m at the highest fixed resolution (512 -> 32x32 tokens).
    # SigLIP normalization is [0.5]*3 / [0.5]*3, i.e. it natively expects [-1,1].
    "siglip2": _EncoderSpec(
        hf_name="google/siglip2-so400m-patch16-512",
        image_size=512,
        patch_size=16,
        d_model=1152,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        drop_prefix_tokens=0,
    ),
    # Second backend to prove the abstraction + exercise the normalization path:
    # DINOv2 expects ImageNet stats and prepends a CLS token.
    "dinov2": _EncoderSpec(
        hf_name="facebook/dinov2-large",
        image_size=518,
        patch_size=14,
        d_model=1024,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        drop_prefix_tokens=1,
    ),
}

ENCODER_REGISTRY: dict[str, Callable[[str], _HFVisionEncoder]] = {
    key: (lambda device, k=key: _HFVisionEncoder(k, _SPECS[k], device))
    for key in _SPECS
}


def build_encoder(name: str, device: str = "cuda") -> _HFVisionEncoder:
    """Instantiate a registered frozen encoder backend by short name."""
    if name not in ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown encoder '{name}'. Available: {sorted(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[name](device)


def encoder_spec(name: str) -> _EncoderSpec:
    """Return the static spec (resolution/grid/width) without loading weights."""
    if name not in _SPECS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {sorted(_SPECS)}")
    return _SPECS[name]

def letterbox_tensor(img: Tensor, size: int, fill: float = 1.0) -> Tensor:
    """Resize [C,H,W] to size x size preserving aspect ratio, padding with `fill`.

    A 1280x720 screenshot squashed to 512x512 is compressed 2.5x horizontally vs
    1.4x vertically — text becomes unreadable smears before the encoder sees it.
    Letterboxing keeps glyph shapes intact. fill=1.0 = white in [0,1] space
    (screenshots are white-background; black bars would create fake edges).
    """
    c, h, w = img.shape
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    img = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bicubic",
                        align_corners=False, antialias=True).squeeze(0)
    out = img.new_full((c, size, size), fill)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[:, top:top + nh, left:left + nw] = img
    return out