"""
PyTorch dataset for DashboardQA screenshot transitions.

Each sample is one row from ``actions_triplets.json``: predict ``screenshot_after`` from
``screenshot_before`` conditioned on the action text (diffusion target = next screen, condition =
current screen + action).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


ActionField = Literal["action_full", "action_raw", "actions_resolved"]
VisualChangeFilter = Literal["any", "token", "metric", "both"]

DEFAULT_DINO_CHANGES_FILENAME = "actions_triplets_dino_changes.json"


def _episode_sort_key(p: Path) -> Tuple[int, str]:
    name = p.name
    return (int(name), name) if name.isdigit() else (10**9, name)


def _format_action(entry: dict[str, Any], field: ActionField) -> str:
    if field == "actions_resolved":
        lines = entry.get("actions_resolved") or []
        if isinstance(lines, list):
            return "\n".join(str(x) for x in lines)
        return str(lines)
    return str(entry.get(field, ""))


def _visual_change_flags(dino_row: dict[str, Any]) -> Tuple[bool, bool]:
    return bool(dino_row.get("has_visual_change_token")), bool(dino_row.get("has_visual_change_metric"))


def _passes_visual_change_filter(dino_row: dict[str, Any], mode: VisualChangeFilter) -> bool:
    tok, met = _visual_change_flags(dino_row)
    if mode == "any":
        return tok or met
    if mode == "token":
        return tok
    if mode == "metric":
        return met
    if mode == "both":
        return tok and met
    raise ValueError(f"unknown visual_change filter mode: {mode!r}")


def _tensorize_pil(
    img: Image.Image,
    tfm: Callable[[Image.Image], torch.Tensor],
    spatial: T.Compose,
) -> torch.Tensor:
    img = img.convert("RGB")
    return tfm(spatial(img))


@dataclass(frozen=True)
class TransitionIndex:
    episode_dir: Path
    """Directory containing ``actions_triplets.json`` and PNG files."""
    json_index: int
    """Index of this step inside the episode's ``actions_triplets.json`` list."""
    step_number: int
    """``step_number`` from the JSON row (for logging)."""


class DashboardQAScreenshotTransitionDataset(Dataset):
    """
    Loads (before image, action text, after image) triplets from DashboardQA CUA screenshot runs.

    Parameters
    ----------
    root
        Path to ``results_cua_screenshot`` (parent of numbered episode folders).
    action_field
        Which action string to expose as conditioning text.
    image_size
        If set, ``Resize`` (bilinear) to ``(H, W)``. Images are already a fixed size in this dump
        (e.g. 1080×1920); resize when your model needs a different resolution.
    value_range
        Pixel range for returned tensors: ``[0, 1]`` or ``[-1, 1]`` (common for diffusion).
    split
        ``None`` (all episodes), ``\"train\"``, or ``\"val\"``. Uses a deterministic per-episode split.
    val_fraction
        Fraction of episodes reserved for validation when ``split`` is set.
    split_seed
        RNG seed for the train/val episode partition.
    skip_missing_files
        If True, drops triplets whose PNG paths are missing. If False, raises when a file is missing.
    episodes
        Optional explicit list of episode directory paths or names under ``root`` to restrict loading.
    visual_change_filter
        If set, only keep triplets whose row in ``dino_changes_filename`` passes the rule:
        ``any`` — token OR metric true; ``token`` / ``metric`` — that flag true; ``both`` — both true.
    dino_changes_filename
        Per-episode JSON (e.g. ``actions_triplets_dino_changes.json``) aligned with
        ``actions_triplets.json`` (same length and matching ``screenshot_before`` / ``screenshot_after``).
    include_visual_change_flags
        If True, add ``has_visual_change_token`` and ``has_visual_change_metric`` to each sample
        when the DINO JSON was loaded and aligned for that episode.
    attach_visual_change_flags
        If True, load ``dino_changes_filename`` when present (and aligned) and attach flags to samples
        even when ``visual_change_filter`` is None (no triplet filtering).
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        action_field: ActionField = "action_full",
        image_size: Optional[Tuple[int, int]] = None,
        value_range: Literal["0_1", "neg1_1"] = "neg1_1",
        split: Optional[Literal["train", "val"]] = None,
        val_fraction: float = 0.05,
        split_seed: int = 0,
        skip_missing_files: bool = True,
        episodes: Optional[Sequence[Union[str, Path]]] = None,
        visual_change_filter: Optional[VisualChangeFilter] = None,
        dino_changes_filename: str = DEFAULT_DINO_CHANGES_FILENAME,
        include_visual_change_flags: bool = True,
        attach_visual_change_flags: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.action_field = action_field
        self.value_range = value_range
        self.skip_missing_files = skip_missing_files
        self.visual_change_filter = visual_change_filter
        self.dino_changes_filename = dino_changes_filename
        self._include_visual_flags_in_sample = include_visual_change_flags

        self.build_stats: dict[str, Any] = {
            "episodes_seen": 0,
            "episodes_used": 0,
            "episodes_skipped_no_dino_file": 0,
            "episodes_skipped_dino_length_mismatch": 0,
            "episodes_skipped_dino_row_mismatch": 0,
            "triplets_candidates": 0,
            "triplets_skipped_missing_images": 0,
            "triplets_skipped_visual_change_filter": 0,
            "triplets_kept": 0,
        }

        all_episodes = sorted(
            [p for p in self.root.iterdir() if p.is_dir() and (p / "actions_triplets.json").is_file()],
            key=_episode_sort_key,
        )
        if episodes is not None:
            chosen = {Path(e).name for e in episodes}
            all_episodes = [p for p in all_episodes if p.name in chosen]

        if split is not None:
            rng = random.Random(split_seed)
            shuffled = all_episodes[:]
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_fraction))) if shuffled else 0
            val_set = set(p.name for p in shuffled[:n_val])
            if split == "train":
                all_episodes = [p for p in all_episodes if p.name not in val_set]
            else:
                all_episodes = [p for p in all_episodes if p.name in val_set]

        self._indices: list[TransitionIndex] = []
        self._rows: list[dict[str, Any]] = []
        self._dino_rows: list[Optional[dict[str, Any]]] = []

        need_dino = visual_change_filter is not None or attach_visual_change_flags

        for ep in all_episodes:
            self.build_stats["episodes_seen"] += 1
            json_path = ep / "actions_triplets.json"
            dino_path = ep / dino_changes_filename

            with open(json_path, encoding="utf-8") as f:
                triplets: list[dict[str, Any]] = json.load(f)

            dino_triplets: Optional[list[dict[str, Any]]] = None
            if need_dino:
                if not dino_path.is_file():
                    self.build_stats["episodes_skipped_no_dino_file"] += 1
                    if visual_change_filter is not None:
                        if skip_missing_files:
                            continue
                        raise FileNotFoundError(
                            f"{dino_path} is required when visual_change_filter is set "
                            f"(episode {ep.name})."
                        )
                else:
                    with open(dino_path, encoding="utf-8") as df:
                        loaded: list[dict[str, Any]] = json.load(df)
                    if len(loaded) != len(triplets):
                        self.build_stats["episodes_skipped_dino_length_mismatch"] += 1
                        if visual_change_filter is not None:
                            if skip_missing_files:
                                continue
                            raise ValueError(
                                f"{dino_path} length {len(loaded)} != "
                                f"{json_path} length {len(triplets)} (episode {ep.name})"
                            )
                    else:
                        misaligned = any(
                            row.get("screenshot_before") != drow.get("screenshot_before")
                            or row.get("screenshot_after") != drow.get("screenshot_after")
                            for row, drow in zip(triplets, loaded)
                        )
                        if misaligned:
                            self.build_stats["episodes_skipped_dino_row_mismatch"] += 1
                            if visual_change_filter is not None:
                                if skip_missing_files:
                                    continue
                                raise ValueError(
                                    f"DINO rows misaligned with {json_path} (episode {ep.name})"
                                )
                            loaded = []
                        dino_triplets = loaded if loaded else None

            self.build_stats["episodes_used"] += 1

            for j, row in enumerate(triplets):
                self.build_stats["triplets_candidates"] += 1
                drow: Optional[dict[str, Any]] = None
                if dino_triplets is not None:
                    drow = dino_triplets[j]

                if visual_change_filter is not None:
                    if drow is None:
                        raise RuntimeError(
                            f"Internal error: missing DINO row for filter at {json_path} index {j}"
                        )
                    if not _passes_visual_change_filter(drow, visual_change_filter):
                        self.build_stats["triplets_skipped_visual_change_filter"] += 1
                        continue

                before_name = row.get("screenshot_before")
                after_name = row.get("screenshot_after")
                if not before_name or not after_name:
                    if skip_missing_files:
                        continue
                    raise FileNotFoundError(f"Missing screenshot keys in {json_path} index {j}")
                before_path = ep / before_name
                after_path = ep / after_name
                if not before_path.is_file() or not after_path.is_file():
                    self.build_stats["triplets_skipped_missing_images"] += 1
                    if skip_missing_files:
                        continue
                    raise FileNotFoundError(f"Missing image: {before_path} or {after_path}")
                step_no = int(row.get("step_number", j))
                self._indices.append(TransitionIndex(ep, j, step_no))
                self._rows.append(row)
                self._dino_rows.append(drow)
                self.build_stats["triplets_kept"] += 1

        spatial: list[Any] = []
        if image_size is not None:
            h, w = image_size
            spatial.append(T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR))
        self._spatial = T.Compose(spatial) if spatial else (lambda x: x)

        to_tensor = T.ToTensor()
        if value_range == "neg1_1":
            self._finalize = lambda t: t * 2.0 - 1.0
        else:
            self._finalize = lambda t: t

        self._to_tensor = to_tensor

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self._indices[idx]
        row = self._rows[idx]
        action_text = _format_action(row, self.action_field)

        before_path = meta.episode_dir / row["screenshot_before"]
        after_path = meta.episode_dir / row["screenshot_after"]

        with Image.open(before_path) as im_b, Image.open(after_path) as im_a:
            source = _tensorize_pil(im_b, self._to_tensor, self._spatial)
            target = _tensorize_pil(im_a, self._to_tensor, self._spatial)

        source = self._finalize(source)
        target = self._finalize(target)

        out: dict[str, Any] = {
            "source": source,
            "target": target,
            "action_text": action_text,
            "episode": meta.episode_dir.name,
            "step_number": meta.step_number,
            "json_index": meta.json_index,
            "screenshot_before": row["screenshot_before"],
            "screenshot_after": row["screenshot_after"],
        }
        if self._include_visual_flags_in_sample:
            dr = self._dino_rows[idx]
            if dr is not None:
                t, m = _visual_change_flags(dr)
                out["has_visual_change_token"] = t
                out["has_visual_change_metric"] = m
        return out

    def iter_episode_batches(self) -> Iterator[list[int]]:
        """Yield lists of dataset indices, grouped by episode (order preserved within episode)."""
        by_ep: dict[str, list[int]] = {}
        for i, meta in enumerate(self._indices):
            by_ep.setdefault(meta.episode_dir.name, []).append(i)
        for name in sorted(by_ep.keys(), key=lambda n: int(n) if n.isdigit() else n):
            yield by_ep[name]


def make_dataloader(
    root: Union[str, Path],
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Convenience ``DataLoader`` with default collate (string actions stay as a list of str)."""
    ds = DashboardQAScreenshotTransitionDataset(root, **dataset_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Inspect DashboardQA screenshot transition dataset: counts, optional DINO visual-change "
            "filtering, and one sample tensor/shape."
        )
    )
    p.add_argument(
        "root",
        nargs="?",
        default="",
        type=Path,
        help="Dataset root (directory that contains numbered episode folders).",
    )
    p.add_argument(
        "--action-field",
        choices=("action_full", "action_raw", "actions_resolved"),
        default="action_full",
        help="Which text field is returned as action_text.",
    )
    p.add_argument(
        "--value-range",
        choices=("neg1_1", "0_1"),
        default="neg1_1",
        help="Tensor range for source/target.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Resize both screenshots to this height and width (bilinear).",
    )
    p.add_argument(
        "--split",
        choices=("all", "train", "val"),
        default="all",
        help="Episode-level train/val partition (deterministic; see --val-fraction and --split-seed).",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of episodes held out for --split val (at least one episode if possible).",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for the train/val episode shuffle.",
    )
    p.add_argument(
        "--visual-change-filter",
        choices=("any", "token", "metric", "both"),
        default="both",
        help=(
            "Require DINO-derived visual change flags: any=token OR metric; both=token AND metric. "
            "Requires per-episode actions_triplets_dino_changes.json (see --dino-changes-filename)."
        ),
    )
    p.add_argument(
        "--dino-changes-filename",
        default=DEFAULT_DINO_CHANGES_FILENAME,
        help="Filename inside each episode directory with has_visual_change_* fields.",
    )
    p.add_argument(
        "--attach-visual-change-flags",
        action="store_true",
        help=(
            "Load DINO JSON when present and attach has_visual_change_token/metric to samples "
            "without filtering triplets."
        ),
    )
    p.add_argument(
        "--no-include-visual-change-flags",
        action="store_true",
        help="Do not add has_visual_change_* keys to sample dicts (still applies filter if set).",
    )
    p.add_argument(
        "--strict-files",
        action="store_true",
        help="Raise if images or required DINO files are missing instead of skipping.",
    )
    args = p.parse_args()

    split_kw: dict[str, Any] = {}
    if args.split != "all":
        split_kw["split"] = args.split
        split_kw["val_fraction"] = args.val_fraction
        split_kw["split_seed"] = args.split_seed

    size = tuple(args.image_size) if args.image_size else None
    ds = DashboardQAScreenshotTransitionDataset(
        args.root,
        image_size=size,
        action_field=args.action_field,
        value_range=args.value_range,
        skip_missing_files=not args.strict_files,
        visual_change_filter=args.visual_change_filter,
        dino_changes_filename=args.dino_changes_filename,
        attach_visual_change_flags=args.attach_visual_change_flags,
        include_visual_change_flags=not args.no_include_visual_change_flags,
        **split_kw,
    )
    print(f"root={args.root}")
    print(f"len(dataset)={len(ds)}")
    for k, v in sorted(ds.build_stats.items()):
        print(f"  {k}: {v}")
    if len(ds):
        s = ds[0]
        print("sample keys:", sorted(s.keys()))
        print(
            "source:",
            tuple(s["source"].shape),
            s["source"].dtype,
            float(s["source"].min()),
            float(s["source"].max()),
        )
        print("target:", tuple(s["target"].shape))
        print("action_text[:200]:", s["action_text"][:200].replace("\n", " / "))
