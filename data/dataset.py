from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset


_RESULT_DIRS = ("results_cua_screenshot", "results_gemini_pro_25_screenshot")
_DEFAULT_DATA_ROOT = ""

_TRIPLETS_FILENAME = "actions_triplets_dino_changes.json"
_TRIPLETS_FALLBACK = "actions_triplets.json"

_DEFAULT_MIN_CHANGE_METRIC = 10000.0
_DEFAULT_MAX_CHANGE_METRIC = 126500.0

_SLEEP_RE = re.compile(r"^\s*sleep\s+[\d.]+\s*s\s*$", re.IGNORECASE)
_ACTION_JOINER = " and then "

# Whole-triplet drop if any action matches (ctrl+a as Control+a / ctrl+a).
_DEFAULT_DROP_ACTION_PATTERNS = (r"(?:control|ctrl)\s*\+\s*a\b",)


def _swap_annotated(filename: str) -> str:
    return filename.replace("_not_annotated_with_cursor.png", "_annotated_with_cursor.png")


def is_sleep(action: str) -> bool:
    return isinstance(action, str) and bool(_SLEEP_RE.match(action))


def compile_action_patterns(patterns: Optional[Sequence[str]]) -> list:
    if not patterns:
        return []
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def as_action_list(actions_resolved) -> list:
    if isinstance(actions_resolved, list):
        return list(actions_resolved)
    if actions_resolved:
        return [str(actions_resolved)]
    return []


def entry_has_blocked_action(actions_resolved, action_full, compiled_patterns) -> bool:
    if not compiled_patterns:
        return False
    items = [a for a in as_action_list(actions_resolved) if isinstance(a, str)]
    items.extend(s for s in (action_full or "").split(_ACTION_JOINER) if s)
    return any(rx.search(a) for a in items for rx in compiled_patterns)


def strip_sleep_from_list(actions: list) -> list:
    return [a for a in actions if not is_sleep(a)]


def strip_sleep_from_full(action_full: str) -> str:
    return _ACTION_JOINER.join(
        s for s in action_full.split(_ACTION_JOINER) if not is_sleep(s)
    )


def clean_step_actions(actions_resolved_raw, action_full_raw, *, strip_sleep: bool, use_annotated: bool):
    """Return (cleaned_list, action_full, action_text, had_sleep)."""
    resolved = as_action_list(actions_resolved_raw)
    full_raw = action_full_raw or ""
    had_sleep = any(is_sleep(a) for a in resolved) or any(
        is_sleep(s) for s in full_raw.split(_ACTION_JOINER)
    )
    if strip_sleep:
        resolved = strip_sleep_from_list(resolved)
        full = strip_sleep_from_full(full_raw)
    else:
        full = full_raw
    if use_annotated:
        return resolved, full, "; ".join(resolved), had_sleep
    segments = [s for s in full.split(_ACTION_JOINER) if s]
    return segments, full, full, had_sleep


def metric_bucket(value, min_m, max_m):
    """Return 'ok', 'low', 'high', or 'missing' given optional thresholds."""
    if min_m is None and max_m is None:
        return "ok"
    if not isinstance(value, (int, float)):
        return "missing"
    if min_m is not None and value < min_m:
        return "low"
    if max_m is not None and value > max_m:
        return "high"
    return "ok"


def classify_action(action: str) -> str:
    t = (action or "").strip().lower()
    if not t:
        return "other"
    if is_sleep(action):
        return "sleep"
    if t.startswith("type ") or t.startswith("type'") or "type '" in t:
        return "type"
    if t.startswith("hotkey") or t.startswith("press ") or "keypress" in t:
        return "key"
    if "double" in t and "click" in t:
        return "double_click"
    if "drag" in t:
        return "drag"
    if "scroll" in t:
        return "scroll"
    if t.startswith(("move to", "move the cursor", "move ")):
        return "move"
    if t.startswith("click") or "click on" in t:
        return "click"
    return "other"


def extract_action_kinds(actions) -> list:
    """Ordered unique kinds in a step (list or 'A and then B' string). Sleep omitted."""
    items = actions.split(_ACTION_JOINER) if isinstance(actions, str) else (actions or [])
    kinds = []
    for a in items:
        k = classify_action(a)
        if k != "sleep" and k not in kinds:
            kinds.append(k)
    return kinds


class WorldModelDataset(Dataset):
    """UI world-model dataset: (before, after, action) triplets with optional cleaning.

    Cleaning knobs (each independently disableable; or pass disable_filters=True):
      - min/max_change_metric: keep only if metric is in range
      - drop_action_patterns: drop whole triplet if any action matches (default ctrl+a)
      - strip_sleep: remove sleep actions; drop sleep-only steps
    """

    def __init__(
        self,
        data_root: Union[str, os.PathLike] = _DEFAULT_DATA_ROOT,
        use_annotated_image: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        min_change_metric: Optional[float] = _DEFAULT_MIN_CHANGE_METRIC,
        max_change_metric: Optional[float] = _DEFAULT_MAX_CHANGE_METRIC,
        strip_sleep: bool = True,
        drop_action_patterns: Optional[Sequence[str]] = _DEFAULT_DROP_ACTION_PATTERNS,
        disable_filters: bool = False,
    ):
        self.data_root = Path(data_root)
        self.use_annotated_image = use_annotated_image
        self.transform = transform
        self.target_transform = target_transform
        self.min_change_metric = None if disable_filters else min_change_metric
        self.max_change_metric = None if disable_filters else max_change_metric
        self.strip_sleep = False if disable_filters else strip_sleep
        self.drop_action_patterns = (
            () if disable_filters else tuple(drop_action_patterns or ())
        )
        self._blocked = compile_action_patterns(self.drop_action_patterns)
        self.samples: list = []
        self._build_index()

    def _build_index(self) -> None:
        root = self.data_root
        for result_dir in sorted(root.iterdir()):
            if not result_dir.is_dir() or result_dir.name not in _RESULT_DIRS:
                continue
            episodes = sorted(
                result_dir.iterdir(),
                key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name),
            )
            for episode_dir in episodes:
                path = episode_dir / _TRIPLETS_FILENAME
                if not path.exists():
                    path = episode_dir / _TRIPLETS_FALLBACK
                if not path.exists():
                    continue
                with open(path) as f:
                    triplets = json.load(f)

                source = str(episode_dir.relative_to(root))
                for entry in triplets:
                    if metric_bucket(
                        entry.get("change_metric_value"),
                        self.min_change_metric,
                        self.max_change_metric,
                    ) != "ok":
                        continue

                    before = entry.get("screenshot_before", "")
                    after = entry.get("screenshot_after", "")
                    if self.use_annotated_image:
                        before, after = _swap_annotated(before), _swap_annotated(after)
                    before_path, after_path = episode_dir / before, episode_dir / after
                    if not before_path.exists() or not after_path.exists():
                        continue

                    ar_raw = entry.get("actions_resolved", [])
                    af_raw = entry.get("action_full", "")
                    if entry_has_blocked_action(ar_raw, af_raw, self._blocked):
                        continue

                    cleaned, action_full, action_text, _ = clean_step_actions(
                        ar_raw, af_raw,
                        strip_sleep=self.strip_sleep,
                        use_annotated=self.use_annotated_image,
                    )
                    if not cleaned:
                        continue

                    self.samples.append({
                        "step_number": entry.get("step_number"),
                        "action_full": action_full,
                        "action_text": action_text,
                        "action_kinds": extract_action_kinds(cleaned),
                        "change_metric_value": entry.get("change_metric_value"),
                        "input_path": before_path,
                        "output_path": after_path,
                        "source": source,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        inp = Image.open(s["input_path"]).convert("RGB")
        out = Image.open(s["output_path"]).convert("RGB")
        if self.transform is not None:
            inp = self.transform(inp)
        if self.target_transform is not None:
            out = self.target_transform(out)
        return {
            "step_number": s["step_number"],
            "action_full": s["action_full"],
            "action_text": s["action_text"],
            "action_kinds": s["action_kinds"],
            "change_metric_value": s["change_metric_value"],
            "input_image": inp,
            "output_image": out,
            "input_path": s["input_path"],
            "output_path": s["output_path"],
            "source": s["source"],
        }
