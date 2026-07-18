import json
import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


_RESULT_DIR_PREFIXES = ("results_cua_screenshot", "results_gemini_pro_25_screenshot")


def _swap_annotated(filename: str) -> str:
    return filename.replace("_not_annotated_with_cursor.png", "_annotated_with_cursor.png")


class WorldModelDataset(Dataset):
    """
    PyTorch Dataset for UI world-model training.

    Each sample contains:
        step_number      (int)   – bookkeeping step index within the episode
        action_full      (str)   – raw action string for bookkeeping
        action_text      (str)   – actions_resolved joined as text; used as diffusion conditioning
        input_image      (PIL.Image)  – screenshot_before (optionally annotated)
        output_image     (PIL.Image)  – screenshot_after / ground truth (optionally annotated)
        source           (str)   – relative path to the episode folder for traceability
    """

    def __init__(
        self,
        data_root: Union[str, os.PathLike] = "./train_data",
        use_annotated_image: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.use_annotated_image = use_annotated_image
        self.transform = transform
        self.target_transform = target_transform

        self.samples: list[dict] = []
        self._build_index()

    def _build_index(self) -> None:
        for result_dir in sorted(self.data_root.iterdir()):
            if not result_dir.is_dir():
                continue
            if not any(result_dir.name.startswith(p) for p in _RESULT_DIR_PREFIXES):
                continue

            for episode_dir in sorted(result_dir.iterdir(), key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name)):
                triplets_path = episode_dir / "actions_triplets.json"
                if not triplets_path.exists():
                    continue

                with open(triplets_path) as f:
                    triplets = json.load(f)

                for entry in triplets:
                    before = entry.get("screenshot_before", "")
                    after = entry.get("screenshot_after", "")

                    if self.use_annotated_image:
                        before = _swap_annotated(before)
                        after = _swap_annotated(after)

                    before_path = episode_dir / before
                    after_path = episode_dir / after

                    if not before_path.exists() or not after_path.exists():
                        continue

                    if self.use_annotated_image:
                        actions_resolved = entry.get("actions_resolved", [])
                        action_text = "; ".join(actions_resolved) if isinstance(actions_resolved, list) else str(actions_resolved)
                    else:
                        action_text = entry.get("action_full", "")

                    self.samples.append({
                        "step_number": entry.get("step_number"),
                        "action_full": entry.get("action_full", ""),
                        "action_text": action_text,
                        "input_path": before_path,
                        "output_path": after_path,
                        "source": str(episode_dir.relative_to(self.data_root)),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        input_image = Image.open(sample["input_path"]).convert("RGB")
        output_image = Image.open(sample["output_path"]).convert("RGB")

        if self.transform is not None:
            input_image = self.transform(input_image)
        if self.target_transform is not None:
            output_image = self.target_transform(output_image)

        return {
            "step_number": sample["step_number"],
            "action_full": sample["action_full"],
            "action_text": sample["action_text"],
            "input_image": input_image,
            "output_image": output_image,
            "source": sample["source"],
        }
