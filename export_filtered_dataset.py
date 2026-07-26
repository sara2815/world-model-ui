from pathlib import Path
import json
import shutil

from dataset_filter import WorldModelDataset


# -----------------------------
# Paths
# -----------------------------
src_root = Path(r"C:\Users\adyes\Downloads")
out_root = Path(r"C:\Users\adyes\Downloads\filtered_world_model")


# -----------------------------
# Load filtered dataset
# -----------------------------
dataset = WorldModelDataset(
    data_root=src_root
)

print("Filtered dataset size:", len(dataset))


# -----------------------------
# Group by episode
# -----------------------------
episodes = {}

for sample in dataset.samples:
    source = sample["source"]

    if source not in episodes:
        episodes[source] = []

    episodes[source].append(sample)


print("Episodes found:", len(episodes))


# -----------------------------
# Export images + JSON
# -----------------------------
for source, samples in episodes.items():

    out_episode = out_root / source
    out_episode.mkdir(parents=True, exist_ok=True)
    
        # Load original actions_triplets_new.json
    original_episode = src_root / source
    action_file = original_episode / "actions_triplets_new.json"

    if not action_file.exists():
        print("Missing:", action_file)
        continue

    with open(action_file, "r", encoding="utf-8") as f:
        original_triplets = json.load(f)

    filtered_entries = []

    for sample in samples:

        # Copy input screenshot
        shutil.copy(
            sample["input_path"],
            out_episode / sample["input_path"].name
        )

        # Copy output screenshot
        shutil.copy(
            sample["output_path"],
            out_episode / sample["output_path"].name
        )

        # Save matching metadata
        # Find matching original triplet and keep all fields
        for triplet in original_triplets:
            if triplet.get("step_number") == sample["step_number"]:

                # Add action_kinds from dataset_filter.py
                triplet["action_kinds"] = sample["action_kinds"]

                filtered_entries.append(triplet)
                break
        

    # Write filtered triplets JSON
    with open(
        out_episode / "actions_triplets_filtered.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            filtered_entries,
            f,
            indent=2,
            ensure_ascii=False
        )


print("Finished exporting.")
print("Saved to:", out_root)