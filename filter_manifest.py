import json, random
from pathlib import Path

root = Path(r'C:\Users\adyes\Downloads\results_gemini_pro_25_screenshot\results_gemini_pro_25_screenshot')
triplets = []

for ep in sorted(root.iterdir()):
    j = ep / 'actions_triplets_dino_changes.json'
    if not ep.is_dir() or not j.is_file():
        continue
    rows = json.load(open(j, encoding='utf-8'))
    for row in rows:
        b = ep / row.get('screenshot_before', '')
        a = ep / row.get('screenshot_after', '')
        act = row.get('action_full') or row.get('action_raw', '')
        rmse = row.get('global_rmse_area_value')
        if b.is_file() and a.is_file() and act and rmse is not None:
            triplets.append({'before': b, 'after': a, 'action': act, 'ep': ep.name, 'rmse_area': float(rmse)})

triplets.sort(key=lambda x: x['rmse_area'], reverse=True)
top10 = triplets[:max(1, int(len(triplets)*0.10))]

N = 128
random.seed(42)
samples = random.sample(top10, min(N, len(top10)))

json.dump(
    [{'before': str(s['before']), 'after': str(s['after']), 'action': s['action'], 'ep': s['ep'], 'rmse_area': s['rmse_area']} for s in samples],
    open('manifest_filtered.json', 'w'), indent=2
)
print(f'Saved {len(samples)} filtered triplets to manifest_filtered.json')
print(f'Total images to upload: {len(samples)*2}')