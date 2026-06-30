import json, random, subprocess
from pathlib import Path

root = Path(r'C:\Users\adyes\Downloads\results_gemini_pro_25_screenshot\results_gemini_pro_25_screenshot')
triplets = []
for ep in sorted(root.iterdir()):
    j = ep / 'actions_triplets.json'
    if not ep.is_dir() or not j.is_file():
        continue
    rows = json.load(open(j, encoding='utf-8'))
    for row in rows:
        b = ep / row.get('screenshot_before', '')
        a = ep / row.get('screenshot_after', '')
        act = row.get('action_full') or row.get('action_raw', '')
        if b.is_file() and a.is_file() and act:
            triplets.append({'before': b, 'after': a, 'action': act, 'ep': ep.name})

random.seed(42)
samples = random.sample(triplets, min(20, len(triplets)))

# Save manifest for later
import json as j2
j2.dump([{'before': str(s['before']), 'after': str(s['after']), 'action': s['action'], 'ep': s['ep']} for s in samples], open('manifest.json', 'w'), indent=2)
print(f'Selected {len(samples)} triplets, saved to manifest.json')
for s in samples:
    print(s['before'])
    print(s['after'])
