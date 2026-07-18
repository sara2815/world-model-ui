import json, zipfile, os

manifest = json.load(open("manifest_filtered.json"))
files = set()
for item in manifest:
    files.add(item["before"])
    files.add(item["after"])

with zipfile.ZipFile("filtered_images.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, arcname=os.path.basename(f))

print("Zipped", len(files), "files into filtered_images.zip")
