import json

manifest = json.load(open("manifest_filtered.json"))
files = []
for item in manifest:
    files.append(item["before"])
    files.append(item["after"])

lines = []
lines.append('scp -i "C:\\Users\\adyes\\OneDrive\\Desktop\\SSH\\id_ed25519" `')
for f in files:
    lines.append('"' + f + '" `')
lines.append('adyesha7@fir.alliancecan.ca:~/dataset/271/')

with open("upload_command.ps1", "w") as out:
    out.write("\n".join(lines))

print("Saved to upload_command.ps1, total files:", len(files))
