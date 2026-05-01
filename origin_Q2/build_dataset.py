import json
import shutil
from pathlib import Path

BASE = Path("/mnt/nfs/zyxing/VLMEvalKit/origin_Q2")
OUT_IMAGES = BASE / "images"
OUT_JSONL = BASE / "qa_dataset_merged_5000.jsonl"

OUT_IMAGES.mkdir(exist_ok=True)

FILE_1657 = BASE / "qa_dataset_filtered_1657.jsonl"
FILE_6165 = BASE / "qa_dataset_filtered_6165.jsonl"

# Load all entries
with open(FILE_1657) as f:
    entries_1657 = [json.loads(line) for line in f if line.strip()]

with open(FILE_6165) as f:
    entries_6165 = [json.loads(line) for line in f if line.strip()][:3363]

all_entries = entries_1657 + entries_6165
print(f"Total entries: {len(all_entries)} ({len(entries_1657)} from 1657 + {len(entries_6165)} from 6165)")

FALLBACK_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]

def resolve_image(base, rel_path):
    src = base / rel_path
    if src.exists():
        return src
    # jsonl extension may be wrong; try other common extensions
    # Avoid Path.with_suffix() — it treats intermediate dots as suffixes on filenames like "foo.hash.jpg"
    name_no_ext = src.name[:-len(src.suffix)]  # strip only the trailing extension
    for ext in FALLBACK_EXTS:
        candidate = src.parent / (name_no_ext + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found (tried all extensions): {src}")

out_records = []
for idx, entry in enumerate(all_entries, start=1):
    rel_path = entry["image"]
    src = resolve_image(BASE, rel_path)

    ext = src.suffix  # preserve original extension
    new_name = f"{idx:04d}{ext}"
    dst = OUT_IMAGES / new_name

    shutil.copy2(src, dst)

    new_entry = dict(entry)
    new_entry["image"] = f"images/{new_name}"
    out_records.append(new_entry)

    if idx % 500 == 0:
        print(f"  Processed {idx}/{len(all_entries)}")

with open(OUT_JSONL, "w") as f:
    for record in out_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Done. Output jsonl: {OUT_JSONL}")
print(f"Images copied to: {OUT_IMAGES}")
