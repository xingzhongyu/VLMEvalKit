import json
import csv
import os

INPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q5/q5_hard_v8_1075.jsonl'
OUTPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q5/benchmark_q5.tsv'
# Images live under Q2; use absolute paths so dump_image resolves them directly.
IMAGE_BASE = '/mnt/nfs/zyxing/VLMEvalKit/Q2'

LETTERS = list('ABCDEF')
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff']


def resolve_image(abs_path):
    """Return the resolved absolute path for an image.

    If the exact path exists, return it as-is.
    Otherwise try swapping the extension with each entry in IMAGE_EXTS.
    Returns (resolved_path, status) where status is 'ok', 'ext_fixed', or 'missing'.
    """
    if os.path.exists(abs_path):
        return abs_path, 'ok'

    base, _ = os.path.splitext(abs_path)
    for ext in IMAGE_EXTS:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate, 'ext_fixed'

    return abs_path, 'missing'


with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f]

rows = []
missing_images = []
ext_fixed_images = []

for idx, r in enumerate(records):
    abs_image = os.path.join(IMAGE_BASE, r['image'])
    resolved, status = resolve_image(abs_image)

    if status == 'missing':
        missing_images.append((idx, abs_image))
    elif status == 'ext_fixed':
        ext_fixed_images.append((idx, abs_image, resolved))

    row = {
        'index': idx,
        'question': r['question'],
        'image_path': resolved,
        'answer': r['gt_letter'],
        'category': r.get('company', ''),
    }
    for c in LETTERS:
        row[c] = r.get(c, '')

    rows.append(row)

fieldnames = ['index', 'question', 'image_path'] + LETTERS + ['answer', 'category']

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

if ext_fixed_images:
    print(f"INFO: {len(ext_fixed_images)} image(s) had wrong extension and were fixed:")
    for i, orig, fixed in ext_fixed_images[:5]:
        print(f"  row {i}: {os.path.basename(orig)} -> {os.path.basename(fixed)}")
    if len(ext_fixed_images) > 5:
        print(f"  ... and {len(ext_fixed_images) - 5} more")

if missing_images:
    print(f"WARNING: {len(missing_images)} image(s) not found on disk (even after trying all extensions):")
    for i, p in missing_images[:5]:
        print(f"  row {i}: {p}")
    if len(missing_images) > 5:
        print(f"  ... and {len(missing_images) - 5} more")
else:
    print("All images found.")

n_single = sum(1 for r in rows if ';' not in str(r['answer']))
n_multi  = sum(1 for r in rows if ';' in str(r['answer']))
print(f"Done: {len(rows)} rows ({n_single} single-answer, {n_multi} multi-answer) -> {OUTPUT}")
