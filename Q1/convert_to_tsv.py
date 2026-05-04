import json
import csv
import os
import subprocess
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

INPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q1/qa_dataset_unified_filtered_q1.jsonl'
OUTPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q1/benchmark_q1.tsv'
# Images extracted from q1_images.zip into q1_images/; JSONL paths use 'filtered_images/' prefix.
IMAGE_BASE = '/mnt/nfs/zyxing/VLMEvalKit/Q1/q1_images'

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


def is_iso_media(filepath):
    """使用 Linux 的 file 命令检查文件是否为 ISO Media (HEIC/AVIF)"""
    try:
        result = subprocess.check_output(['file', filepath], text=True)
        return 'ISO Media' in result
    except Exception:
        return False

def check_and_convert_iso_media(filepath):
    """检查并转换 ISO Media 文件为真正的 JPEG"""
    if is_iso_media(filepath):
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            img.save(filepath, format='JPEG')
            return True, None
        except Exception as e:
            return False, str(e)
    return False, None


with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f]

rows = []
missing_images = []
ext_fixed_images = []
iso_converted_images = []
iso_failed_images = []

for idx, r in enumerate(records):
    # JSONL paths use 'filtered_images/company/file'; images live under q1_images/company/file
    rel = r['image'].removeprefix('filtered_images/')
    abs_image = os.path.join(IMAGE_BASE, rel)
    resolved, status = resolve_image(abs_image)

    if status == 'missing':
        missing_images.append((idx, abs_image))
    else:
        if status == 'ext_fixed':
            ext_fixed_images.append((idx, abs_image, resolved))

        converted, error = check_and_convert_iso_media(resolved)
        if converted:
            iso_converted_images.append((idx, resolved))
        elif error:
            iso_failed_images.append((idx, resolved, error))

    # correct_letters is always single for Q1, but handle comma-sep just in case
    answer = r.get('correct_letters', r['gt_letter']).replace(',', ';')
    row = {
        'index': idx,
        'question': r['question'],
        'image_path': resolved,
        'answer': answer,
        'category': r.get('company', ''),
        'type': r.get('question_id', 'Q1'),
        'subtype': 'product_type',
        'difficulty': r.get('difficulty', ''),
        'manufacturer': r.get('company', ''),
        'material_capabilities': '',
        'process_capabilities': '',
        'justification': r.get('gpt_reason', ''),
        'answer_source': '',
    }
    for c in LETTERS:
        row[c] = r.get(c, '')
    rows.append(row)

META_COLS = ['category', 'type', 'subtype', 'difficulty', 'manufacturer',
             'material_capabilities', 'process_capabilities', 'justification', 'answer_source']
fieldnames = ['index', 'question', 'image_path'] + LETTERS + ['answer'] + META_COLS

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

print("\n--- 处理报告 ---")

if ext_fixed_images:
    print(f"INFO: {len(ext_fixed_images)} image(s) had wrong extension and were fixed:")
    for i, orig, fixed in ext_fixed_images[:5]:
        print(f"  row {i}: {os.path.basename(orig)} -> {os.path.basename(fixed)}")
    if len(ext_fixed_images) > 5:
        print(f"  ... and {len(ext_fixed_images) - 5} more")

if iso_converted_images:
    print(f"\nSUCCESS: {len(iso_converted_images)} ISO Media (HEIC/AVIF) image(s) detected and converted to real JPEG:")
    for i, p in iso_converted_images[:5]:
        print(f"  row {i}: {os.path.basename(p)}")
    if len(iso_converted_images) > 5:
        print(f"  ... and {len(iso_converted_images) - 5} more")

if iso_failed_images:
    print(f"\nERROR: {len(iso_failed_images)} ISO Media image(s) failed to convert:")
    for i, p, err in iso_failed_images[:5]:
        print(f"  row {i}: {os.path.basename(p)} (Error: {err})")
    if len(iso_failed_images) > 5:
        print(f"  ... and {len(iso_failed_images) - 5} more")

if missing_images:
    print(f"\nWARNING: {len(missing_images)} image(s) not found on disk (even after trying all extensions):")
    for i, p in missing_images[:5]:
        print(f"  row {i}: {p}")
    if len(missing_images) > 5:
        print(f"  ... and {len(missing_images) - 5} more")
elif not missing_images and not iso_failed_images:
    print("\nAll images found and valid.")

n_single = sum(1 for r in rows if ';' not in r['answer'])
n_multi  = sum(1 for r in rows if ';' in r['answer'])
print(f"Done: {len(rows)} rows ({n_single} single-answer, {n_multi} multi-answer) -> {OUTPUT}")
