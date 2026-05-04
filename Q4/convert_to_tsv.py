import json
import csv
import os
import subprocess
from PIL import Image
import pillow_heif

# 注册 HEIF/HEIC 支持，让 PIL 能够打开这些 ISO Media 文件
pillow_heif.register_heif_opener()

INPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q4/qa_dataset_unified_filtered_q4.jsonl'
OUTPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q4/benchmark_q4.tsv'
# Images extracted from q4_images.zip into q4_images/; JSONL paths use 'filtered_images/' prefix.
IMAGE_BASE = '/mnt/nfs/zyxing/VLMEvalKit/Q4/q4_images'

LETTERS = list('ABCDEF')
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff']

def resolve_image(abs_path):
    """解析图片路径，处理后缀名不匹配的情况"""
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
            img = img.convert('RGB') # 丢弃可能存在的透明通道
            img.save(filepath, format='JPEG')
            return True, None
        except Exception as e:
            return False, str(e)
    return False, None

# ================= 主流程 =================

with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f]

rows = []
missing_images = []
ext_fixed_images = []
iso_converted_images = []
iso_failed_images = []

for idx, r in enumerate(records):
    rel = r['image'].removeprefix('filtered_images/')
    abs_image = os.path.join(IMAGE_BASE, rel)
    
    # 1. 解析图片路径
    resolved, status = resolve_image(abs_image)

    if status == 'missing':
        missing_images.append((idx, abs_image))
    else:
        if status == 'ext_fixed':
            ext_fixed_images.append((idx, abs_image, resolved))
            
        # 2. 如果图片存在，检查是否为 ISO Media 并进行转换
        converted, error = check_and_convert_iso_media(resolved)
        if converted:
            iso_converted_images.append((idx, resolved))
        elif error:
            iso_failed_images.append((idx, resolved, error))

    # 3. 构建 TSV 行数据
    answer = r.get('correct_letters', r['gt_letter']).replace(',', ';').replace(' ', '')
    row = {
        'index': idx,
        'question': r['question'],
        'image_path': resolved,
        'answer': answer,
        'category': r.get('company', ''),
        'type': r.get('question_id', 'Q4'),
        'subtype': 'manufacturing_steps',
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

# ================= 写入 TSV =================

META_COLS = ['category', 'type', 'subtype', 'difficulty', 'manufacturer',
             'material_capabilities', 'process_capabilities', 'justification', 'answer_source']
fieldnames = ['index', 'question', 'image_path'] + LETTERS + ['answer'] + META_COLS

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

# ================= 打印统计信息 =================

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
print(f"\nDone: {len(rows)} rows ({n_single} single-answer, {n_multi} multi-answer) -> {OUTPUT}")