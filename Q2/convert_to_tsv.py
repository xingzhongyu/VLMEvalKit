import json
import csv
import os

INPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q2/qa_dataset_5000_q2_with_hard_subset.jsonl'
OUTPUT = '/mnt/nfs/zyxing/VLMEvalKit/Q2/benchmark_q2.tsv'
IMAGE_DIR = '/mnt/nfs/zyxing/VLMEvalKit/Q2/images'

LETTERS = list('ABCDEF')

with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f]

rows = []
for idx, r in enumerate(records):
    row = {
        'index': idx,
        'question': r['question'],
        'image_path': os.path.join(IMAGE_DIR, os.path.basename(r['image'])),
        'answer': r['gt_letter'],   # single letter, e.g. "B"
        'category': r.get('company', ''),
        'type': r.get('question_id', 'Q2'),
        'subtype': 'industry_sector',
        'difficulty': r.get('difficulty', ''),
        'manufacturer': r.get('company', ''),
        'material_capabilities': '',
        'process_capabilities': '',
        'justification': r.get('original_reasoning', ''),
        'answer_source': r.get('answer_source', ''),
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

n_single = sum(1 for r in rows if ';' not in r['answer'])
n_multi = sum(1 for r in rows if ';' in r['answer'])
print(f"Done: {len(rows)} rows ({n_single} single-answer, {n_multi} multi-answer) -> {OUTPUT}")
