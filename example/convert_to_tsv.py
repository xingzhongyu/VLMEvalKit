import json
import csv

INPUT = '/mnt/nfs/zyxing/VLMEvalKit/DataSample0420/vqa_data_samples.jsonl'
OUTPUT = '/mnt/nfs/zyxing/VLMEvalKit/example/benchmark_all_choice.tsv'
IMAGE_BASE = '/mnt/nfs/zyxing/VLMEvalKit/example'

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

with open(INPUT) as f:
    records = [json.loads(l) for l in f]

# Flatten: one row per (image, question) pair
flat = []
for rec in records:
    for q in rec['questions']:
        flat.append({
            'domain': rec['domain'],
            'image_path': f"{IMAGE_BASE}/{rec['image_path']}",
            'question': q['question'],
            'choices': q['choices'],
            'answer': q['answer'],          # list of answer text strings
            'difficulty': q.get('difficulty', ''),
            'answer_source': q.get('answer_source', ''),
            'reasoning': q.get('reasoning', ''),
        })

max_choices = max(len(r['choices']) for r in flat)
choice_cols = LETTERS[:max_choices]

rows = []
for idx, r in enumerate(flat):
    choices = r['choices']
    ans_texts = r['answer']
    if isinstance(ans_texts, str):
        ans_texts = [ans_texts]

    ans_letters = []
    skip = False
    for a in ans_texts:
        try:
            ans_letters.append(LETTERS[choices.index(a)])
        except ValueError:
            print(f"[WARN] answer '{a}' not found in choices at index {idx}, skipping")
            skip = True
            break
    if skip:
        continue

    row = {
        'index': idx,
        'question': r['question'],
        'image_path': r['image_path'],
        'answer': ';'.join(ans_letters),
        'domain': r['domain'],
        'difficulty': r['difficulty'],
        'answer_source': r['answer_source'],
        'reasoning': r['reasoning'],
    }
    for i, col in enumerate(choice_cols):
        row[col] = choices[i] if i < len(choices) else ''

    rows.append(row)

META_COLS = ['domain', 'difficulty', 'answer_source', 'reasoning']
fieldnames = ['index', 'question', 'image_path'] + choice_cols + ['answer'] + META_COLS

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

n_single = sum(1 for r in rows if ';' not in r['answer'])
n_multi = sum(1 for r in rows if ';' in r['answer'])
print(f"Done: {len(rows)} rows ({n_single} single-answer, {n_multi} multi-answer) -> {OUTPUT}")
