import json
import csv

INPUT = '/mnt/nfs/zyxing/VLMEvalKit/example/qa_output/benchmark_dataset.jsonl'
OUTPUT = '/mnt/nfs/zyxing/VLMEvalKit/example/benchmark_all_choice.tsv'
OLD_PREFIX = '/workspace/ztang3/kdd/manufacturing_qa_benchmark/filtered_images'
NEW_PREFIX = '/mnt/nfs/zyxing/VLMEvalKit/example/filtered_images'

LETTERS = list('ABCDEFGH')

with open(INPUT) as f:
    lines = [json.loads(l) for l in f]

max_choices = max(len(l['choices']) for l in lines)
choice_cols = LETTERS[:max_choices]

rows = []
for idx, l in enumerate(lines):
    choices = l['choices']
    ans = l['answer']['answer']
    if not isinstance(ans, list):
        ans = [ans]

    # Convert answer texts to letters
    ans_letters = []
    skip = False
    for a in ans:
        try:
            ans_letters.append(LETTERS[choices.index(a)])
        except ValueError:
            print(f"[WARN] answer '{a}' not found in choices at index {idx}, skipping")
            skip = True
            break
    if skip:
        continue

    # For generative eval: multi-answer stored as semicolon-separated letters (e.g. "A;C")
    ans_str = ';'.join(ans_letters)

    evidence = l.get('evidence', {})
    row = {
        'index': idx,
        'question': l['question'],
        'image_path': l['image_path'].replace(OLD_PREFIX, NEW_PREFIX),
        'answer': ans_str,
        'category': l.get('category', ''),
        'type': l.get('type', ''),
        'subtype': l.get('subtype', ''),
        'difficulty': l.get('difficulty', ''),
        'manufacturer': l.get('manufacturer', ''),
        'material_capabilities': ';'.join(evidence.get('material_capabilities', [])),
        'process_capabilities': ';'.join(evidence.get('process_capabilities', [])),
        'justification': l['answer'].get('justification', ''),
        'answer_source': l.get('answer_source', ''),
    }
    for i, col in enumerate(choice_cols):
        row[col] = choices[i] if i < len(choices) else ''

    rows.append(row)

META_COLS = ['category', 'type', 'subtype', 'difficulty', 'manufacturer',
             'material_capabilities', 'process_capabilities', 'justification', 'answer_source']
fieldnames = ['index', 'question', 'image_path'] + choice_cols + ['answer'] + META_COLS

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

n_single = sum(1 for r in rows if ';' not in r['answer'])
n_multi = sum(1 for r in rows if ';' in r['answer'])
print(f"Done: {len(rows)} rows ({n_single} single-answer, {n_multi} multi-answer) -> {OUTPUT}")
