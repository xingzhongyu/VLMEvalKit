import json
import csv
import re

INPUT = '/mnt/nfs/zyxing/VLMEvalKit/DataSample0420/text sample questions_20260420.json'
OUTPUT = '/mnt/nfs/zyxing/VLMEvalKit/example/benchmark_text_mcq.tsv'
PLACEHOLDER_IMAGE = '/mnt/nfs/zyxing/VLMEvalKit/example/blank_placeholder.png'

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def strip_letter_prefix(choice):
    """Remove leading 'A) ' style prefix from choice text."""
    return re.sub(r'^[A-Z]\)\s*', '', choice)


def parse_answer(answer_raw):
    """Convert 'H)', 'B); E)' -> 'H', 'B;E'. None/null -> ''."""
    if answer_raw is None or str(answer_raw).strip() in ('None', ''):
        return ''
    # Extract only letters that appear before ')' to avoid false positives
    letters = re.findall(r'([A-J])\)', str(answer_raw))
    return ';'.join(letters)


with open(INPUT, encoding='utf-8') as f:
    records = json.load(f)

max_choices = max(len(r['choices']) for r in records)
choice_cols = LETTERS[:max_choices]

rows = []
for idx, r in enumerate(records):
    choices_text = [strip_letter_prefix(c) for c in r['choices']]
    answer = parse_answer(r.get('answer'))

    row = {
        'index': idx,
        'question': r['question'],
        'image_path': PLACEHOLDER_IMAGE,
        'answer': answer,
        'category': r.get('category', ''),
        'difficulty': r.get('difficulty', ''),
        'masked_attribute': r.get('masked_attribute', ''),
        'n_correct': r.get('n_correct', ''),
    }
    for i, col in enumerate(choice_cols):
        row[col] = choices_text[i] if i < len(choices_text) else ''

    rows.append(row)

META_COLS = ['category', 'difficulty', 'masked_attribute', 'n_correct']
fieldnames = ['index', 'question', 'image_path'] + choice_cols + ['answer'] + META_COLS

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

n_zero = sum(1 for r in rows if r['answer'] == '')
n_single = sum(1 for r in rows if r['answer'] and ';' not in r['answer'])
n_multi = sum(1 for r in rows if r['answer'] and ';' in r['answer'])
print(f"Done: {len(rows)} rows ({n_single} single, {n_multi} multi, {n_zero} no-correct) -> {OUTPUT}")
