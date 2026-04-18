import os
import os.path as osp
import re
import string
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd

from vlmeval.smp import (LMUDataRoot, dump, file_size, get_intermediate_file_path, load,
                         get_logger, toliststr)
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Judge helper (module-level so it works with ThreadPoolExecutor)             #
# --------------------------------------------------------------------------- #

def _judge_extract_one(model, item):
    """Call the judge model to extract selected letters from a prediction.

    Returns a frozenset of uppercase letter strings, or frozenset({'Z'}) on failure.
    """
    valid_letters = item['valid_letters']  # set of strings, e.g. {'A','B','C','D'}
    pred = item['prediction']
    question = item.get('question', '')
    options_str = item.get('options_str', '')

    prompt = (
        f'The following is a multiple-select question.\n'
        f'Question: {question}\nOptions:\n{options_str}\n\n'
        f'A student responded: "{pred}"\n\n'
        f'Extract ALL option letters the student selected from {sorted(valid_letters)}. '
        f'Output only the selected letters separated by semicolons (e.g. "A;C"). '
        f'If nothing can be determined, output "Z".'
    )

    retry = 3
    while retry:
        try:
            ans = model.generate(prompt)
            extracted = set(re.findall(r'[A-H]', ans.upper())) & valid_letters
            if extracted:
                return frozenset(extracted), ans
        except Exception as e:
            logger.warning(f'Judge call failed: {e}')
        retry -= 1

    return frozenset({'Z'}), 'Failed after retries'


def _process_one(model, item):
    """Top-level function for track_progress_rich (called as func(**task_dict))."""
    valid_letters = item['valid_letters']
    pred = item['prediction']

    # Fast-path: prediction matches clean semicolon/comma-separated letter pattern
    pred_stripped = pred.strip().upper()
    fast_match = re.fullmatch(r'[A-H](\s*[;,]\s*[A-H])*', pred_stripped)
    if fast_match:
        letters = frozenset(re.findall(r'[A-H]', pred_stripped)) & valid_letters
        if letters:
            return dict(pred_set=letters, log=f'fast: {pred_stripped}')

    # Single letter
    if re.fullmatch(r'[A-H]', pred_stripped) and pred_stripped in valid_letters:
        return dict(pred_set=frozenset({pred_stripped}), log=f'fast_single: {pred_stripped}')

    # Fall back to judge
    if model is None:
        # No judge available: collect all valid uppercase letters found in prediction
        letters = frozenset(
            c for c in re.findall(r'\b[A-H]\b', pred_stripped)
            if c in valid_letters
        )
        return dict(pred_set=letters or frozenset(), log=f'no_judge: {pred_stripped[:80]}')

    pred_set, raw = _judge_extract_one(model, item)
    return dict(pred_set=pred_set, log=f'judge: {raw!r:.80}')


# --------------------------------------------------------------------------- #
# Dataset class                                                               #
# --------------------------------------------------------------------------- #

class ManufactureMCQDataset(ImageBaseDataset):
    """Multi-select MCQ dataset for manufacturing image benchmarks.

    Expected TSV columns:
        index, question, image_path, A, B, ..., H, answer, [subtype, category, ...]

    The ``answer`` column stores ground-truth as semicolon-separated option
    letters, e.g. ``"A;C;D"`` for multi-select or ``"B"`` for single-select.
    """

    TYPE = 'MCQ'
    DATASET_URL = {'benchmark_all_choice': ''}
    DATASET_MD5 = {}
    force_use_dataset_prompt = True

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from vlmeval.tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        self.meta_only=False
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            c: line[c]
            for c in string.ascii_uppercase
            if c in line and not pd.isna(line[c])
        }
        options_prompt = '\n'.join(f'{k}. {v}' for k, v in options.items())

        prompt = (
            f'{question}\n'
            f'{options_prompt}\n'
            'This question may have one or more correct answers. '
            'Select ALL that apply and output their letters separated by semicolons '
            '(e.g. A or A;C;D). Output the letters only, nothing else.'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    # ----------------------------------------------------------------------- #
    # Evaluate                                                                 #
    # ----------------------------------------------------------------------- #

    def evaluate(self, eval_file, **judge_kwargs):
        nproc = judge_kwargs.pop('nproc', 4)
        model_name = judge_kwargs.get('model', 'exact_matching')

        if model_name == 'exact_matching':
            judge_model = None
            name_str = 'exact'
        else:
            judge_model = build_judge(**judge_kwargs)
            if not judge_model.working():
                logger.warning('Judge API not working, will use fast extraction only.')
                judge_model = None
                name_str = 'exact'
            else:
                name_str = model_name

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_multiselect', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        meta = self.data
        meta_idx = {str(row['index']): row for _, row in meta.iterrows()}

        # Load cached results
        cached = load(result_file) if osp.exists(result_file) else {}

        # Build task list for items not yet cached
        tasks, keys = [], []
        for i in range(len(data)):
            row = data.iloc[i]
            idx = str(row['index'])
            if idx in cached:
                continue

            valid_letters = frozenset(
                c for c in string.ascii_uppercase
                if c in row and not pd.isna(row[c])
            )

            meta_row = meta_idx.get(idx, row)
            options_str = '\n'.join(
                f'{c}. {meta_row[c]}'
                for c in sorted(valid_letters)
                if c in meta_row and not pd.isna(meta_row[c])
            )

            tasks.append(dict(
                model=judge_model,
                item=dict(
                    prediction=row['prediction'],
                    valid_letters=valid_letters,
                    question=meta_row.get('question', ''),
                    options_str=options_str,
                ),
            ))
            keys.append(idx)

        # Run (possibly parallel) extraction
        if tasks:
            results = track_progress_rich(
                _process_one, tasks, nproc=nproc, save=result_file, keys=keys
            )
            cached = load(result_file)

        # Compute metrics
        hits, f1s, logs, pred_strs, gt_strs = [], [], [], [], []
        for i in range(len(data)):
            row = data.iloc[i]
            idx = str(row['index'])
            gt_set = frozenset(str(row['answer']).split(';'))

            res = cached.get(idx, {})
            pred_set = res.get('pred_set', frozenset())
            log = res.get('log', 'missing')

            # Exact match
            hit = 1 if pred_set == gt_set else 0

            # F1
            if pred_set and gt_set:
                tp = len(pred_set & gt_set)
                precision = tp / len(pred_set)
                recall = tp / len(gt_set)
                denom = precision + recall
                f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
            else:
                f1 = 0.0

            hits.append(hit)
            f1s.append(f1)
            logs.append(log)
            pred_strs.append(';'.join(sorted(pred_set)))
            gt_strs.append(';'.join(sorted(gt_set)))

        data['hit'] = hits
        data['f1'] = f1s
        data['log'] = logs
        data['pred_letters'] = pred_strs
        data['gt_letters'] = gt_strs

        detail_file = get_intermediate_file_path(eval_file, f'_{name_str}_detail')
        dump(data, detail_file)

        # ------------------------------------------------------------------- #
        # Build summary report                                                 #
        # ------------------------------------------------------------------- #
        res = defaultdict(list)
        res['split'] = ['none']
        res['Overall_ExactMatch'] = [np.mean(hits)]
        res['Overall_F1'] = [np.mean(f1s)]

        for col in ['subtype', 'category']:
            if col not in data.columns:
                continue
            for val in sorted(data[col].dropna().unique()):
                sub = data[data[col] == val]
                safe = val.replace(' ', '_')
                res[f'{safe}_ExactMatch'] = [np.mean(sub['hit'])]
                res[f'{safe}_F1'] = [np.mean(sub['f1'])]

        acc = pd.DataFrame(res)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        logger.info(f'\n{acc.to_string(index=False)}')
        return acc
