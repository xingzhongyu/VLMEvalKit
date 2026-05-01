import os
import os.path as osp
import re
import string
from collections import defaultdict

import numpy as np
import pandas as pd

from vlmeval.smp import (LMUDataRoot, dump, file_size, get_intermediate_file_path, load,
                         get_logger, toliststr)
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Judge helpers (module-level so they work with ThreadPoolExecutor)           #
# --------------------------------------------------------------------------- #

def _judge_extract_one(model, item):
    """Call the judge model to extract selected letters from a prediction.

    Returns a frozenset of uppercase letter strings, or frozenset({'Z'}) on failure.
    """
    valid_letters = item['valid_letters']
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
            extracted = set(re.findall(r'[A-Z]', ans.upper())) & valid_letters
            if extracted:
                return frozenset(extracted), ans
        except Exception as e:
            logger.warning(f'Judge call failed: {e}')
        retry -= 1

    return frozenset({'Z'}), 'Failed after retries'


def _process_one(model, item):
    """Extract selected option letters from a prediction string.

    Tries last-line extraction first (supports reasoning + final-answer format),
    then falls back to full-string regex, then to LLM judge.
    """
    valid_letters = item['valid_letters']
    pred = item['prediction']

    # Last non-empty line first (model outputs reasoning then answer on final line)
    lines = [l.strip() for l in pred.strip().split('\n') if l.strip()]
    if lines:
        last = lines[-1].upper()
        if re.fullmatch(r'[A-Z](\s*[;,]\s*[A-Z])*', last):
            letters = frozenset(re.findall(r'[A-Z]', last)) & valid_letters
            if letters:
                return dict(pred_set=letters, log=f'last_line: {last}')

    # Full prediction matches clean letter pattern
    pred_stripped = pred.strip().upper()
    fast_match = re.fullmatch(r'[A-Z](\s*[;,]\s*[A-Z])*', pred_stripped)
    if fast_match:
        letters = frozenset(re.findall(r'[A-Z]', pred_stripped)) & valid_letters
        if letters:
            return dict(pred_set=letters, log=f'fast: {pred_stripped}')

    # Single letter
    if re.fullmatch(r'[A-Z]', pred_stripped) and pred_stripped in valid_letters:
        return dict(pred_set=frozenset({pred_stripped}), log=f'fast_single: {pred_stripped}')

    # No judge: collect all valid uppercase letters found as whole words
    if model is None:
        letters = frozenset(
            c for c in re.findall(r'\b[A-Z]\b', pred_stripped)
            if c in valid_letters
        )
        return dict(pred_set=letters or frozenset(), log=f'no_judge: {pred_stripped[:80]}')

    pred_set, raw = _judge_extract_one(model, item)
    return dict(pred_set=pred_set, log=f'judge: {raw!r:.80}')


def _judge_reasoning_one(model, item):
    """Judge whether the model's reasoning correctly explains the correct answer.

    Returns 1 (correct) or 0 (incorrect/undetermined).
    """
    prompt = (
        'You are evaluating a model\'s reasoning for a manufacturing image question.\n\n'
        f'Question: {item["question"]}\n'
        f'Options:\n{item["options_str"]}\n'
        f'Ground truth answer: {item["gt_letters"]} ({item["gt_text"]})\n'
        f'Ground truth reasoning: {item["gt_reasoning"]}\n\n'
        f'Model response: {item["prediction"]}\n\n'
        'Does the model\'s reasoning correctly explain why the correct option(s) were selected? '
        'Output only "Yes" or "No".'
    )

    retry = 3
    while retry:
        try:
            ans = model.generate(prompt)
            ans_lower = ans.strip().lower()
            if ans_lower.startswith('yes'):
                return 1
            elif ans_lower.startswith('no'):
                return 0
        except Exception as e:
            logger.warning(f'Reasoning judge call failed: {e}')
        retry -= 1

    return 0


def _judge_reasoning_wrapper(model, item):
    """Wrapper for track_progress_rich compatibility."""
    correct = _judge_reasoning_one(model, item)
    return dict(correct=correct)


# --------------------------------------------------------------------------- #
# Dataset class                                                               #
# --------------------------------------------------------------------------- #

class ManufactureMCQDataset(ImageBaseDataset):
    """Multi-select MCQ dataset for manufacturing image benchmarks.

    Expected TSV columns:
        index, question, image_path, A, B, ..., answer, category, type,
        subtype, difficulty, manufacturer, material_capabilities,
        process_capabilities, justification, answer_source

    ``answer`` stores ground-truth as semicolon-separated option letters,
    e.g. ``"A;C"`` for multi-select or ``"B"`` for single-select.

    Metrics
    -------
    Option Accuracy (F1)  : token-level F1 between predicted and GT option sets.
    Strict Accuracy       : LLM judge evaluates whether the model's reasoning
                            correctly explains the correct answer (requires judge).
    """

    TYPE = 'MCQ'
    DATASET_URL = {'benchmark_all_choice': '', 'benchmark_text_mcq': ''}
    DATASET_MD5 = {}
    DATA_ROOT = '/mnt/nfs/zyxing/VLMEvalKit/example'
    force_use_dataset_prompt = True

    def __init__(self, dataset='benchmark_all_choice', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        # Override img_root so relative image_path values resolve under DATA_ROOT
        self.img_root = self.DATA_ROOT

    def load_data(self, dataset):
        data_path = osp.join(self.DATA_ROOT, f'{dataset}.tsv')
        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

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
            'First explain your reasoning step by step. '
            'Then on the last line output only the selected option letters '
            'separated by semicolons (e.g. A or A;C;D).'
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
        reasoning_file = get_intermediate_file_path(eval_file, f'_{name_str}_reasoning', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        meta = self.data
        meta_idx = {str(row['index']): row for _, row in meta.iterrows()}

        # ------------------------------------------------------------------- #
        # Phase 1: Extract predicted option letters                           #
        # ------------------------------------------------------------------- #
        cached = load(result_file) if osp.exists(result_file) else {}

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
                    question=str(meta_row.get('question', '')),
                    options_str=options_str,
                ),
            ))
            keys.append(idx)

        if tasks:
            track_progress_rich(_process_one, tasks, nproc=nproc, save=result_file, keys=keys)
            cached = load(result_file)

        # ------------------------------------------------------------------- #
        # Phase 2: Reasoning judge (Strict Accuracy) — requires judge model   #
        # ------------------------------------------------------------------- #
        reasoning_cached = load(reasoning_file) if osp.exists(reasoning_file) else {}

        if judge_model is not None:
            r_tasks, r_keys = [], []
            for i in range(len(data)):
                row = data.iloc[i]
                idx = str(row['index'])
                if idx in reasoning_cached:
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
                ans_raw = meta_row.get('answer', '')
                gt_letters = '' if pd.isna(ans_raw) or str(ans_raw).strip() in ('None', 'nan') else str(ans_raw)
                gt_text = '; '.join(
                    str(meta_row.get(l, '')) for l in gt_letters.split(';') if l.strip()
                )
                r_tasks.append(dict(
                    model=judge_model,
                    item=dict(
                        prediction=row['prediction'],
                        question=str(meta_row.get('question', '')),
                        options_str=options_str,
                        gt_letters=gt_letters,
                        gt_text=gt_text,
                        gt_reasoning=str(meta_row.get('justification', '')),
                    ),
                ))
                r_keys.append(idx)

            if r_tasks:
                track_progress_rich(
                    _judge_reasoning_wrapper, r_tasks,
                    nproc=nproc, save=reasoning_file, keys=r_keys
                )
                reasoning_cached = load(reasoning_file)

        # ------------------------------------------------------------------- #
        # Compute per-row metrics                                              #
        # ------------------------------------------------------------------- #
        hits, f1s, strict_accs = [], [], []
        logs, pred_strs, gt_strs = [], [], []

        for i in range(len(data)):
            row = data.iloc[i]
            idx = str(row['index'])
            ans_raw = row['answer']
            if pd.isna(ans_raw) or str(ans_raw).strip() in ('', 'None', 'nan'):
                gt_set = frozenset()
            else:
                gt_set = frozenset(x.strip() for x in str(ans_raw).split(';') if x.strip())

            res = cached.get(idx, {})
            pred_set = res.get('pred_set', frozenset())
            log = res.get('log', 'missing')

            hit = 1 if pred_set == gt_set else 0

            if pred_set == gt_set:
                f1 = 1.0
            elif pred_set and gt_set:
                tp = len(pred_set & gt_set)
                precision = tp / len(pred_set)
                recall = tp / len(gt_set)
                denom = precision + recall
                f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
            else:
                f1 = 0.0

            r_res = reasoning_cached.get(idx, {})
            if hit:
                strict = r_res.get('correct', np.nan)
            else:
                strict = 0

            hits.append(hit)
            f1s.append(f1)
            strict_accs.append(strict)
            logs.append(log)
            pred_strs.append(';'.join(sorted(pred_set)))
            gt_strs.append(';'.join(sorted(gt_set)))

        data['hit'] = hits
        data['option_f1'] = f1s
        data['strict_acc'] = strict_accs
        data['log'] = logs
        data['pred_letters'] = pred_strs
        data['gt_letters'] = gt_strs

        detail_file = get_intermediate_file_path(eval_file, f'_{name_str}_detail')
        dump(data, detail_file)

        # ------------------------------------------------------------------- #
        # Summary report                                                       #
        # ------------------------------------------------------------------- #
        res = defaultdict(list)
        res['split'] = ['none']
        res['Overall_OptionAccuracy_F1'] = [np.mean(f1s)]
        res['Overall_ExactMatch'] = [np.mean(hits)]

        valid_strict = [x for x in strict_accs if not (isinstance(x, float) and np.isnan(x))]
        if valid_strict:
            res['Overall_StrictAccuracy'] = [np.mean(valid_strict)]

        for col in ['category', 'difficulty']:
            if col not in data.columns:
                continue
            for val in sorted(data[col].dropna().unique()):
                sub = data[data[col] == val]
                safe = str(val).replace(' ', '_')
                res[f'{safe}_F1'] = [np.mean(sub['option_f1'])]
                sub_strict = [x for x in sub['strict_acc']
                              if not (isinstance(x, float) and np.isnan(x))]
                if sub_strict:
                    res[f'{safe}_StrictAccuracy'] = [np.mean(sub_strict)]

        acc = pd.DataFrame(res)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        logger.info(f'\n{acc.to_string(index=False)}')
        return acc


def _read_throughput(eval_file):
    """Read throughput_samples_per_sec from the sidecar JSON written by inference.py."""
    import json as _json
    timing_file = eval_file + '_timing.json'
    if osp.exists(timing_file):
        with open(timing_file) as f:
            t = _json.load(f)
        return t.get('throughput_samples_per_sec')
    return None


class ManufactureMCQDatasetQBase(ImageBaseDataset):
    """Parameterizable single-choice MCQ benchmark base for Q1-Q5 datasets.

    Subclasses must set DATA_ROOT and DATASET_URL. Common workflow:
      1. Run evaluate() on each QN dataset independently → saves *_detail.tsv
      2. Call combine_manufacture_q_results() with the detail files you have so far

    TSV columns expected: index, question, image_path, A..F, answer, category, difficulty
    """

    TYPE = 'MCQ'
    DATASET_URL = {}
    DATASET_MD5 = {}
    DATA_ROOT = None  # subclasses must set this
    force_use_dataset_prompt = True

    def __init__(self, dataset, skip_noimg=True):
        assert self.DATA_ROOT is not None, f'{self.__class__.__name__}.DATA_ROOT must be set'
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.img_root = self.DATA_ROOT

    def load_data(self, dataset):
        data_path = osp.join(self.DATA_ROOT, f'{dataset}.tsv')
        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

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
            'Output the single letter of the correct option only (e.g. A).'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    @staticmethod
    def _extract_letter(pred_str, valid_letters):
        """Pull a single option letter out of a free-form prediction string."""
        if pred_str in valid_letters:
            return pred_str
        lines = [ln.strip() for ln in pred_str.split('\n') if ln.strip()]
        for line in reversed(lines):
            m = re.search(r'\b([A-Z])\b', line)
            if m and m.group(1) in valid_letters:
                return m.group(1)
        m = re.search(r'\b([A-Z])\b', pred_str)
        if m and m.group(1) in valid_letters:
            return m.group(1)
        return ''

    @staticmethod
    def _metrics(pred_list, gt_list, hit_list):
        from sklearn.metrics import f1_score as _f1
        acc = float(np.mean(hit_list)) if hit_list else 0.0
        valid = [(p, g) for p, g in zip(pred_list, gt_list) if g != '']
        if valid:
            vp, vg = zip(*valid)
            f1 = float(_f1(list(vg), list(vp), average='macro', zero_division=0))
        else:
            f1 = 0.0
        return acc, f1

    def evaluate(self, eval_file, **judge_kwargs):
        judge_kwargs.pop('nproc', None)

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        meta_idx = {str(row['index']): row for _, row in self.data.iterrows()}

        preds, gts = [], []
        for i in range(len(data)):
            row = data.iloc[i]
            idx = str(row['index'])
            meta_row = meta_idx.get(idx, row)

            valid_letters = frozenset(
                c for c in string.ascii_uppercase
                if c in meta_row and not pd.isna(meta_row[c])
            )
            pred_letter = self._extract_letter(row['prediction'].strip().upper(), valid_letters)
            ans_raw = row['answer']
            gt_letter = str(ans_raw).strip().upper() if not pd.isna(ans_raw) else ''

            preds.append(pred_letter)
            gts.append(gt_letter)

        data['pred_letter'] = preds
        data['gt_letter'] = gts
        data['hit'] = [int(p == g and g != '') for p, g in zip(preds, gts)]

        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(data, detail_file)

        overall_acc, overall_f1 = self._metrics(preds, gts, data['hit'].tolist())

        throughput = _read_throughput(eval_file)

        rows = []
        row0 = {'split': 'Overall', 'ACC': overall_acc, 'F1_macro': overall_f1}
        if throughput is not None:
            row0['throughput(samples/s)'] = round(throughput, 3)
        rows.append(row0)

        for col in ['category', 'difficulty']:
            if col not in data.columns:
                continue
            for val in sorted(data[col].dropna().unique()):
                sub = data[data[col] == val]
                a, f = self._metrics(
                    sub['pred_letter'].tolist(),
                    sub['gt_letter'].tolist(),
                    sub['hit'].tolist(),
                )
                entry = {'split': str(val), 'ACC': a, 'F1_macro': f}
                if throughput is not None:
                    entry['throughput(samples/s)'] = round(throughput, 3)
                rows.append(entry)

        acc_df = pd.DataFrame(rows)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc_df, score_file)

        logger.info(f'\n{acc_df.to_string(index=False)}')
        return acc_df


# --------------------------------------------------------------------------- #
# Per-batch dataset classes (update DATA_ROOT when each batch arrives)        #
# --------------------------------------------------------------------------- #

class ManufactureMCQDatasetQ1(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q1': ''}
    DATA_ROOT = '/mnt/nfs/zyxing/VLMEvalKit/Q1'

    def __init__(self, dataset='benchmark_q1', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ2(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q2': ''}
    DATA_ROOT = '/mnt/nfs/zyxing/VLMEvalKit/Q2'

    def __init__(self, dataset='benchmark_q2', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ3(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q3': ''}
    DATA_ROOT = '/mnt/nfs/zyxing/VLMEvalKit/Q3'

    def __init__(self, dataset='benchmark_q3', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ4(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q4': ''}
    DATA_ROOT = '/mnt/nfs/zyxing/VLMEvalKit/Q4'

    def __init__(self, dataset='benchmark_q4', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ5(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q5': ''}
    DATA_ROOT = '/mnt/nfs/zyxing/VLMEvalKit/Q5'

    def __init__(self, dataset='benchmark_q5', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


# --------------------------------------------------------------------------- #
# Cross-dataset aggregation                                                   #
# --------------------------------------------------------------------------- #

def combine_manufacture_q_results(detail_files, eval_files=None, output_file=None):
    """Aggregate metrics across multiple ManufactureMCQDatasetQ* evaluations.

    Call this after running evaluate() on whichever Q-datasets you have so far.
    Each evaluate() saves a *_detail.tsv; pass those paths here.

    Parameters
    ----------
    detail_files : dict[str, str]
        {label: path_to_detail_tsv}  e.g. {'Q1': '/path/Q1_detail.tsv', 'Q2': '...'}
    eval_files : dict[str, str], optional
        {label: original_eval_file} — used to read *_timing.json sidecars for
        throughput. Keys must match detail_files. Omit if timing is not needed.
    output_file : str, optional
        If given, the combined metrics DataFrame is saved here as CSV.

    Returns
    -------
    pd.DataFrame with columns: split, ACC, F1_macro, n[, throughput(samples/s)]
    """
    frames = []
    for label, path in detail_files.items():
        df = load(path)
        df['_dataset'] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if 'pred_letter' not in combined.columns or 'gt_letter' not in combined.columns:
        raise ValueError(
            'detail files must contain pred_letter and gt_letter columns '
            '(produced by ManufactureMCQDatasetQBase.evaluate)'
        )

    throughputs = {}
    if eval_files:
        for label, ef in eval_files.items():
            t = _read_throughput(ef)
            if t is not None:
                throughputs[label] = round(t, 3)

    def _add_row(label, sub, throughput=None):
        preds = sub['pred_letter'].tolist()
        gts = sub['gt_letter'].tolist()
        hits = sub['hit'].tolist()
        acc, f1 = ManufactureMCQDatasetQBase._metrics(preds, gts, hits)
        entry = {'split': label, 'ACC': acc, 'F1_macro': f1, 'n': len(sub)}
        if throughput is not None:
            entry['throughput(samples/s)'] = throughput
        rows.append(entry)

    rows = []
    _add_row('Overall', combined)

    for label in detail_files:
        _add_row(label, combined[combined['_dataset'] == label],
                 throughput=throughputs.get(label))

    for col in ['category', 'difficulty']:
        if col not in combined.columns:
            continue
        for val in sorted(combined[col].dropna().unique()):
            _add_row(str(val), combined[combined[col] == val])

    result = pd.DataFrame(rows)

    if output_file:
        dump(result, output_file)

    logger.info(f'\n{result.to_string(index=False)}')
    return result
