import os
import os.path as osp
import re
import string

import numpy as np
import pandas as pd

from vlmeval.smp import (dump, file_size, get_intermediate_file_path, load,
                         get_logger, toliststr)
from .image_base import ImageBaseDataset

logger = get_logger(__name__)
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

def _read_throughput(eval_file):
    """Read throughput_samples_per_sec from the sidecar JSON written by inference.py."""
    import json as _json
    timing_file = eval_file + '_timing.json'
    if osp.exists(timing_file):
        with open(timing_file) as f:
            t = _json.load(f)
        return t.get('throughput_samples_per_sec')
    return None


# --------------------------------------------------------------------------- #
# Q-Series Base Classes                                                       #
# --------------------------------------------------------------------------- #

class ManufactureMCQDatasetQBase(ImageBaseDataset):
    """Parameterizable single-choice MCQ benchmark base for Q1-Q5 datasets."""

    TYPE = 'MCQ'
    DATASET_URL = {}
    DATASET_MD5 = {}
    DATA_ROOT = None  # subclasses must set this
    DEV_DATA_ROOT = None  # optional separate root for *_dev datasets
    force_use_dataset_prompt = True

    def __init__(self, dataset, skip_noimg=True):
        assert self.DATA_ROOT is not None, f'{self.__class__.__name__}.DATA_ROOT must be set'
        self.data_root = self.resolve_data_root(dataset)
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.img_root = self.data_root

    def resolve_data_root(self, dataset):
        if dataset.endswith('_dev') and self.DEV_DATA_ROOT is not None:
            return self.DEV_DATA_ROOT
        return self.DATA_ROOT

    def load_data(self, dataset):
        data_path = osp.join(self.data_root, f'{dataset}.tsv')
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
            'Please directly provide your final answer by enclosing the single letter of the correct option in <answer> tags (e.g., <answer>A</answer>).'
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
        """Pull a single option letter out of a free-form prediction string with strict fallback."""
        pred_str = pred_str.strip()
        
        # 1. Exact match
        if pred_str in valid_letters:
            return pred_str
            
        # 2. Try <answer> tags
        m_tag = re.search(r'<answer>\s*([A-Z])\s*</answer>', pred_str, re.IGNORECASE)
        if m_tag:
            ans = m_tag.group(1).upper()
            if ans in valid_letters:
                return ans

        # 3. Strict Fallback Strategy
        lines = [ln.strip() for ln in pred_str.split('\n') if ln.strip()]
        if not lines:
            return ''
            
        last_line = lines[-1].upper()
        
        # 3a. Look for explicit prefix (e.g., "Answer: A", "Option B")
        m_prefix = re.search(r'(?:ANSWER|CORRECT|OPTION)S?[\s:=]+([A-Z])\b', last_line)
        if m_prefix:
            ans = m_prefix.group(1)
            if ans in valid_letters:
                return ans
                
        # 3b. Look for bracketed option (e.g., "(A)", "[C]")
        m_bracket = re.search(r'[\([]([A-Z])[\)]]', last_line)
        if m_bracket:
            ans = m_bracket.group(1)
            if ans in valid_letters:
                return ans
                
        # 3c. Short line fallback (e.g., last line is just "A" or "A.")
        if len(last_line) < 10 and not re.search(r'[a-z]', lines[-1].lower()):
            m_short = re.search(r'\b([A-Z])\b', last_line)
            if m_short:
                ans = m_short.group(1)
                if ans in valid_letters:
                    return ans
                    
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

    def evaluate(self, eval_file, **kwargs):
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
            pred_letter = self._extract_letter(row['prediction'].strip(), valid_letters)
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


class ManufactureMCQDatasetQMultiBase(ManufactureMCQDatasetQBase):
    """Multi-select variant of the Q-series base."""

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
            'Please directly provide your final answer by enclosing the selected option letters '
            'separated by semicolons in <answer> tags (e.g., <answer>A;C;D</answer>).'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    @staticmethod
    def _extract_multi_letters(pred_str, valid_letters):
        """Pull a set of option letters out of a free-form prediction string with strict fallback."""
        pred_str = str(pred_str).strip()
        
        # 0. 提前拦截最简单的情况：纯单个字母，或者类似 "A;B", "A, C" 的纯选项字符串
        # 只要整个字符串只包含大写字母和分隔符，就直接提取
        if re.match(r'^[A-Z\s,;&]+$', pred_str.upper()):
            letters = set(re.findall(r'[A-Z]', pred_str.upper()))
            valid_found = frozenset(letters & valid_letters)
            if valid_found:
                return valid_found

        # 1. Try <answer> tags
        m_tag = re.search(r'<answer>\s*(.*?)\s*</answer>', pred_str, re.IGNORECASE | re.DOTALL)
        if m_tag:
            content = m_tag.group(1).upper()
            letters = set(re.findall(r'[A-Z]', content))
            valid_found = frozenset(letters & valid_letters)
            if valid_found:
                return valid_found

        # 2. Strict Fallback Strategy
        lines = [ln.strip() for ln in pred_str.split('\n') if ln.strip()]
        if not lines:
            return frozenset()
            
        last_line = lines[-1].upper()
        
        # 2a. Look for explicit prefix (e.g., "ANSWER: A, C", "OPTIONS: A AND B")
        m_prefix = re.search(r'(?:ANSWER|CORRECT|OPTION)S?[\s:=]+([A-Z\s,;AND&]+)', last_line)
        if m_prefix:
            letters = set(re.findall(r'[A-Z]', m_prefix.group(1)))
            valid_found = frozenset(letters & valid_letters)
            if valid_found:
                return valid_found

        # 2b. Look for bracketed options (e.g., "(A)", "[B]")
        bracket_letters = set(re.findall(r'[\([]([A-Z])[\)]]', last_line))
        valid_found = frozenset(bracket_letters & valid_letters)
        if valid_found:
            return valid_found
            
        # 2c. Short line fallback (e.g., last line is just "A, C" or "A")
        # 修复了原来的 bug：直接判断原字符串中是否不包含小写字母，而不是 lower() 之后
        if len(last_line) < 10 and not re.search(r'[a-z]', lines[-1]):
            letters = set(re.findall(r'[A-Z]', last_line))
            valid_found = frozenset(letters & valid_letters)
            if valid_found:
                return valid_found

        return frozenset()

    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        meta = self.data
        meta_idx = {str(row['index']): row for _, row in meta.iterrows()}

        hits, f1s, logs, pred_strs, gt_strs = [], [], [], [], []

        for i in range(len(data)):
            row = data.iloc[i]
            idx = str(row['index'])
            meta_row = meta_idx.get(idx, row)

            valid_letters = frozenset(
                c for c in string.ascii_uppercase
                if c in meta_row and not pd.isna(meta_row[c])
            )
            
            pred_set = self._extract_multi_letters(row['prediction'], valid_letters)

            ans_raw = row['answer']
            if pd.isna(ans_raw) or str(ans_raw).strip() in ('', 'None', 'nan'):
                gt_set = frozenset()
            else:
                gt_set = frozenset(x.strip().upper() for x in str(ans_raw).split(';') if x.strip())

            hit = int(pred_set == gt_set)

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

            hits.append(hit)
            f1s.append(f1)
            logs.append('Extracted via regex/tags')
            pred_strs.append(';'.join(sorted(pred_set)))
            gt_strs.append(';'.join(sorted(gt_set)))

        data['hit'] = hits
        data['option_f1'] = f1s
        data['log'] = logs
        data['pred_letters'] = pred_strs
        data['gt_letters'] = gt_strs

        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(data, detail_file)

        throughput = _read_throughput(eval_file)

        rows = []
        row0 = {
            'split': 'Overall',
            'ExactMatch': np.mean(hits),
            'OptionF1': np.mean(f1s),
            'n': len(hits),
        }
        if throughput is not None:
            row0['throughput(samples/s)'] = round(throughput, 3)
        rows.append(row0)

        for col in ['category', 'difficulty']:
            if col not in data.columns:
                continue
            for val in sorted(data[col].dropna().unique()):
                sub = data[data[col] == val]
                entry = {
                    'split': str(val),
                    'ExactMatch': np.mean(sub['hit']),
                    'OptionF1': np.mean(sub['option_f1']),
                    'n': len(sub),
                }
                if throughput is not None:
                    entry['throughput(samples/s)'] = round(throughput, 3)
                rows.append(entry)

        acc_df = pd.DataFrame(rows)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc_df, score_file)

        logger.info(f'\n{acc_df.to_string(index=False)}')
        return acc_df


# --------------------------------------------------------------------------- #
# Per-batch dataset classes                                                   #
# --------------------------------------------------------------------------- #

class ManufactureMCQDatasetQ1(ManufactureMCQDatasetQMultiBase):
    DATASET_URL = {'benchmark_q1': '', 'benchmark_q1_dev': ''}
    DATA_ROOT = osp.join(REPO_ROOT, 'Q1')
    DEV_DATA_ROOT = osp.join(REPO_ROOT, 'dev', 'Q1_dev')

    def __init__(self, dataset='benchmark_q1', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ2(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q2': '', 'benchmark_q2_dev': ''}
    DATA_ROOT = osp.join(REPO_ROOT, 'Q2')
    DEV_DATA_ROOT = osp.join(REPO_ROOT, 'dev', 'Q2_dev')

    def __init__(self, dataset='benchmark_q2', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ3(ManufactureMCQDatasetQMultiBase):
    DATASET_URL = {'benchmark_q3': '', 'benchmark_q3_dev': ''}
    DATA_ROOT = osp.join(REPO_ROOT, 'Q3')
    DEV_DATA_ROOT = osp.join(REPO_ROOT, 'dev', 'Q3_dev')

    def __init__(self, dataset='benchmark_q3', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ4(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q4': '', 'benchmark_q4_dev': ''}
    DATA_ROOT = osp.join(REPO_ROOT, 'Q4')
    DEV_DATA_ROOT = osp.join(REPO_ROOT, 'dev', 'Q4_dev')

    def __init__(self, dataset='benchmark_q4', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


class ManufactureMCQDatasetQ5(ManufactureMCQDatasetQBase):
    DATASET_URL = {'benchmark_q5': '', 'benchmark_q5_dev': ''}
    DATA_ROOT = osp.join(REPO_ROOT, 'Q5')
    DEV_DATA_ROOT = osp.join(REPO_ROOT, 'dev', 'Q5_dev')

    def __init__(self, dataset='benchmark_q5', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

# --------------------------------------------------------------------------- #
# Cross-dataset aggregation                                                   #
# --------------------------------------------------------------------------- #

def combine_manufacture_q_results(detail_files, eval_files=None, output_file=None):
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
