import math
import re

import numpy as np
from datasets import Dataset


# ---------------------------------------------------------------------------
# Dataset filters
# ---------------------------------------------------------------------------

def filter_single_choice(dataset: Dataset) -> Dataset:
    """Keep only single-choice questions (single_choice_question == 'Yes')."""
    return dataset.filter(lambda x: x["single_choice_question"] == "Yes")


def filter_multi_choice(dataset: Dataset) -> Dataset:
    """Keep only multi-choice questions (single_choice_question == 'No')."""
    return dataset.filter(lambda x: x["single_choice_question"] == "No")


# ---------------------------------------------------------------------------
# doc_to_choice / doc_to_target helpers
# ---------------------------------------------------------------------------

def get_choice_labels(doc) -> list:
    """Return space-prefixed letter labels matching the number of choices."""
    return [" " + chr(ord("A") + i) for i in range(len(doc["choices"]))]


def get_target_index(doc) -> int:
    """Return the 0-based index of the first correct answer letter."""
    raw = doc["answer"]
    # Handle both "G)" and "A) and B)" formats — take the first letter
    letters = re.findall(r"[A-J]", raw)
    return ord(letters[0]) - ord("A")


def _parse_gold_indices(doc) -> list[int]:
    """Parse all correct answer indices from the answer field.

    Handles: "G)", "A) and B)", "A) and B) and C)" etc.
    Returns sorted list of 0-based indices.
    """
    letters = re.findall(r"[A-J]", doc["answer"])
    return sorted(ord(c) - ord("A") for c in letters)


# ---------------------------------------------------------------------------
# IR metric helpers
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain up to position k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because position is 1-indexed
    return dcg


def _ndcg_at_k(gold_indices: list[int], ranked_indices: list[int], k: int) -> float:
    """NDCG@K given gold answer indices and model-ranked indices."""
    gold_set = set(gold_indices)

    # Actual relevance in model's ranking order
    actual_rels = [1.0 if idx in gold_set else 0.0 for idx in ranked_indices]

    # Ideal relevance: all relevant docs at the top
    ideal_rels = sorted(actual_rels, reverse=True)

    dcg = _dcg(actual_rels, k)
    idcg = _dcg(ideal_rels, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _recall_at_k(gold_indices: list[int], ranked_indices: list[int], k: int) -> float:
    """Recall@K: fraction of gold answers found in top-K ranked results."""
    if not gold_indices:
        return 1.0
    top_k = set(ranked_indices[:k])
    hits = len(set(gold_indices) & top_k)
    return hits / len(gold_indices)


def _mrr(gold_indices: list[int], ranked_indices: list[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document."""
    gold_set = set(gold_indices)
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in gold_set:
            return 1.0 / rank
    return 0.0


def _rank_by_loglikelihood(results: list) -> list[int]:
    """Sort choice indices by descending loglikelihood, return ranked index list."""
    lls = [ll for ll, _is_greedy in results]
    return sorted(range(len(lls)), key=lambda i: -lls[i])


# ---------------------------------------------------------------------------
# process_results: single-choice (multiple_choice output_type)
# ---------------------------------------------------------------------------

def process_results_single(doc: dict, results: list) -> dict:
    """Compute IR metrics for single-choice questions.

    Metrics: ndcg@5, ndcg@10, recall@1, recall@5, mrr
    """
    ranked = _rank_by_loglikelihood(results)
    gold = _parse_gold_indices(doc)

    return {
        "ndcg@5": _ndcg_at_k(gold, ranked, 5),
        "ndcg@10": _ndcg_at_k(gold, ranked, 10),
        "recall@1": _recall_at_k(gold, ranked, 1),
        "recall@5": _recall_at_k(gold, ranked, 5),
        "mrr": _mrr(gold, ranked),
    }


# ---------------------------------------------------------------------------
# process_results: multi-choice (multiple_choice output_type)
# ---------------------------------------------------------------------------

def process_results_multi(doc: dict, results: list) -> dict:
    """Compute IR metrics for multi-choice questions.

    Metrics: ndcg@5, ndcg@10, recall@1, recall@5
    """
    ranked = _rank_by_loglikelihood(results)
    gold = _parse_gold_indices(doc)

    return {
        "ndcg@5": _ndcg_at_k(gold, ranked, 5),
        "ndcg@10": _ndcg_at_k(gold, ranked, 10),
        "recall@1": _recall_at_k(gold, ranked, 1),
        "recall@5": _recall_at_k(gold, ranked, 5),
    }
