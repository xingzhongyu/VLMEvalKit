import json
import os
import re

import requests
from datasets import Dataset

# ---------------------------------------------------------------------------
# Configurable settings (override via env vars)
# ---------------------------------------------------------------------------

_LLM_JUDGE_URL = os.environ.get("LLM_JUDGE_URL", "http://localhost:8000/v1/chat/completions")
_LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# ---------------------------------------------------------------------------
# Dataset filters
# ---------------------------------------------------------------------------

def filter_feasibility_qa(dataset):
    return dataset.filter(lambda x: x["type_id"] == "feasibility_qa")

def filter_capability_gap(dataset):
    return dataset.filter(lambda x: x["type_id"] == "capability_gap")

def get_capability_gap_target(doc):
    """Return the correct answer letter (A/B/C/D) for display/logging."""
    return chr(ord("A") + doc["correct_index"])

# ---------------------------------------------------------------------------
# LLM-as-judge helper (shared by both task types)
# ---------------------------------------------------------------------------

def _call_llm_judge(question, evidence, correct_answer, pred_justification):
    prompt = f"""You are an expert evaluator.
Question: {question}
Evidence: {evidence}
Correct Answer: {correct_answer}
Student's Justification: {pred_justification}

Task: Determine if the student's justification is logically sound and correctly uses the evidence to arrive at the correct answer.
If the justification is correct and makes sense, output exactly 'PASS'.
If the justification is wrong, hallucinates, or illogical, output exactly 'FAIL'.
Output nothing else."""

    payload = {
        "model": _LLM_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict and fair judge."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    try:
        response = requests.post(_LLM_JUDGE_URL, json=payload, timeout=10)
        judge_output = response.json()["choices"][0]["message"]["content"].strip().upper()
        return "PASS" in judge_output
    except Exception as e:
        print(f"Judge API Error: {e}")
        return False

# ---------------------------------------------------------------------------
# process_results: Type II — Capability Gap Reasoning (generate_until)
# ---------------------------------------------------------------------------

def process_capability_gap_results(doc: dict, results: list) -> dict:
    """Score one capability_gap example.

    Returns acc (answer-only) and strict_acc (answer + justification quality).
    """
    pred_text = results[0] if results else ""
    correct_letter = chr(ord("A") + doc["correct_index"])

    answer_match = re.search(r"Answer:\s*([A-D])", pred_text, re.IGNORECASE)
    justification_match = re.search(r"Justification:\s*(.*)", pred_text, re.IGNORECASE | re.DOTALL)

    pred_letter = answer_match.group(1).upper() if answer_match else None
    pred_justification = justification_match.group(1).strip() if justification_match else ""

    if pred_letter != correct_letter:
        return {"acc": 0.0, "strict_acc": 0.0}

    # Answer is correct — evaluate justification via LLM judge
    ref_justification = (doc.get("justification") or "").strip()
    if not ref_justification:
        # No reference justification to verify against
        return {"acc": 1.0, "strict_acc": 1.0}

    is_just_ok = _call_llm_judge(
        question=doc["question"],
        evidence=doc["evidence"],
        correct_answer=correct_letter,
        pred_justification=pred_justification,
    )
    return {"acc": 1.0, "strict_acc": 1.0 if is_just_ok else 0.0}

# ---------------------------------------------------------------------------
# process_results: Type I — Feasibility QA (generate_until)
# ---------------------------------------------------------------------------

def process_feasibility_results(doc: dict, results: list) -> dict:
    """Score one feasibility_qa example.

    Returns acc (answer-only) and strict_acc (answer + justification quality).
    """
    pred_text = results[0] if results else ""
    correct_answer = doc["answer"]["answer"]  # "Yes" or "No"
    ref_justification = (doc["answer"].get("justification") or "").strip()

    answer_match = re.search(r"Answer:\s*(Yes|No)", pred_text, re.IGNORECASE)
    justification_match = re.search(r"Justification:\s*(.*)", pred_text, re.IGNORECASE | re.DOTALL)

    pred_answer = answer_match.group(1).capitalize() if answer_match else None
    pred_justification = justification_match.group(1).strip() if justification_match else ""

    if pred_answer is None or pred_answer.lower() != correct_answer.lower():
        return {"acc": 0.0, "strict_acc": 0.0}

    # Answer is correct
    if not ref_justification:
        # No reference justification to verify against
        return {"acc": 1.0, "strict_acc": 1.0}

    is_just_ok = _call_llm_judge(
        question=doc["question"],
        evidence=doc["evidence"],
        correct_answer=correct_answer,
        pred_justification=pred_justification,
    )
    return {"acc": 1.0, "strict_acc": 1.0 if is_just_ok else 0.0}
