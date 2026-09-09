from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0
    precision = shared / len(pred_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return float(normalize_answer(prediction) == "")
    normalized_prediction = normalize_answer(prediction)
    return max(float(normalized_prediction == normalize_answer(answer)) for answer in gold_answers)


def best_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return float(normalize_answer(prediction) == "")
    return max(token_f1(prediction, answer) for answer in gold_answers)


def contains_normalized_answer(text: str, gold_answers: list[str]) -> bool:
    text_tokens = normalize_answer(text).split()
    for answer in gold_answers:
        answer_tokens = normalize_answer(answer).split()
        answer_length = len(answer_tokens)
        if not answer_tokens or answer_length > len(text_tokens):
            continue
        if any(
            text_tokens[start : start + answer_length] == answer_tokens
            for start in range(len(text_tokens) - answer_length + 1)
        ):
            return True
    return False
