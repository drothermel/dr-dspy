"""Token-based QA evaluation metrics.

Import from ``dspy.evaluate.metrics``. Example/Prediction metrics such as
``answer_exact_match`` require ``example.answer`` and ``pred.answer``; scores
are in the 0-1 range.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy.primitives import Example, Prediction

__all__ = [
    "answer_exact_match",
    "em_score",
    "hotpot_f1_score",
    "max_em_score",
    "max_hotpot_f1_score",
    "max_token_f1_score",
    "normalize_text",
    "token_f1_score",
]

_ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = set(string.punctuation)


def _lower(text: str) -> str:
    return text.lower()


def _remove_articles(text: str) -> str:
    return _ARTICLE_PATTERN.sub(" ", text)


def _remove_punc(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCTUATION)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def normalize_text(s: str) -> str:
    normalized = unicodedata.normalize("NFD", s)
    return _white_space_fix(_remove_articles(_remove_punc(_lower(normalized))))


def em_score(*, prediction: str, ground_truth: str) -> bool:
    return normalize_text(s=prediction) == normalize_text(s=ground_truth)


def token_f1_score(*, prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_text(s=prediction).split()
    ground_truth_tokens = normalize_text(s=ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def hotpot_f1_score(*, prediction: str, ground_truth: str) -> float:
    normalized_prediction = normalize_text(s=prediction)
    normalized_ground_truth = normalize_text(s=ground_truth)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def max_em_score(*, prediction: str, answers_list: list[str]) -> bool:
    if not isinstance(answers_list, list):
        raise ValueError(f"`answers_list` must be a list, got {type(answers_list)}")
    return max(em_score(prediction=prediction, ground_truth=ans) for ans in answers_list)


def max_token_f1_score(*, prediction: str, answers_list: list[str]) -> float:
    if not isinstance(answers_list, list):
        raise ValueError(f"`answers_list` must be a list, got {type(answers_list)}")
    return max(token_f1_score(prediction=prediction, ground_truth=ans) for ans in answers_list)


def max_hotpot_f1_score(*, prediction: str, answers_list: list[str]) -> float:
    if not isinstance(answers_list, list):
        raise ValueError(f"`answers_list` must be a list, got {type(answers_list)}")
    return max(hotpot_f1_score(prediction=prediction, ground_truth=ans) for ans in answers_list)


def _answer_match(*, prediction: str, answers: list[str], frac: float = 1.0) -> bool:
    if frac >= 1.0:
        return max_em_score(prediction=prediction, answers_list=answers)
    return max_token_f1_score(prediction=prediction, answers_list=answers) >= frac


def answer_exact_match(
    example: Example,
    pred: Prediction,
    trace: Any = None,
    *,
    frac: float = 1.0,
) -> bool:
    del trace
    if isinstance(example.answer, str):
        return _answer_match(prediction=pred.answer, answers=[example.answer], frac=frac)
    if isinstance(example.answer, list):
        return _answer_match(prediction=pred.answer, answers=example.answer, frac=frac)
    raise ValueError(f"Invalid answer type: {type(example.answer)}")
