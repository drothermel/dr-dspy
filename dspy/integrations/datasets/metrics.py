"""Benchmark OptimizerMetric callables paired with integration datasets."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Protocol, cast

from dspy.evaluate.metrics import max_hotpot_f1_score

if TYPE_CHECKING:
    from dspy.primitives import Example, Prediction

__all__ = ["gsm8k_metric", "hotpotqa_metric", "math_metric", "parse_integer_answer"]


def parse_integer_answer(answer: str, only_first_line: bool = True) -> int:
    parsed_answer = 0
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]
        answer_token = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer_token = answer_token.split(".")[0]
        answer_digits = "".join(c for c in answer_token if c.isdigit())
        parsed_answer = int(answer_digits)
    except (ValueError, IndexError):
        parsed_answer = 0
    return parsed_answer


def gsm8k_metric(example: Example, pred: Prediction, trace: Any = None) -> bool:
    del trace
    return int(parse_integer_answer(str(example.answer))) == int(parse_integer_answer(str(pred.answer)))


def hotpotqa_metric(example: Example, pred: Prediction, trace: Any = None) -> float:
    del trace
    gold = example.answer
    answers = [str(item) for item in gold] if isinstance(gold, list) else [str(gold)]
    return max_hotpot_f1_score(prediction=str(pred.answer), answers_list=answers)


class MathEquivalenceModule(Protocol):
    def is_equiv(self, left: object, right: object) -> bool: ...


def math_metric(example: Example, pred: Prediction, trace: Any = None) -> bool:
    del trace
    try:
        math_equivalence = cast("MathEquivalenceModule", importlib.import_module("math_equivalence"))
    except ImportError as err:
        raise ImportError("MATH's metric requires `pip install git+https://github.com/hendrycks/math.git`") from err
    return math_equivalence.is_equiv(example.answer, pred.answer)


def _register_default_metrics() -> None:
    from dspy.integrations.datasets.gsm8k import GSM8K
    from dspy.integrations.datasets.hotpotqa import HotPotQA
    from dspy.integrations.datasets.math import MATH

    GSM8K.default_metric = gsm8k_metric
    HotPotQA.default_metric = hotpotqa_metric
    MATH.default_metric = math_metric


_register_default_metrics()
