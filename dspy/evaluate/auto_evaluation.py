"""LLM-judge evaluation metrics.

Import ``SemanticF1`` and ``CompleteAndGrounded`` from ``dspy.evaluate.auto_evaluation``.
These metrics read ``example.question``, ``example.response``, ``pred.response``, and
(for groundedness) ``pred.context``. Token metrics in ``dspy.evaluate.metrics`` use
``example.answer`` / ``pred.answer`` instead.
"""

from __future__ import annotations

from typing import Any

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives import Module, Prediction
from dspy.task_spec.framework.evaluate import (
    AnswerCompletenessTaskSpec,
    AnswerGroundednessTaskSpec,
    DecompositionalSemanticRecallPrecisionTaskSpec,
    SemanticRecallPrecisionTaskSpec,
)

__all__ = ["SemanticF1", "CompleteAndGrounded"]

QUESTION_FIELD = "question"
GROUND_TRUTH_FIELD = "response"
SYSTEM_RESPONSE_FIELD = "response"
RETRIEVED_CONTEXT_FIELD = "context"


def _require_str_field(obj: object, field: str, *, role: str) -> str:
    if not hasattr(obj, field):
        raise AttributeError(f"{role} missing required field {field!r}")
    value = getattr(obj, field)
    if not isinstance(value, str):
        raise ValueError(f"{role}.{field} must be a str, got {type(value).__name__}")
    return value


def harmonic_f1(*, precision: float, recall: float) -> float:
    precision, recall = (max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall)))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(Module):
    def __init__(self, threshold=0.66, decompositional=False) -> None:
        super().__init__()
        self.threshold = threshold
        if decompositional:
            self.module = ChainOfThought(DecompositionalSemanticRecallPrecisionTaskSpec())
        else:
            self.module = ChainOfThought(SemanticRecallPrecisionTaskSpec())

    async def _aforward_impl(
        self,
        *,
        run,
        options=None,
        example,
        pred,
        trace: Any = None,
        use_threshold: bool = False,
    ):
        del trace, options
        question = _require_str_field(example, QUESTION_FIELD, role="example")
        ground_truth = _require_str_field(example, GROUND_TRUTH_FIELD, role="example")
        system_response = _require_str_field(pred, SYSTEM_RESPONSE_FIELD, role="pred")
        scores = await self.module(
            question=question,
            ground_truth=ground_truth,
            system_response=system_response,
            run=run,
        )
        score = harmonic_f1(precision=scores.precision, recall=scores.recall)
        return Prediction(score=score if not use_threshold else score >= self.threshold)


class CompleteAndGrounded(Module):
    def __init__(self, threshold=0.66) -> None:
        super().__init__()
        self.threshold = threshold
        self.completeness_module = ChainOfThought(AnswerCompletenessTaskSpec())
        self.groundedness_module = ChainOfThought(AnswerGroundednessTaskSpec())

    async def _aforward_impl(
        self,
        *,
        run,
        options=None,
        example,
        pred,
        trace: Any = None,
        use_threshold: bool = False,
    ):
        del trace, options
        question = _require_str_field(example, QUESTION_FIELD, role="example")
        ground_truth = _require_str_field(example, GROUND_TRUTH_FIELD, role="example")
        system_response = _require_str_field(pred, SYSTEM_RESPONSE_FIELD, role="pred")
        retrieved_context = _require_str_field(pred, RETRIEVED_CONTEXT_FIELD, role="pred")
        completeness = await self.completeness_module(
            question=question,
            ground_truth=ground_truth,
            system_response=system_response,
            run=run,
        )
        groundedness = await self.groundedness_module(
            question=question,
            retrieved_context=retrieved_context,
            system_response=system_response,
            run=run,
        )
        score = harmonic_f1(precision=groundedness.groundedness, recall=completeness.completeness)
        return Prediction(score=score if not use_threshold else score >= self.threshold)
