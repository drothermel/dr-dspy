from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dspy.runtime.async_parallel import resolve_max_errors
from dspy.teleprompt.core.evaluator import make_optimizer_evaluator

if TYPE_CHECKING:
    from dspy.evaluate.evaluator import Evaluate
    from dspy.primitives import Example
    from dspy.runtime.run_context import RunContext
    from dspy.teleprompt.metrics import OptimizerMetric


@dataclass(frozen=True)
class GRPOValidationEvaluators:
    valset_only: Evaluate | None = None
    valset_plus_train: Evaluate | None = None
    train_as_val: Evaluate | None = None


def build_validation_evaluators(
    *,
    run: RunContext,
    trainset: list[Example],
    valset: list[Example] | None,
    use_train_as_val: bool,
    report_train_scores: bool,
    metric: OptimizerMetric | None,
    max_concurrency: int,
    max_errors: int | None,
    failure_score: float,
) -> GRPOValidationEvaluators:
    effective_max_errors = resolve_max_errors(max_errors, run)
    common_kwargs = {
        "max_concurrency": max_concurrency,
        "max_errors": effective_max_errors,
        "display_progress": True,
        "provide_traceback": False,
        "failure_score": failure_score,
    }
    valset_only = None
    valset_plus_train = None
    train_as_val = None
    if valset is not None:
        if report_train_scores:
            valset_plus_train = make_optimizer_evaluator(
                run,
                devset=valset + trainset,
                metric=metric,
                **common_kwargs,
            )
        else:
            valset_only = make_optimizer_evaluator(
                run,
                devset=valset,
                metric=metric,
                **common_kwargs,
            )
    elif report_train_scores and use_train_as_val:
        train_as_val = make_optimizer_evaluator(
            run,
            devset=trainset,
            metric=metric,
            **common_kwargs,
        )
    return GRPOValidationEvaluators(
        valset_only=valset_only,
        valset_plus_train=valset_plus_train,
        train_as_val=train_as_val,
    )
