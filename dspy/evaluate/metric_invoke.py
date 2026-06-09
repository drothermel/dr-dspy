"""Metric dispatch and score normalization for evaluation and optimizers.

Import ``invoke_metric``, ``call_metric``, and ``normalize_metric_score`` from
``dspy.evaluate.metric_invoke``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from dspy.primitives import Module, Prediction

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext

__all__ = ["call_metric", "invoke_metric", "normalize_metric_score"]


def normalize_metric_score(output: Any) -> float:
    if isinstance(output, bool):
        return 1.0 if output else 0.0
    if isinstance(output, (int, float)):
        return float(output)
    if isinstance(output, Prediction):
        if not hasattr(output, "score"):
            raise ValueError(
                "When `metric` returns a `dspy.primitives.prediction.Prediction`, it must contain a `score` field."
            )
        return normalize_metric_score(output.score)
    score_attr = getattr(output, "score", None)
    if score_attr is not None and isinstance(score_attr, (bool, int, float)):
        return normalize_metric_score(score_attr)
    raise TypeError(
        f"Metric returned unsupported type {type(output)!r}; expected bool, number, or Prediction with score."
    )


async def call_metric(
    metric: Any,
    *,
    example: Any,
    prediction: Any,
    trace: Any,
    run: RunContext,
) -> Any:
    if inspect.iscoroutinefunction(metric):
        return await metric(example, prediction, trace)
    if isinstance(metric, Module):
        # Optimizer/evaluate paths pass a trace list; module metrics interpret that as threshold mode.
        use_threshold = trace is not None
        return await metric(
            example=example,
            pred=prediction,
            trace=trace,
            use_threshold=use_threshold,
            run=run,
        )
    return metric(example, prediction, trace)


async def invoke_metric(
    metric: Any,
    *,
    example: Any,
    prediction: Any,
    trace: Any,
    run: RunContext,
) -> float:
    return normalize_metric_score(
        await call_metric(
            metric,
            example=example,
            prediction=prediction,
            trace=trace,
            run=run,
        )
    )
