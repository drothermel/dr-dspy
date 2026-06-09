"""Optimizer metric contract for teleprompters.

Standard optimizers accept a sync or async callable ``(example, prediction, trace)``
returning bool, a numeric score in ``[0, 1]``, or a ``Prediction`` with ``score``.
``Module`` metrics are also supported; a non-``None`` ``trace`` enables threshold mode.

``invoke_metric`` normalizes all return shapes to a 0-1 float. GEPA uses the
separate five-argument ``GEPAFeedbackMetric`` protocol in ``dspy.teleprompt.gepa``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from dspy.primitives import Example, Module, Prediction

MetricScore = bool | float | int | Prediction
SyncOptimizerMetric = Callable[[Example, Prediction, list | None], MetricScore]
AsyncOptimizerMetric = Callable[[Example, Prediction, list | None], Awaitable[MetricScore]]
OptimizerMetric = SyncOptimizerMetric | AsyncOptimizerMetric | Module

__all__ = [
    "AsyncOptimizerMetric",
    "MetricScore",
    "OptimizerMetric",
    "SyncOptimizerMetric",
]
