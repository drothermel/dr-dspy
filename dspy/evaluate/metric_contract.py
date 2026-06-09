"""Evaluation and optimizer metric type contracts."""

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
