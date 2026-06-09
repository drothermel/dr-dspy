"""Shared compile-spine helpers for teleprompters and integrations."""

from dspy.teleprompt.core.demos import UnknownPredictorInTraceError, trace_to_demos
from dspy.teleprompt.core.evaluator import (
    make_optimizer_evaluator,
    optimizer_lm_context,
    optimizer_run_context,
)
from dspy.teleprompt.core.split import split_trainset_holdout
from dspy.teleprompt.core.trace_collection import collect_trace_data

__all__ = [
    "UnknownPredictorInTraceError",
    "collect_trace_data",
    "make_optimizer_evaluator",
    "optimizer_lm_context",
    "optimizer_run_context",
    "split_trainset_holdout",
    "trace_to_demos",
]
