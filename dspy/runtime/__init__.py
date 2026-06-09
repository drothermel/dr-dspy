"""Runtime execution context, telemetry, concurrency, and transparency.

Public spine for RunContext, telemetry config, bounded async execution,
usage tracking, transparency validation, and call-log inspection.
"""

from __future__ import annotations

import importlib
from typing import Any

from dspy.runtime.async_parallel import BoundedRunStats, resolve_max_concurrency, resolve_max_errors, run_bounded
from dspy.runtime.call_options import ModuleCallOptions
from dspy.runtime.callback import ACTIVE_CALL_ID, Callback, NoOpCallback, with_callbacks
from dspy.runtime.config import CallLogMode, CallSite, ExecutionConfig, TelemetryConfig, TransparencyMode
from dspy.runtime.inspect_call_log import pretty_print_call_log
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.runtime.transparency.report import enforce_compiled_call_transparency
from dspy.runtime.transparency.types import CompiledCall, TransparencyViolation
from dspy.runtime.usage_tracker import UsageTracker, track_usage

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FailedPrediction": ("dspy.runtime.optimization_trace", "FailedPrediction"),
    "Parallel": ("dspy.runtime.batch", "Parallel"),
    "TraceData": ("dspy.runtime.optimization_trace", "TraceData"),
    "run_with_trace": ("dspy.runtime.optimization_trace", "run_with_trace"),
}

__all__ = [
    "ACTIVE_CALL_ID",
    "Callback",
    "NoOpCallback",
    "Parallel",
    "BoundedRunStats",
    "CallLogMode",
    "CallSite",
    "CompiledCall",
    "ExecutionConfig",
    "ModuleCallOptions",
    "FailedPrediction",
    "RunContext",
    "TelemetryConfig",
    "TraceData",
    "TransparencyMode",
    "TransparencyViolation",
    "UsageTracker",
    "enforce_compiled_call_transparency",
    "pretty_print_call_log",
    "resolve_max_concurrency",
    "resolve_max_errors",
    "resolve_run",
    "run_bounded",
    "run_with_trace",
    "track_usage",
    "with_callbacks",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        obj = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
