from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from dspy.evaluate.evaluator import Evaluate
from dspy.runtime.async_parallel import resolve_max_errors
from dspy.runtime.config import CallSite
from dspy.runtime.transparency import resolve_adapter
from dspy.teleprompt.trace_helpers import run_program_with_trace, trace_to_demos

if TYPE_CHECKING:
    from dspy.primitives.example import Example
    from dspy.runtime.run_context import RunContext


def make_optimizer_evaluator(
    run: RunContext,
    *,
    devset: list[Example],
    metric,
    max_concurrency: int | None,
    max_errors: int | None,
    **kwargs: Any,
):
    effective_max_errors = resolve_max_errors(max_errors, run)
    return Evaluate(
        devset=devset,
        metric=metric,
        max_concurrency=max_concurrency,
        max_errors=effective_max_errors,
        **kwargs,
    )


def optimizer_run_context(
    run: RunContext,
    *,
    lm,
    adapter=None,
    phase: str,
    lm_role: str,
    **extra,
) -> RunContext:
    resolved_adapter, _notes = resolve_adapter(adapter or run.adapter)
    return run.fork(lm=lm, adapter=resolved_adapter, **extra)


@contextmanager
def optimizer_lm_context(run: RunContext, *, lm, adapter=None, phase: str, lm_role: str, **extra):
    module = extra.pop("module", "optimizer")
    call_site = CallSite(module=module, phase=phase, lm_role=lm_role)
    child = optimizer_run_context(run, lm=lm, adapter=adapter, phase=phase, lm_role=lm_role, **extra).fork(
        call_site=call_site
    )
    yield child


__all__ = [
    "make_optimizer_evaluator",
    "optimizer_lm_context",
    "optimizer_run_context",
    "run_program_with_trace",
    "trace_to_demos",
]
