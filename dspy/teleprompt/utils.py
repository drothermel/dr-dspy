from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from dspy.compile.resolve import resolve_adapter
from dspy.primitives.example import Example
from dspy.utils.transparency import CallSite

if TYPE_CHECKING:
    from dspy.core.types.call_options import ModuleCallOptions
    from dspy.primitives.module import Module
    from dspy.primitives.prediction import Prediction
    from dspy.runtime.run_context import RunContext


def resolve_max_errors(optimizer_max_errors: int | None, run: RunContext) -> int:
    from dspy.utils.async_parallel import resolve_max_errors as _resolve_max_errors

    return _resolve_max_errors(optimizer_max_errors, run)


def make_optimizer_evaluator(
    run: RunContext,
    *,
    devset: list[Example],
    metric,
    max_concurrency: int | None,
    max_errors: int | None,
    **kwargs: Any,
):
    from dspy.evaluate.evaluate import Evaluate
    from dspy.utils.async_parallel import resolve_max_errors as _resolve_max_errors

    effective_max_errors = _resolve_max_errors(max_errors, run)
    return Evaluate(
        devset=devset,
        metric=metric,
        max_concurrency=max_concurrency,
        max_errors=effective_max_errors,
        **kwargs,
    )


async def run_program_with_trace(
    program: Module,
    example: Example | dict[str, Any],
    run: RunContext,
    *,
    options: ModuleCallOptions | None = None,
) -> tuple[Prediction, list]:
    item_run = run.fork(optimization_trace=[], call_log=[])
    inputs = example.as_inputs() if isinstance(example, Example) else example
    prediction = await program(**inputs, run=item_run, options=options)
    trace = list(item_run.optimization_trace)
    return prediction, trace


def trace_to_demos(trace: list, predictor2name: dict[int, str]) -> dict[str, list[Example]]:
    name2traces: dict[str, list[Example]] = {}
    for step in trace:
        predictor, inputs, outputs = step
        demo = Example.from_record({"augmented": True, **inputs, **outputs})
        try:
            predictor_name = predictor2name[id(predictor)]
        except KeyError:
            continue
        name2traces[predictor_name] = name2traces.get(predictor_name, [])
        name2traces[predictor_name].append(demo)
    return name2traces


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
