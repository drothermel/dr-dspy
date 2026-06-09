from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.primitives.example import Example

if TYPE_CHECKING:
    from dspy.core.types.call_options import ModuleCallOptions
    from dspy.primitives.module import Module
    from dspy.primitives.prediction import Prediction
    from dspy.runtime.run_context import RunContext


async def run_program_with_trace(
    program: Module,
    example: Example | dict[str, Any],
    run: RunContext,
    *,
    options: ModuleCallOptions | None = None,
) -> tuple[Prediction, list[Any]]:
    item_run = run.fork(optimization_trace=[], call_log=[])
    inputs = example.as_inputs() if isinstance(example, Example) else example
    prediction = await program(**inputs, run=item_run, options=options)
    trace = list(item_run.optimization_trace)
    return prediction, trace


def trace_to_demos(trace: list[Any], predictor2name: dict[int, str]) -> dict[str, list[Example]]:
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
