from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.teleprompt.core.trace_collection import collect_trace_data, make_trace_collection_evaluator

if TYPE_CHECKING:
    from dspy.primitives import Example, Module
    from dspy.runtime.optimization_trace import TraceData
    from dspy.runtime.run_context import RunContext
    from dspy.teleprompt.metrics import OptimizerMetric

TraceGrid = list[list[list["TraceData"]]]


async def collect_teacher_trace_grid(
    *,
    teachers: list[Module],
    subsample: list[Example],
    num_samples_per_input: int,
    run: RunContext,
    metric: OptimizerMetric | None,
    max_concurrency: int,
    failure_score: float,
    format_failure_score: float,
) -> TraceGrid:
    trace_data: TraceGrid = [[[] for _ in range(len(teachers))] for _ in range(len(subsample))]
    subsample_repeated = [example for _ in range(num_samples_per_input) for example in subsample]
    trace_evaluator = make_trace_collection_evaluator(
        run,
        dataset=subsample_repeated,
        max_concurrency=max_concurrency,
        failure_score=failure_score,
    )
    for tind, teacher in enumerate(teachers):
        round_data = await collect_trace_data(
            program=teacher,
            dataset=subsample_repeated,
            run=run,
            evaluator=trace_evaluator,
            metric=metric,
            raise_on_error=False,
            capture_parse_failures=True,
            failure_score=failure_score,
            format_failure_score=format_failure_score,
            log_format_failures=True,
        )
        for data_dict in round_data:
            example_ind_in_subsample = data_dict["example_ind"] % len(subsample)
            data_dict["example_ind"] = example_ind_in_subsample
            trace_data[example_ind_in_subsample][tind].append(data_dict)
    return trace_data
