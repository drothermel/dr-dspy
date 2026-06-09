from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from dspy.runtime.optimization_trace import FailedPrediction, TraceCapturingModule, TraceData
from dspy.teleprompt.core.evaluator import make_optimizer_evaluator

if TYPE_CHECKING:
    from dspy.primitives import Example, Module
    from dspy.runtime.run_context import RunContext

logger = logging.getLogger(__name__)


async def collect_trace_data(
    program: Module,
    dataset: list[Example],
    run: RunContext,
    metric: Callable | None = None,
    max_concurrency: int | None = None,
    raise_on_error: bool = True,
    capture_parse_failures: bool = False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
    callback_metadata: dict[str, Any] | None = None,
) -> list[TraceData]:
    evaluator = make_optimizer_evaluator(
        run,
        devset=dataset,
        metric=None,
        max_concurrency=max_concurrency,
        max_errors=len(dataset) * 10,
        display_progress=True,
        provide_traceback=False,
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            reward = prediction.format_reward if prediction.format_reward is not None else format_failure_score
            if reward < 0.0 or reward > 1.0:
                return failure_score
            return reward
        return metric(example, prediction, trace) if metric else True

    capturing_program = TraceCapturingModule(
        program,
        capture_parse_failures=capture_parse_failures,
        failure_score=failure_score,
        format_failure_score=format_failure_score,
        log_format_failures=log_format_failures,
    )
    results = (
        await evaluator(capturing_program, run=run, metric=wrapped_metric, callback_metadata=callback_metadata)
    ).results
    data: list[TraceData] = []
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, trace = prediction
        except ValueError:
            logger.warning(
                "Failed to unpack prediction and trace. This is likely due to the LLM response not following dspy formatting."
            )
            if raise_on_error:
                raise
            continue
        entry: TraceData = {
            "example": example,
            "prediction": prediction,
            "trace": trace,
            "example_ind": example_ind,
            "score": score if metric else None,
        }
        data.append(entry)
    return data
