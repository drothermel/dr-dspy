from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from dspy.errors import AdapterParseError
from dspy.primitives import Example, Module, Prediction
from dspy.runtime.run_fork import fork_worker_run

if TYPE_CHECKING:
    from dspy.runtime.call_options import ModuleCallOptions
    from dspy.runtime.run_context import RunContext
    from dspy.task_spec import TaskSpec

logger = logging.getLogger(__name__)


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


OptimizationTraceStep = tuple[Any, dict[str, Any], Prediction | FailedPrediction]


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction | FailedPrediction
    trace: list[tuple[Any, dict[str, Any], Prediction | FailedPrediction]]
    score: float | None


def _find_predictor_for_task_spec(program: Module, task_spec: TaskSpec) -> Any:
    for pred in program.predictors():
        if pred.task_spec == task_spec:
            return pred
    raise ValueError(f"Failed to find the predictor for the failed task spec: {task_spec}")


def _failed_prediction_from_parse_error(
    program: Module,
    item_run: RunContext,
    error: AdapterParseError,
    inputs: dict[str, Any],
    *,
    format_failure_score: float,
    failure_score: float,
    log_format_failures: bool,
) -> tuple[FailedPrediction, list[OptimizationTraceStep]]:
    completion_str = error.lm_response
    parsed_result = error.parsed_result
    failed_task_spec = error.task_spec
    present = list(parsed_result.keys()) if parsed_result else None
    expected = list(failed_task_spec.output_fields.keys())
    found_pred = _find_predictor_for_task_spec(program, failed_task_spec)
    trace: list[OptimizationTraceStep] = list(item_run.optimization_trace)
    if present:
        failed_pred = FailedPrediction(
            completion_text=completion_str,
            format_reward=format_failure_score
            + (failure_score - format_failure_score) * (len(present) / len(expected)),
        )
    else:
        failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)
    trace.append((found_pred, inputs, failed_pred))
    if log_format_failures:
        logger.warning(
            "Failed to parse output for example. This is likely due to the LLM response not following the adapter's formatting."
        )
    return failed_pred, trace


async def _capture_forward(
    program: Module,
    *,
    run: RunContext,
    options: ModuleCallOptions | None = None,
    capture_parse_failures: bool = False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
    **kwargs: Any,
) -> tuple[Prediction | FailedPrediction, list[OptimizationTraceStep]]:
    item_run = fork_worker_run(run)
    try:
        prediction = await program.aforward(run=item_run, options=options, **kwargs)
        return prediction, list(item_run.optimization_trace)
    except AdapterParseError as error:
        if not capture_parse_failures:
            raise
        return _failed_prediction_from_parse_error(
            program,
            item_run,
            error,
            kwargs,
            format_failure_score=format_failure_score,
            failure_score=failure_score,
            log_format_failures=log_format_failures,
        )


async def run_with_trace(
    program: Module,
    example: Example | dict[str, Any],
    run: RunContext,
    *,
    options: ModuleCallOptions | None = None,
    capture_parse_failures: bool = False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
) -> tuple[Prediction | FailedPrediction, list[OptimizationTraceStep]]:
    inputs = example.as_inputs() if isinstance(example, Example) else example
    return await _capture_forward(
        program,
        run=run,
        options=options,
        capture_parse_failures=capture_parse_failures,
        failure_score=failure_score,
        format_failure_score=format_failure_score,
        log_format_failures=log_format_failures,
        **inputs,
    )


class TraceCapturingModule(Module):
    """Wraps a program so ``_aforward_impl`` returns ``(prediction, trace)`` for batch eval."""

    def __init__(
        self,
        inner: Module,
        *,
        capture_parse_failures: bool = False,
        failure_score: float = 0,
        format_failure_score: float = -1,
        log_format_failures: bool = False,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._capture_parse_failures = capture_parse_failures
        self._failure_score = failure_score
        self._format_failure_score = format_failure_score
        self._log_format_failures = log_format_failures

    def predictors(self) -> list[Any]:
        return self._inner.predictors()

    def named_predictors(self) -> list[tuple[str, Any]]:
        return self._inner.named_predictors()

    def set_lm(self, lm: Any) -> None:
        self._inner.set_lm(lm)

    def optional_lm(self) -> Any:
        return self._inner.optional_lm()

    def deepcopy(self) -> TraceCapturingModule:
        return TraceCapturingModule(
            self._inner.deepcopy(),
            capture_parse_failures=self._capture_parse_failures,
            failure_score=self._failure_score,
            format_failure_score=self._format_failure_score,
            log_format_failures=self._log_format_failures,
        )

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **kwargs: Any,
    ) -> tuple[Prediction | FailedPrediction, list[OptimizationTraceStep]]:
        return await _capture_forward(
            self._inner,
            run=run,
            options=options,
            capture_parse_failures=self._capture_parse_failures,
            failure_score=self._failure_score,
            format_failure_score=self._format_failure_score,
            log_format_failures=self._log_format_failures,
            **kwargs,
        )


__all__ = [
    "FailedPrediction",
    "OptimizationTraceStep",
    "TraceCapturingModule",
    "TraceData",
    "run_with_trace",
]
