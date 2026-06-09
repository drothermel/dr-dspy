import logging
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, TypedDict

from dspy.core.types.call_options import ModuleCallOptions
from dspy.errors import AdapterParseError
from dspy.primitives import Example, Module, Prediction
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.task_spec_context import get_task_spec
from dspy.teleprompt.utils import make_optimizer_evaluator

logger = logging.getLogger(__name__)


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None


async def bootstrap_trace_data(
    program: Module,
    dataset: list[Example],
    run: RunContext,
    metric: Callable | None = None,
    max_concurrency: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
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

    if capture_failed_parses:
        original_aforward_impl = object.__getattribute__(program, "_aforward_impl")

        async def patched_aforward_impl(
            program_to_use: Module,
            *,
            run: RunContext,
            options: ModuleCallOptions | None = None,
            **kwargs,
        ):
            item_run = run.fork(optimization_trace=[], call_log=[])
            try:
                return (
                    await original_aforward_impl(run=item_run, options=options, **kwargs),
                    list(item_run.optimization_trace),
                )
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_task_spec = e.task_spec
                failed_inputs = kwargs
                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_task_spec.output_fields.keys())
                found_pred = None
                for pred in program_to_use.predictors():
                    if get_task_spec(pred) == failed_task_spec:
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed task spec: {failed_task_spec}")
                trace = list(item_run.optimization_trace)
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score) * (len(present) / len(expected)),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)
                trace.append((found_pred, failed_inputs, failed_pred))
                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following the adapter's formatting."
                    )
                return (failed_pred, trace)

        program_any = program
        program_any._aforward_impl = MethodType(patched_aforward_impl, program)
        try:
            results = (
                await evaluator(program, run=run, metric=wrapped_metric, callback_metadata=callback_metadata)
            ).results
        finally:
            program_any._aforward_impl = original_aforward_impl
    else:
        results = (
            await evaluator(program, run=run, metric=wrapped_metric, callback_metadata=callback_metadata)
        ).results
    data = []
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
