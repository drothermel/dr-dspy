import logging
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, TypedDict

from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError

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
    metric: Callable | None = None,
    num_threads: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
    callback_metadata: dict[str, Any] | None = None,
) -> list[TraceData]:
    _ = capture_failed_parses
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=False,
        max_errors=len(dataset) * 10,
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return metric(example, prediction, trace) if metric else True

    original_aforward = object.__getattribute__(program, "aforward")

    async def patched_aforward(program_to_use: Module, **kwargs):
        with settings.context(trace=[]):
            try:
                return (await original_aforward(**kwargs), settings.trace.copy())
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_task_spec = e.task_spec
                failed_inputs = kwargs
                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_task_spec.output_fields.keys())
                from dspy.teleprompt.utils import get_task_spec

                found_pred = None
                for pred in program_to_use.predictors():
                    if get_task_spec(pred).equals(failed_task_spec):
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed task spec: {failed_task_spec}")
                trace = settings.trace.copy()
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

    program.aforward = MethodType(patched_aforward, program)
    try:
        results = (await evaluator(program, metric=wrapped_metric, callback_metadata=callback_metadata)).results
    finally:
        program.aforward = original_aforward
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
