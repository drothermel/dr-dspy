from __future__ import annotations

from typing import Any

from dspy.core.types.call_options import PredictOptions
from dspy.task_spec.fields import validate_task_inputs_from_spec
from dspy.task_spec.task_spec import TaskSpec  # noqa: TC001 — runtime isinstance checks

RESERVED_PREDICT_INPUT_KEYS = frozenset(
    {"run", "options", "lm", "config", "demos", "task_spec", "_trace", "prediction"}
)


def reject_reserved_predict_inputs(inputs: dict[str, Any]) -> None:
    reserved = sorted(key for key in inputs if key in RESERVED_PREDICT_INPUT_KEYS)
    if reserved:
        raise ValueError(
            f"Reserved keyword(s) {reserved} must not be passed as task inputs. "
            "Use run= for RunContext and options=PredictOptions(...) for lm, config, demos, task_spec, trace, and prediction."
        )


def validate_task_inputs(task_spec: TaskSpec, inputs: dict[str, Any]) -> dict[str, Any]:
    reject_reserved_predict_inputs(inputs)
    return validate_task_inputs_from_spec(task_spec, inputs)


def resolve_predict_options(options: PredictOptions | None) -> PredictOptions:
    return options or PredictOptions()
