from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from dspy.core.types.call_options import PredictOptions, ensure_predict_options_built
from dspy.history.discovery import is_agent_history_type
from dspy.task_spec import validate_task_inputs_from_spec
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
    validated = validate_task_inputs_from_spec(task_spec, inputs)
    for field_name, field in task_spec.input_fields.items():
        if field_name not in validated:
            continue
        value = validated[field_name]
        if value is None or field.is_type_undefined:
            continue
        if is_agent_history_type(field.type_):
            validated[field_name] = TypeAdapter(field.type_).validate_python(value)
    return validated


def resolve_predict_options(options: PredictOptions | None) -> PredictOptions:
    ensure_predict_options_built()
    return options or PredictOptions()
