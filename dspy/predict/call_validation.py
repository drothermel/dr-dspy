from __future__ import annotations

import types
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

from pydantic_core import PydanticUndefined

from dspy.core.types.call_options import PredictOptions
from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos
from dspy.task_spec.task_spec import TaskSpec  # noqa: TC001 — runtime isinstance checks
from dspy.utils.constants import IS_TYPE_UNDEFINED

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
    input_fields = task_spec_input_field_infos(task_spec)
    validated = dict(inputs)
    for field_name, field_info in input_fields.items():
        if field_name not in validated and field_info.default is not PydanticUndefined:
            validated[field_name] = field_info.default
    unknown = sorted(key for key in validated if key not in input_fields)
    if unknown:
        raise ValueError(
            f"Unknown task input field(s) {unknown} for task spec {task_spec.name!r}. "
            f"Expected input fields: {list(input_fields.keys())}."
        )
    missing = sorted(
        field_name
        for field_name, field_info in input_fields.items()
        if field_name not in inputs
        and field_info.default is PydanticUndefined
        and not _annotation_allows_none(field_info.annotation)
    )
    if missing:
        raise ValueError(
            f"Missing required task input field(s) {missing} for task spec {task_spec.name!r}. "
            f"Provided fields: {sorted(inputs.keys())}."
        )
    for field_name, field_info in input_fields.items():
        if field_name not in validated:
            continue
        value = validated[field_name]
        expected_type = field_info.annotation
        json_schema_extra = field_info.json_schema_extra
        schema_extra = cast("dict[str, Any]", json_schema_extra) if isinstance(json_schema_extra, dict) else {}
        if value is None or schema_extra.get(IS_TYPE_UNDEFINED):
            continue
        if expected_type is None or not isinstance(expected_type, type):
            continue
        if not _is_value_compatible_with_type(value, expected_type):
            raise ValueError(
                f"Type mismatch for task input field {field_name!r}: expected {_get_type_name(expected_type)}, "
                f"got incompatible value {value!r}."
            )
    return validated


def resolve_predict_options(options: PredictOptions | None) -> PredictOptions:
    return options or PredictOptions()


def _get_type_name(type_annotation: Any) -> str:
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)
    if origin is None:
        if hasattr(type_annotation, "__name__"):
            return type_annotation.__name__
        return str(type_annotation)
    if origin is Literal:
        literal_values = ", ".join(repr(arg) for arg in args)
        return f"Literal[{literal_values}]"
    if args:
        args_str = ", ".join("..." if arg is ... else _get_type_name(arg) for arg in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]"
    return getattr(origin, "__name__", str(origin))


def _annotation_allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is None or annotation is type(None):
        return True
    if origin is Annotated:
        return bool(args) and _annotation_allows_none(args[0])
    if origin is Union or origin is types.UnionType:
        return any(_annotation_allows_none(arg) for arg in args)
    return False


def _is_value_compatible_with_type(value: Any, expected: type) -> bool:
    if expected is str and isinstance(value, list) and all(isinstance(item, str) for item in value):
        return True
    return _check_type(value, expected)


def _check_type(value: Any, expected: type) -> bool:
    if expected is Any:
        return True
    origin = get_origin(expected)
    args = get_args(expected)
    if origin is Union or origin is types.UnionType:
        return any(_check_type(value, arg) for arg in args)
    if origin is Literal:
        return value in args
    if origin is list:
        if not isinstance(value, list):
            return False
        if args:
            return all(_check_type(item, args[0]) for item in value)
        return True
    if origin is dict:
        if not isinstance(value, dict):
            return False
        if args:
            key_type, val_type = args
            return all(_check_type(k, key_type) and _check_type(v, val_type) for k, v in value.items())
        return True
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                return all(_check_type(item, args[0]) for item in value)
            if len(value) != len(args):
                return False
            return all(_check_type(item, arg) for item, arg in zip(value, args, strict=False))
        return True
    if origin is set or origin is frozenset:
        if not isinstance(value, origin):
            return False
        if args:
            return all(_check_type(item, args[0]) for item in value)
        return True
    if isinstance(expected, type):
        return isinstance(value, expected)
    return False
