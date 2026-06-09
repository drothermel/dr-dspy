from __future__ import annotations

import types
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

from dspy.task_spec.annotation_format import get_type_name

if TYPE_CHECKING:
    from dspy.task_spec.task_spec import TaskSpec


def validate_task_inputs_from_spec(task_spec: TaskSpec, inputs: dict[str, Any]) -> dict[str, Any]:
    validated = dict(inputs)
    input_fields = task_spec.input_fields
    for field_name, field in input_fields.items():
        if field_name not in validated and field.has_default:
            validated[field_name] = field.default
    unknown = sorted(key for key in validated if key not in input_fields)
    if unknown:
        raise ValueError(
            f"Unknown task input field(s) {unknown} for task spec {task_spec.name!r}. "
            f"Expected input fields: {list(input_fields.keys())}."
        )
    missing = sorted(
        field_name
        for field_name, field in input_fields.items()
        if field_name not in inputs and not field.has_default and not _annotation_allows_none(field.type_)
    )
    if missing:
        raise ValueError(
            f"Missing required task input field(s) {missing} for task spec {task_spec.name!r}. "
            f"Provided fields: {sorted(inputs.keys())}."
        )
    for field_name, field in input_fields.items():
        if field_name not in validated:
            continue
        value = validated[field_name]
        if field.is_type_undefined:
            continue
        if value is None:
            if not _annotation_allows_none(field.type_):
                raise ValueError(
                    f"Type mismatch for task input field {field_name!r}: expected {get_type_name(field.type_)}, "
                    f"got incompatible value None."
                )
            continue
        if not _is_value_compatible_with_type(value, field.type_):
            raise ValueError(
                f"Type mismatch for task input field {field_name!r}: expected {get_type_name(field.type_)}, "
                f"got incompatible value {value!r}."
            )
    return validated


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
