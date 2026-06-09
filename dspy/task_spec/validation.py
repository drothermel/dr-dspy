"""Runtime task input validation.

Special case preserved intentionally: ``str`` fields accept ``list[str]`` values for
input formatting (multi-line text joined before the LM call).
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Annotated, Any, Union, get_args, get_origin

from pydantic import TypeAdapter, ValidationError

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
    try:
        TypeAdapter(expected).validate_python(value)
    except ValidationError:
        return False
    return True
