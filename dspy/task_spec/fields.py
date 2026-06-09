from __future__ import annotations

import enum
import inspect
import json
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, cast, get_args, get_origin

import pydantic
from pydantic import BaseModel, ConfigDict

from dspy.adapters.types.code import Code
from dspy.adapters.types.reasoning import Reasoning
from dspy.task_spec.field_spec import FieldRole, FieldSpec

if TYPE_CHECKING:
    from dspy.task_spec.task_spec import TaskSpec


class FieldBinding(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    field: FieldSpec


def field_bindings(task_spec: TaskSpec, *, role: FieldRole) -> tuple[FieldBinding, ...]:
    fields = task_spec.input_fields if role == FieldRole.INPUT else task_spec.output_fields
    return tuple(FieldBinding(name=name, field=field) for name, field in fields.items())


def format_field_value(field: FieldSpec, value: object, assume_text: bool = True) -> str | dict[str, str]:
    from dspy.adapters.utils.json import serialize_for_json

    string_value = None
    if isinstance(value, list) and field.type_ is str:
        string_value = _format_input_list_field_value(value)
    else:
        jsonable_value = serialize_for_json(value)
        if isinstance(jsonable_value, (dict, list)):
            string_value = json.dumps(jsonable_value, ensure_ascii=False)
        else:
            string_value = str(jsonable_value)
    if assume_text:
        return string_value
    return {"type": "text", "text": string_value}


def translate_field_type(field: FieldSpec) -> str:
    field_type = field.type_
    if field.role == FieldRole.INPUT or field_type is str or field_type is Reasoning:
        desc = ""
    elif field_type is bool:
        desc = "must be True or False"
    elif field_type in (int, float):
        desc = f"must be a single {field_type.__name__} value"
    elif _annotation_is_subclass(annotation=field_type, expected_base=enum.Enum):
        enum_type = cast("type[enum.Enum]", field_type)
        enum_vals = "; ".join(str(member.value) for member in enum_type)
        desc = f"must be one of: {enum_vals}"
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        desc = f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in field_type.__args__])}"
    elif (
        _annotation_is_subclass(annotation=field_type, expected_base=Code)
        and cast("type[Code]", field_type).description()
    ):
        desc = ""
    else:
        desc = f"must adhere to the JSON schema: {json.dumps(_get_json_schema(field_type), ensure_ascii=False)}"
    desc = " " * 8 + f"# note: the value you produce {desc}" if desc else ""
    return f"{{{field.name}}}{desc}"


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
                    f"Type mismatch for task input field {field_name!r}: expected {_get_type_name(field.type_)}, "
                    f"got incompatible value None."
                )
            continue
        if not _is_value_compatible_with_type(value, field.type_):
            raise ValueError(
                f"Type mismatch for task input field {field_name!r}: expected {_get_type_name(field.type_)}, "
                f"got incompatible value {value!r}."
            )
    return validated


def get_annotation_name(annotation: object) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return cast("str", annotation.__name__)
        return str(annotation)
    if origin is Literal:
        args_str = ", ".join(
            _quoted_string_for_literal_type_annotation(a) if isinstance(a, str) else get_annotation_name(a)
            for a in args
        )
        return f"{get_annotation_name(origin)}[{args_str}]"
    args_str = ", ".join(get_annotation_name(a) for a in args)
    return f"{get_annotation_name(origin)}[{args_str}]"


def _annotation_is_subclass(annotation: object, expected_base: type) -> bool:
    try:
        return inspect.isclass(annotation) and issubclass(annotation, expected_base)
    except TypeError:
        return False


def _get_json_schema(field_type: object) -> object:
    def move_type_to_front(d: object) -> object:
        if isinstance(d, Mapping):
            return {
                k: move_type_to_front(v) for k, v in sorted(d.items(), key=lambda item: (item[0] != "type", item[0]))
            }
        if isinstance(d, list):
            return [move_type_to_front(item) for item in d]
        return d

    schema = pydantic.TypeAdapter(field_type).json_schema()
    return move_type_to_front(schema)


def _format_input_list_field_value(value: list[Any]) -> str:
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return _format_blob(str(value[0]))
    return "\n".join([f"[{idx + 1}] {_format_blob(str(txt))}" for idx, txt in enumerate(value)])


def _format_blob(blob: str) -> str:
    if "\n" not in blob and "«" not in blob and ("»" not in blob):
        return f"«{blob}»"
    modified_blob = blob.replace("\n", "\n    ")
    return f"«««\n    {modified_blob}\n»»»"


def _quoted_string_for_literal_type_annotation(s: str) -> str:
    has_single = "'" in s
    has_double = '"' in s
    if has_single and (not has_double):
        return f'"{s}"'
    if has_double and (not has_single):
        return f"'{s}'"
    if has_single and has_double:
        escaped = s.replace("'", "\\'")
        return f"'{escaped}'"
    return f"'{s}'"


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
