"""Adapter-owned prompt rendering for task field values, types, and descriptions."""

from __future__ import annotations

import enum
import inspect
import json
from collections.abc import Mapping
from typing import Any, Literal, cast, get_args, get_origin

import pydantic

from dspy.adapters.types.code import Code
from dspy.adapters.types.field_type import extract_field_types_from_annotation
from dspy.adapters.types.reasoning import Reasoning
from dspy.serialization.json import to_jsonable
from dspy.task_spec.field_spec import FieldRole, FieldSpec


def format_field_value(field: FieldSpec, value: object, assume_text: bool = True) -> str | dict[str, str]:
    string_value = None
    if isinstance(value, list) and field.type_ is str:
        string_value = _format_input_list_field_value(value)
    else:
        jsonable_value = to_jsonable(value)
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


def get_field_spec_description_string(fields: dict[str, FieldSpec]) -> str:
    entries = [
        (
            name,
            field.type_,
            field.desc,
            field.constraints,
        )
        for name, field in fields.items()
    ]
    return _format_field_description_lines(entries=entries)


def _format_field_description_lines(
    *,
    entries: list[tuple[str, object, str, str | None]],
) -> str:
    field_descriptions = []
    for idx, (name, annotation, desc, constraints) in enumerate(entries):
        field_message = f"{idx + 1}. `{name}`"
        field_message += f" ({get_annotation_name(annotation)})"
        custom_types = extract_field_types_from_annotation(annotation)
        for custom_type in custom_types:
            if len(custom_type.description()) > 0:
                desc += f"\n    Type description of {get_annotation_name(custom_type)}: {custom_type.description()}"
        field_message += f": {desc}"
        if constraints:
            field_message += f"\nConstraints: {constraints}"
        field_descriptions.append(field_message)
    return "\n".join(field_descriptions).strip()


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
