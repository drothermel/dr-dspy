from __future__ import annotations

import enum
import inspect
import json
from collections.abc import Mapping
from typing import Any, Literal, cast

import pydantic

from dspy.adapters.types.code import Code
from dspy.adapters.types.reasoning import Reasoning
from dspy.task_spec.field_spec import FieldRole, FieldSpec


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
