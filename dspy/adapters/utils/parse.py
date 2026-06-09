import ast
import enum
import types
from typing import Any, Literal, Union, cast, get_args, get_origin

import json_repair
import pydantic
from pydantic import TypeAdapter

from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.utils.fields import _annotation_is_subclass
from dspy.task_spec.field_spec import FieldSpec
from dspy.utils.exceptions import AdapterParseError


def validate_parsed_fields(
    *,
    adapter_name: str,
    task_spec: Any,
    lm_response: str,
    fields: dict[str, Any],
) -> None:
    if fields.keys() != task_spec.output_fields.keys():
        raise AdapterParseError(
            adapter_name=adapter_name,
            task_spec=task_spec,
            lm_response=lm_response,
            parsed_result=fields,
        )


def parse_output_field(
    *,
    adapter_name: str,
    task_spec: Any,
    field_name: str,
    raw_value: object,
    lm_response: str,
    field: FieldSpec,
) -> object:
    try:
        return parse_value(raw_value, field.type_)
    except Exception as exc:
        raise AdapterParseError(
            adapter_name=adapter_name,
            task_spec=task_spec,
            lm_response=lm_response,
            message=f"Failed to parse field {field_name!r}: {exc}",
        ) from exc


def find_enum_member(enum_type: enum.EnumMeta, identifier: object) -> enum.Enum:
    for member in enum_type:
        member = cast("enum.Enum", member)
        if member.value == identifier:
            return member
    if isinstance(identifier, str) and identifier in enum_type.__members__:
        return cast("enum.Enum", enum_type[identifier])
    raise ValueError(f"{identifier} is not a valid name or value for the enum {enum_type.__name__}")


def parse_value(value: object, annotation: object) -> object:
    if annotation is str:
        return str(value)
    if isinstance(annotation, enum.EnumMeta):
        return find_enum_member(enum_type=annotation, identifier=value)
    origin = get_origin(annotation)
    if origin is Literal:
        allowed = get_args(annotation)
        if value in allowed:
            return value
        if isinstance(value, str):
            v = value.strip()
            if v.startswith(("Literal[", "str[")) and v.endswith("]"):
                v = v[v.find("[") + 1 : -1]
            if len(v) > 1 and v[0] == v[-1] and (v[0] in "\"'"):
                v = v[1:-1]
            if v in allowed:
                return v
        raise ValueError(f"{value!r} is not one of {allowed!r}")
    if not isinstance(value, str):
        return TypeAdapter(annotation).validate_python(value)
    if origin in (Union, types.UnionType) and type(None) in get_args(annotation) and (str in get_args(annotation)):
        return TypeAdapter(annotation).validate_python(value)
    candidate = json_repair.loads(value)
    if candidate == "" and value != "":
        try:
            candidate = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            candidate = value
    try:
        return TypeAdapter(annotation).validate_python(candidate)
    except pydantic.ValidationError as e:
        if _annotation_is_subclass(annotation=annotation, expected_base=DspyType):
            try:
                return TypeAdapter(annotation).validate_python(value)
            except Exception:
                raise e
        raise
