import ast
import enum
import types
from typing import Literal, Union, cast, get_args, get_origin

import json_repair
import pydantic
from pydantic import TypeAdapter

from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.utils.fields import _annotation_is_subclass


def find_enum_member(enum_type: enum.EnumMeta, identifier: object) -> enum.Enum:
    """
    Finds the enum member corresponding to the specified identifier, which may be the
    enum member's name or value.

    Args:
        enum: The enum to search for the member.
        identifier: If the enum is explicitly-valued, this is the value of the enum member to find.
                    If the enum is auto-valued, this is the name of the enum member to find.
    Returns:
        The enum member corresponding to the specified identifier.
    """
    # Check if the identifier is a valid enum member value *before* checking if it's a valid enum
    # member name, since the identifier will be a value for explicitly-valued enums. This handles
    # the (rare) case where an enum member value is the same as another enum member's name in
    # an explicitly-valued enum
    for member in enum_type:
        member = cast("enum.Enum", member)
        if member.value == identifier:
            return member

    # If the identifier is not a valid enum member value, check if it's a valid enum member name,
    # since the identifier will be a member name for auto-valued enums
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
            if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]

            if v in allowed:
                return v

        raise ValueError(f"{value!r} is not one of {allowed!r}")

    if not isinstance(value, str):
        return TypeAdapter(annotation).validate_python(value)

    if origin in (Union, types.UnionType) and type(None) in get_args(annotation) and str in get_args(annotation):
        # Handle union annotations such as `str | None`.
        return TypeAdapter(annotation).validate_python(value)

    candidate = json_repair.loads(value)  # json_repair.loads returns "" on failure.
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
                # For dspy.Type, try parsing from the original value in case it has a custom parser
                return TypeAdapter(annotation).validate_python(value)
            except Exception:
                raise e
        raise
