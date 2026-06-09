import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_UNSET = object()


class FieldRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


def infer_prefix(attribute_name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", "\\1_\\2", attribute_name)
    intermediate_name = re.sub("([a-z0-9])([A-Z])", "\\1_\\2", s1)
    with_underscores_around_numbers = re.sub("([a-zA-Z])(\\d)", "\\1_\\2", intermediate_name)
    with_underscores_around_numbers = re.sub("(\\d)([a-zA-Z])", "\\1_\\2", with_underscores_around_numbers)
    words = with_underscores_around_numbers.split("_")
    title_cased_words = [word if word.isupper() else word.capitalize() for word in words]
    return " ".join(title_cased_words)


def field_desc_from_name(name: str) -> str:
    return f"The {infer_prefix(name).lower()}."


def input_field(
    name: str,
    type_: Any = str,
    *,
    desc: str,
    prefix: str | None = None,
    is_type_undefined: bool = False,
    constraints: str | None = None,
    default: Any = _UNSET,
) -> "FieldSpec":
    has_default = default is not _UNSET
    return FieldSpec(
        name=name,
        type=type_,
        desc=desc,
        role=FieldRole.INPUT,
        prefix=prefix if prefix is not None else infer_prefix(name) + ":",
        is_type_undefined=is_type_undefined,
        constraints=constraints,
        has_default=has_default,
        default=None if default is _UNSET else default,
    )


def output_field(
    name: str,
    type_: Any = str,
    *,
    desc: str,
    prefix: str | None = None,
    constraints: str | None = None,
) -> "FieldSpec":
    return FieldSpec(
        name=name,
        type=type_,
        desc=desc,
        role=FieldRole.OUTPUT,
        prefix=prefix if prefix is not None else infer_prefix(name) + ":",
        constraints=constraints,
    )


class FieldSpec(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    type_: Any = Field(alias="type")
    desc: str
    role: FieldRole
    prefix: str
    is_type_undefined: bool = False
    constraints: str | None = None
    has_default: bool = False
    default: Any = None

    def with_updates(
        self, *, desc: str | None = None, prefix: str | None = None, type_: Any = None, constraints: str | None = None
    ) -> "FieldSpec":
        updates: dict[str, Any] = {}
        if desc is not None:
            updates["desc"] = desc
        if prefix is not None:
            updates["prefix"] = prefix
        if type_ is not None:
            updates["type"] = type_
        if constraints is not None:
            updates["constraints"] = constraints
        return self.model_copy(update=updates)
