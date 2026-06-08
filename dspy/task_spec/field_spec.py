import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_UNSET = object()


class FieldRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


def infer_prefix(attribute_name: str) -> str:
    """Infer a human-readable prefix from a field name."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", attribute_name)
    intermediate_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    with_underscores_around_numbers = re.sub(r"([a-zA-Z])(\d)", r"\1_\2", intermediate_name)
    with_underscores_around_numbers = re.sub(
        r"(\d)([a-zA-Z])",
        r"\1_\2",
        with_underscores_around_numbers,
    )
    words = with_underscores_around_numbers.split("_")
    title_cased_words = [word if word.isupper() else word.capitalize() for word in words]
    return " ".join(title_cased_words)


def input_field(
    name: str,
    type_: Any = str,
    *,
    desc: str | None = None,
    prefix: str | None = None,
    is_type_undefined: bool = False,
    constraints: str | None = None,
    default: Any = _UNSET,
) -> "FieldSpec":
    has_default = default is not _UNSET
    return FieldSpec(
        name=name,
        type=type_,
        desc=desc if desc is not None else f"${{{name}}}",
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
    desc: str | None = None,
    prefix: str | None = None,
    constraints: str | None = None,
) -> "FieldSpec":
    return FieldSpec(
        name=name,
        type=type_,
        desc=desc if desc is not None else f"${{{name}}}",
        role=FieldRole.OUTPUT,
        prefix=prefix if prefix is not None else infer_prefix(name) + ":",
        constraints=constraints,
    )


class FieldSpec(BaseModel):
    """Immutable description of one TaskSpec input or output field."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    type_: Any = Field(alias="type")
    desc: str
    role: Literal["input", "output"]
    prefix: str
    is_type_undefined: bool = False
    constraints: str | None = None
    has_default: bool = False
    default: Any = None

    @classmethod
    def input(
        cls,
        name: str,
        type_: Any = str,
        *,
        desc: str | None = None,
        prefix: str | None = None,
        is_type_undefined: bool = False,
        constraints: str | None = None,
        default: Any = _UNSET,
    ) -> "FieldSpec":
        return input_field(
            name,
            type_,
            desc=desc,
            prefix=prefix,
            is_type_undefined=is_type_undefined,
            constraints=constraints,
            default=default,
        )

    @classmethod
    def output(
        cls,
        name: str,
        type_: Any = str,
        *,
        desc: str | None = None,
        prefix: str | None = None,
        constraints: str | None = None,
    ) -> "FieldSpec":
        return output_field(
            name,
            type_,
            desc=desc,
            prefix=prefix,
            constraints=constraints,
        )

    def with_updates(
        self,
        *,
        desc: str | None = None,
        prefix: str | None = None,
        type_: Any = None,
        constraints: str | None = None,
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
