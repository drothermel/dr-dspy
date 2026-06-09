"""Pydantic wire models for TaskSpec persistence and state round-trips."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from dspy.task_spec.field_spec import _UNSET, FieldSpec, input_field, output_field, validate_field_name
from dspy.task_spec.type_registry import type_from_str, type_to_str

TASK_SPEC_VERSION = 3


class FieldSpecWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    desc: str
    role: Literal["input", "output"]
    prefix: str
    is_type_undefined: bool = False
    constraints: str | None = None
    has_default: bool = False
    default: Any = None

    @model_validator(mode="after")
    def _default_consistency(self) -> Self:
        if self.has_default and "default" not in self.model_fields_set:
            raise ValueError(f"field_spec for field {self.name!r} has has_default=true but missing key 'default'.")
        return self

    def to_field_spec(self, *, custom_types: dict[str, type] | None = None) -> FieldSpec:
        validate_field_name(self.name)
        type_ = type_from_str(self.type, custom_types=custom_types)
        common = {"desc": self.desc, "prefix": self.prefix, "constraints": self.constraints}
        if self.role == "input":
            return input_field(
                self.name,
                type_,
                is_type_undefined=self.is_type_undefined,
                default=self.default if self.has_default else _UNSET,
                **common,
            )
        return output_field(self.name, type_, **common)

    @classmethod
    def from_field_spec(cls, field: FieldSpec) -> Self:
        data: dict[str, Any] = {
            "name": field.name,
            "type": type_to_str(field.type_),
            "desc": field.desc,
            "role": field.role.value,
            "prefix": field.prefix,
            "is_type_undefined": field.is_type_undefined,
            "constraints": field.constraints,
        }
        if field.has_default:
            data["has_default"] = True
            data["default"] = field.default
        return cls.model_validate(data)


class TaskSpecWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_spec_version: int
    name: str
    instructions: str
    inputs: list[FieldSpecWire]
    outputs: list[FieldSpecWire]

    @model_validator(mode="after")
    def _version_gate(self) -> Self:
        if self.task_spec_version != TASK_SPEC_VERSION:
            raise ValueError(
                f"Unsupported task_spec_version: {self.task_spec_version!r}. Expected {TASK_SPEC_VERSION}. Recompile or recreate the program with the current DSPy version."
            )
        return self


def field_spec_to_dict(field: FieldSpec) -> dict[str, Any]:
    return FieldSpecWire.from_field_spec(field).model_dump(mode="python")


def field_spec_from_dict(data: dict[str, Any], *, custom_types: dict[str, type] | None = None) -> FieldSpec:
    return FieldSpecWire.model_validate(data).to_field_spec(custom_types=custom_types)
