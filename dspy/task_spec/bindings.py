from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

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
