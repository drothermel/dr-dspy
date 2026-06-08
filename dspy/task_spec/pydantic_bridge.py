"""Synthesize Pydantic models from TaskSpec instances for JSON-schema tooling."""

from typing import Any

from pydantic import BaseModel, Field, create_model

from dspy.task_spec.task_spec import TaskSpec
from dspy.utils.constants import IS_TYPE_UNDEFINED


def task_spec_to_pydantic_model(spec: TaskSpec) -> type[BaseModel]:
    """Build a dynamic Pydantic model mirroring a TaskSpec's fields and metadata."""
    field_defs: dict[str, Any] = {}
    for field in (*spec.inputs, *spec.outputs):
        json_schema_extra: dict[str, Any] = {
            "__dspy_field_type": field.role,
            "desc": field.desc,
            "prefix": field.prefix,
        }
        if field.is_type_undefined:
            json_schema_extra[IS_TYPE_UNDEFINED] = True
        if field.constraints:
            json_schema_extra["constraints"] = field.constraints
        field_defs[field.name] = (
            field.type_,
            Field(json_schema_extra=json_schema_extra, description=field.desc),
        )
    model_name = spec.name.replace(" ", "_")
    return create_model(model_name, **field_defs)
