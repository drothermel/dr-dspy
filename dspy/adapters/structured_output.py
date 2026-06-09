from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, get_origin

import pydantic

from dspy.adapters.types.tool import ToolCalls

if TYPE_CHECKING:
    from dspy.task_spec import TaskSpec


def has_open_ended_mapping(task_spec: TaskSpec) -> bool:
    return any(get_origin(field.type_) is dict for field in task_spec.output_fields.values())


def get_structured_outputs_response_format(
    task_spec: TaskSpec, use_native_function_calling: bool = True
) -> type[pydantic.BaseModel]:
    for name, field in task_spec.output_fields.items():
        if get_origin(field.type_) is dict:
            raise ValueError(
                f"Field '{name}' has an open-ended mapping type which is not supported by Structured Outputs."
            )
    fields = {}
    for name, field in task_spec.output_fields.items():
        field_type = field.type_
        if use_native_function_calling and field_type == ToolCalls:
            continue
        fields[name] = (field_type, ...)
    pydantic_model = pydantic.create_model(
        "DSPyProgramOutputs",
        __config__=pydantic.ConfigDict(extra="forbid"),
        **cast("dict[str, Any]", fields),
    )
    schema = pydantic_model.model_json_schema()
    for prop in schema.get("properties", {}).values():
        prop.pop("json_schema_extra", None)

    def enforce_required(schema_part: dict[str, Any]) -> None:
        if schema_part.get("type") == "object":
            props = schema_part.get("properties")
            if props is not None:
                schema_part["required"] = list(props.keys())
                schema_part["additionalProperties"] = False
                for sub_schema in props.values():
                    if isinstance(sub_schema, dict):
                        enforce_required(sub_schema)
            else:
                schema_part["properties"] = {}
                schema_part["required"] = []
                schema_part["additionalProperties"] = False
        if schema_part.get("type") == "array" and isinstance(schema_part.get("items"), dict):
            enforce_required(schema_part["items"])
        for key in ("$defs", "definitions"):
            if key in schema_part:
                for def_schema in schema_part[key].values():
                    enforce_required(def_schema)

    enforce_required(schema)

    def model_json_schema(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return schema

    pydantic_model.model_json_schema = model_json_schema
    return pydantic_model
