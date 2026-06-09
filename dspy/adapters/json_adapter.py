from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast, get_origin

import json_repair
import pydantic
import regex
from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.policies.response_format import StructuredOutputPolicy
from dspy.adapters.format_shared import ChatFormatMixin
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_output_field,
    serialize_for_json,
    translate_field_type,
    validate_parsed_fields,
)
from dspy.errors import AdapterParseError
from dspy.task_spec import FieldBinding, field_bindings
from dspy.task_spec.field_spec import FieldRole

if TYPE_CHECKING:
    from dspy.adapters.types.base_type import Type
    from dspy.runtime.callback import Callback
    from dspy.task_spec import TaskSpec


def _has_open_ended_mapping(task_spec: TaskSpec) -> bool:
    return any(get_origin(field.type_) is dict for field in task_spec.output_fields.values())


class JSONAdapter(ChatFormatMixin, Adapter):
    capabilities = AdapterCapabilities(supports_finetune=False, field_value_role="assistant")

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        use_native_function_calling: bool = True,
        parallel_tool_calls: bool | None = None,
        native_response_types: list[type[Type]] | None = None,
    ) -> None:
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
            native_response_types=native_response_types,
        )
        self.response_format_policy = StructuredOutputPolicy()

    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_task_spec_fields_for_instructions(role: FieldRole, role_label: str) -> str:
            return self.format_field_with_value(
                fields_with_values={
                    binding: translate_field_type(binding.field) for binding in field_bindings(task_spec, role=role)
                },
                role=role_label,
            )

        parts.append("Inputs will have the following structure:")
        parts.append(format_task_spec_fields_for_instructions(FieldRole.INPUT, "user"))
        parts.append("Outputs will be a JSON object with the following fields.")
        parts.append(format_task_spec_fields_for_instructions(FieldRole.OUTPUT, "assistant"))
        return "\n\n".join(parts).strip()

    @override
    def user_message_output_requirements(self, task_spec: TaskSpec) -> str:
        def type_info(field_type: object) -> str:
            if field_type == ToolCalls:
                return ' (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})'
            return (
                f" (must be formatted as a valid Python {get_annotation_name(field_type)})"
                if field_type is not str
                else ""
            )

        message = "Respond with a JSON object in the following order of fields: "
        message += ", then ".join(f"`{f}`{type_info(field.type_)}" for f, field in task_spec.output_fields.items())
        message += "."
        return message

    @override
    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        fields_with_values = {
            FieldBinding(name=field_name, field=task_spec.output_fields[field_name]): outputs.get(
                field_name, missing_field_message
            )
            for field_name in task_spec.output_fields
        }
        return self.format_field_with_value(fields_with_values=fields_with_values, role="assistant")

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        fields = json_repair.loads(completion)
        if not isinstance(fields, dict):
            pattern = r"\{(?:[^{}]|(?R))*\}"
            match = regex.search(pattern, completion, regex.DOTALL)
            if match:
                completion = match.group(0)
                fields = json_repair.loads(completion)
        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                task_spec=task_spec,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            )
        fields = {k: v for k, v in fields.items() if k in task_spec.output_fields}
        for k, v in fields.items():
            if k in task_spec.output_fields:
                fields[k] = parse_output_field(
                    adapter_name="JSONAdapter",
                    task_spec=task_spec,
                    field_name=k,
                    raw_value=v,
                    lm_response=completion,
                    field=task_spec.output_fields[k],
                )
        validate_parsed_fields(adapter_name="JSONAdapter", task_spec=task_spec, lm_response=completion, fields=fields)
        return fields

    @override
    def format_field_with_value(self, fields_with_values: dict[FieldBinding, Any], role: str = "user") -> str:
        if role == "user":
            output = []
            for binding, field_value in fields_with_values.items():
                formatted_field_value = format_field_value(field=binding.field, value=field_value)
                output.append(f"[[ ## {binding.name} ## ]]\n{formatted_field_value}")
            return "\n\n".join(output).strip()
        d = {binding.name: value for binding, value in fields_with_values.items()}
        return json.dumps(serialize_for_json(d), indent=2, ensure_ascii=False)


def _get_structured_outputs_response_format(
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
