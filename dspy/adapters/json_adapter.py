from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import regex
from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.policies.response_format import StructuredOutputPolicy
from dspy.adapters.format_field_structure import build_field_structure_instructions, build_role_field_sections
from dspy.adapters.format_shared import ChatFormatMixin, format_fields_with_headers, output_field_type_hint
from dspy.adapters.utils import load_json, parse_output_field, validate_parsed_fields
from dspy.errors import AdapterParseError
from dspy.serialization.json import to_jsonable
from dspy.task_spec import FieldBinding
from dspy.task_spec.field_spec import FieldRole

if TYPE_CHECKING:
    from dspy.adapters.types.field_type import NativeResponseFieldType
    from dspy.runtime.callback import Callback
    from dspy.task_spec import TaskSpec


class JSONAdapter(ChatFormatMixin, Adapter):
    capabilities = AdapterCapabilities(
        supports_finetune=False,
        field_value_role="assistant",
        default_native_fc=True,
        supports_structured_output=True,
    )

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        use_native_function_calling: bool = True,
        parallel_tool_calls: bool | None = None,
        native_response_types: list[type[NativeResponseFieldType]] | None = None,
        allow_json_repair: bool = True,
    ) -> None:
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
            native_response_types=native_response_types,
            allow_json_repair=allow_json_repair,
        )
        self.response_format_policy = StructuredOutputPolicy()

    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        return build_field_structure_instructions(
            input_preamble="Inputs will have the following structure:",
            input_section=build_role_field_sections(self, task_spec, FieldRole.INPUT, role_label="user"),
            output_preamble="Outputs will be a JSON object with the following fields.",
            output_section=build_role_field_sections(self, task_spec, FieldRole.OUTPUT, role_label="assistant"),
        )

    @override
    def user_message_output_requirements(self, task_spec: TaskSpec) -> str:
        message = "Respond with a JSON object in the following order of fields: "
        message += ", then ".join(
            f"`{f}`{output_field_type_hint(field.type_)}" for f, field in task_spec.output_fields.items()
        )
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
        try:
            fields = load_json(completion, repair=self.allow_json_repair)
        except json.JSONDecodeError as exc:
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                task_spec=task_spec,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            ) from exc
        if not isinstance(fields, dict):
            pattern = r"\{(?:[^{}]|(?R))*\}"
            match = regex.search(pattern, completion, regex.DOTALL)
            if match:
                completion = match.group(0)
                fields = load_json(completion, repair=self.allow_json_repair)
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
                    repair=self.allow_json_repair,
                )
        validate_parsed_fields(adapter_name="JSONAdapter", task_spec=task_spec, lm_response=completion, fields=fields)
        return fields

    @override
    def format_field_with_value(self, fields_with_values: dict[FieldBinding, Any], **kwargs: Any) -> str:
        if kwargs.get("role", "user") == "user":
            return format_fields_with_headers(fields_with_values)
        d = {binding.name: value for binding, value in fields_with_values.items()}
        return json.dumps(to_jsonable(d), indent=2, ensure_ascii=False)
