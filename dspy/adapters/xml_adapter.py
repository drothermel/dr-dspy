import re
from typing import Any

from typing_extensions import override

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import (
    build_multimodal_user_message_content,
    format_field_value,
    inputs_include_multimodal_custom_type_values,
    parse_output_field,
    translate_field_type,
    validate_parsed_fields,
)
from dspy.task_spec import FieldBinding, TaskSpec, field_bindings
from dspy.task_spec.field_spec import FieldRole


class XMLAdapter(ChatAdapter):
    field_pattern = re.compile("<(?P<name>\\w+)>((?P<content>.*?))</\\1>", re.DOTALL)

    @override
    def format_field_with_value(self, fields_with_values: dict[FieldBinding, Any]) -> str:
        output = []
        for binding, field_value in fields_with_values.items():
            formatted = format_field_value(field=binding.field, value=field_value)
            output.append(f"<{binding.name}>\n{formatted}\n</{binding.name}>")
        return "\n\n".join(output).strip()

    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_task_spec_fields_for_instructions(role: FieldRole) -> str:
            return self.format_field_with_value(
                fields_with_values={
                    binding: translate_field_type(binding.field) for binding in field_bindings(task_spec, role=role)
                }
            )

        parts.append(format_task_spec_fields_for_instructions(FieldRole.INPUT))
        parts.append(format_task_spec_fields_for_instructions(FieldRole.OUTPUT))
        return "\n\n".join(parts).strip()

    @override
    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str | list[dict[str, Any]]:
        if inputs_include_multimodal_custom_type_values(task_spec=task_spec, inputs=inputs):
            output_requirements = self.user_message_output_requirements(task_spec) if main_request else None
            return build_multimodal_user_message_content(
                task_spec=task_spec,
                inputs=inputs,
                prefix=prefix,
                suffix=suffix,
                main_request=main_request,
                output_requirements=output_requirements,
                field_wrapper="xml",
            )
        messages = [prefix]
        messages.append(
            self.format_field_with_value(
                {
                    FieldBinding(name=field_name, field=field): inputs.get(field_name)
                    for field_name, field in task_spec.input_fields.items()
                    if field_name in inputs
                }
            )
        )
        if main_request:
            output_requirements = self.user_message_output_requirements(task_spec)
            if output_requirements is not None:
                messages.append(output_requirements)
        messages.append(suffix)
        return "\n\n".join(messages).strip()

    @override
    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        return self.format_field_with_value(
            {
                FieldBinding(name=field_name, field=field): outputs.get(field_name, missing_field_message)
                for field_name, field in task_spec.output_fields.items()
            }
        )

    @override
    def user_message_output_requirements(self, task_spec: TaskSpec) -> str:
        message = "Respond with the corresponding output fields wrapped in XML tags "
        message += ", then ".join(f"`<{f}>`" for f in task_spec.output_fields)
        message += "."
        return message

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        raw_fields: dict[str, str] = {}
        for match in self.field_pattern.finditer(completion):
            name = match.group("name")
            content = match.group("content").strip()
            if name in task_spec.output_fields and name not in raw_fields:
                raw_fields[name] = content
        fields = {
            k: parse_output_field(
                adapter_name="XMLAdapter",
                task_spec=task_spec,
                field_name=k,
                raw_value=v,
                lm_response=completion,
                field=task_spec.output_fields[k],
            )
            for k, v in raw_fields.items()
        }
        validate_parsed_fields(adapter_name="XMLAdapter", task_spec=task_spec, lm_response=completion, fields=fields)
        return fields
