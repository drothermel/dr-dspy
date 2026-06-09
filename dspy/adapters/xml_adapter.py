import re
from typing import Any

from pydantic.fields import FieldInfo
from typing_extensions import override

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.format_shared import FieldInfoWithName
from dspy.adapters.utils import (
    build_multimodal_user_message_content,
    format_field_value,
    inputs_include_multimodal_custom_type_values,
    parse_output_field,
    translate_field_type,
    validate_parsed_fields,
)
from dspy.task_spec import TaskSpec
from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos, task_spec_output_field_infos


class XMLAdapter(ChatAdapter):
    field_pattern = re.compile("<(?P<name>\\w+)>((?P<content>.*?))</\\1>", re.DOTALL)

    @override
    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, field_value in fields_with_values.items():
            formatted = format_field_value(field_info=field.info, value=field_value)
            output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
        return "\n\n".join(output).strip()

    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")
        input_field_infos = task_spec_input_field_infos(task_spec)
        output_field_infos = task_spec_output_field_infos(task_spec)

        def format_task_spec_fields_for_instructions(field_infos: dict[str, FieldInfo]) -> str:
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(
                        field_name=field_name, field_info=field_info
                    )
                    for field_name, field_info in field_infos.items()
                }
            )

        parts.append(format_task_spec_fields_for_instructions(input_field_infos))
        parts.append(format_task_spec_fields_for_instructions(output_field_infos))
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
        input_field_infos = task_spec_input_field_infos(task_spec)
        messages = [prefix]
        messages.append(
            self.format_field_with_value(
                {
                    FieldInfoWithName(name=k, info=input_field_infos[k]): inputs.get(k)
                    for k in task_spec.input_fields
                    if k in inputs
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
        output_field_infos = task_spec_output_field_infos(task_spec)
        return self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=output_field_infos[k]): outputs.get(k, missing_field_message)
                for k in task_spec.output_fields
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
                field_info=task_spec_output_field_infos(task_spec)[k],
            )
            for k, v in raw_fields.items()
        }
        validate_parsed_fields(adapter_name="XMLAdapter", task_spec=task_spec, lm_response=completion, fields=fields)
        return fields
