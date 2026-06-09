from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.format_field_structure import build_field_structure_instructions, build_role_field_sections
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import build_multimodal_user_message_content, inputs_include_multimodal_custom_type_values
from dspy.clients.openai_format.chat_request import message_to_openai_chat
from dspy.task_spec import (
    FieldBinding,
    format_field_value,
    get_annotation_name,
)
from dspy.task_spec.field_spec import FIELD_NAME_BODY, FieldRole
from dspy.task_spec.formatting import get_field_spec_description_string

if TYPE_CHECKING:
    from dspy.adapters.base.protocols import ChatFormattableAdapter
    from dspy.task_spec import TaskSpec

FIELD_HEADER_PATTERN = re.compile(rf"\[\[ ## ({FIELD_NAME_BODY}) ## \]\]")


def format_fields_with_headers(fields_with_values: dict[FieldBinding, Any]) -> str:
    output = []
    for binding, field_value in fields_with_values.items():
        formatted_field_value = format_field_value(field=binding.field, value=field_value)
        output.append(f"[[ ## {binding.name} ## ]]\n{formatted_field_value}")
    return "\n\n".join(output).strip()


def output_field_type_hint(field_type: object) -> str:
    if field_type == ToolCalls:
        return ' (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})'
    if field_type is not str:
        return f" (must be formatted as a valid Python {get_annotation_name(field_type)})"
    return ""


class ChatFormatMixin:
    def format_field_description(self, task_spec: TaskSpec) -> str:
        return (
            f"Your input fields are:\n{get_field_spec_description_string(task_spec.input_fields)}\n"
            f"Your output fields are:\n{get_field_spec_description_string(task_spec.output_fields)}"
        )

    def format_field_structure(self, task_spec: TaskSpec) -> str:
        return build_field_structure_instructions(
            input_section=build_role_field_sections(self, task_spec, FieldRole.INPUT),
            output_section=build_role_field_sections(self, task_spec, FieldRole.OUTPUT),
            completed_marker="[[ ## completed ## ]]\n",
        )

    def format_task_description(self, task_spec: TaskSpec) -> str:
        instructions = textwrap.dedent(task_spec.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

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
            )
        messages = [prefix]
        for field_name, field in task_spec.input_fields.items():
            if field_name in inputs:
                value = inputs.get(field_name)
                formatted_field_value = format_field_value(field=field, value=value)
                messages.append(f"[[ ## {field_name} ## ]]\n{formatted_field_value}")
        if main_request:
            output_requirements = self.user_message_output_requirements(task_spec)
            if output_requirements is not None:
                messages.append(output_requirements)
        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def user_message_output_requirements(self, task_spec: TaskSpec) -> str:
        message = "Respond with the corresponding output fields, starting with the field "
        message += ", then ".join(
            f"`[[ ## {f} ## ]]`{output_field_type_hint(field.type_)}" for f, field in task_spec.output_fields.items()
        )
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message

    def format_assistant_message_content(
        self,
        task_spec: TaskSpec,
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        assistant_message_content = self.format_field_with_value(
            {
                FieldBinding(name=field_name, field=task_spec.output_fields[field_name]): outputs.get(
                    field_name, missing_field_message
                )
                for field_name in task_spec.output_fields
            }
        )
        assistant_message_content += "\n\n[[ ## completed ## ]]\n"
        return assistant_message_content

    def format_field_with_value(self, fields_with_values: dict[FieldBinding, Any], **kwargs: Any) -> str:
        return format_fields_with_headers(fields_with_values)

    def format_finetune_data(
        self,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        formattable = cast("ChatFormattableAdapter", self)
        system_user_messages = [
            message_to_openai_chat(message)
            for message in formattable.format(task_spec=task_spec, demos=demos, inputs=inputs)
        ]
        assistant_message_content = self.format_assistant_message_content(task_spec=task_spec, outputs=outputs)
        assistant_message = {"role": "assistant", "content": assistant_message_content}
        messages = system_user_messages + [assistant_message]
        return {"messages": messages}
