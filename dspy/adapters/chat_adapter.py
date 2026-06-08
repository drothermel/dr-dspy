from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, Any, NamedTuple

from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    build_multimodal_user_message_content,
    format_field_value,
    get_annotation_name,
    inputs_include_multimodal_custom_type_values,
    parse_value,
    translate_field_type,
)
from dspy.task_spec.formatting import get_field_spec_description_string
from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos, task_spec_output_field_infos
from dspy.utils.exceptions import AdapterParseError, LMError

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.adapters.types.base_type import Type
    from dspy.clients.base_lm import BaseLM
    from dspy.task_spec import TaskSpec
    from dspy.utils.callback import BaseCallback
field_header_pattern = re.compile("\\[\\[ ## (\\w+) ## \\]\\]")


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


class ChatAdapter(Adapter):
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        use_json_adapter_fallback: bool = True,
        parallel_tool_calls: bool | None = None,
    ) -> None:
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
            native_response_types=native_response_types,
        )
        self.use_json_adapter_fallback = use_json_adapter_fallback

    def _make_json_adapter_fallback(self) -> JSONAdapter:
        from dspy.adapters.json_adapter import JSONAdapter

        return JSONAdapter(
            use_native_function_calling=self.use_native_function_calling, parallel_tool_calls=self.parallel_tool_calls
        )

    @override
    async def acall(
        self, *, lm: BaseLM, config: Any, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        try:
            return await super().acall(lm=lm, config=config, task_spec=task_spec, demos=demos, inputs=inputs)
        except TypeError:
            raise
        except Exception as e:
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, LMError) or isinstance(self, JSONAdapter) or (not self.use_json_adapter_fallback):
                raise
            return await self._make_json_adapter_fallback().acall(
                lm=lm, config=config, task_spec=task_spec, demos=demos, inputs=inputs
            )

    @override
    def format_field_description(self, task_spec: TaskSpec) -> str:
        return f"Your input fields are:\n{get_field_spec_description_string(task_spec.input_fields)}\nYour output fields are:\n{get_field_spec_description_string(task_spec.output_fields)}"

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
        parts.append("[[ ## completed ## ]]\n")
        return "\n\n".join(parts).strip()

    @override
    def format_task_description(self, task_spec: TaskSpec) -> str:
        instructions = textwrap.dedent(task_spec.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

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
            )
        input_field_infos = task_spec_input_field_infos(task_spec)
        messages = [prefix]
        for k in task_spec.input_fields:
            if k in inputs:
                value = inputs.get(k)
                formatted_field_value = format_field_value(field_info=input_field_infos[k], value=value)
                messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")
        if main_request:
            output_requirements = self.user_message_output_requirements(task_spec)
            if output_requirements is not None:
                messages.append(output_requirements)
        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def user_message_output_requirements(self, task_spec: TaskSpec) -> str:

        def type_info(field_type: object) -> str:
            if field_type == ToolCalls:
                return ' (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})'
            if field_type is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(field_type)})"
            return ""

        message = "Respond with the corresponding output fields, starting with the field "
        message += ", then ".join(
            (f"`[[ ## {f} ## ]]`{type_info(field.type_)}" for f, field in task_spec.output_fields.items())
        )
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message

    @override
    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        output_field_infos = task_spec_output_field_infos(task_spec)
        assistant_message_content = self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=output_field_infos[k]): outputs.get(k, missing_field_message)
                for k in task_spec.output_fields
            }
        )
        assistant_message_content += "\n\n[[ ## completed ## ]]\n"
        return assistant_message_content

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        sections = [(None, [])]
        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)
        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        fields = {}
        for k, v in sections:
            if k not in fields and k in task_spec.output_fields:
                try:
                    fields[k] = parse_value(value=v, annotation=task_spec.output_fields[k].type_)
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="ChatAdapter",
                        task_spec=task_spec,
                        lm_response=completion,
                        message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
                    )
        if fields.keys() != task_spec.output_fields.keys():
            raise AdapterParseError(
                adapter_name="ChatAdapter", task_spec=task_spec, lm_response=completion, parsed_result=fields
            )
        return fields

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
        return "\n\n".join(output).strip()

    def format_finetune_data(
        self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        from dspy.clients.openai_format import message_to_openai_chat

        system_user_messages = [
            message_to_openai_chat(message) for message in self.format(task_spec=task_spec, demos=demos, inputs=inputs)
        ]
        assistant_message_content = self.format_assistant_message_content(task_spec=task_spec, outputs=outputs)
        assistant_message = {"role": "assistant", "content": assistant_message_content}
        messages = system_user_messages + [assistant_message]
        return {"messages": messages}
