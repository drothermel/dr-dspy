from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.policies.json_parse_fallback import JSONParseFallbackPolicy
from dspy.adapters.format.prompt_sections import (
    format_field_description,
    format_header_finetune_data,
    format_task_description,
)
from dspy.adapters.format.xml_formatter import XmlFieldFormatter
from dspy.adapters.format_field_structure import build_field_structure_instructions, build_role_field_sections
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import (
    build_multimodal_user_message_content,
    inputs_include_multimodal_custom_type_values,
    parse_output_field,
    validate_parsed_fields,
)
from dspy.task_spec import FieldBinding, TaskSpec
from dspy.task_spec.field_spec import FIELD_NAME_BODY, FieldRole

if TYPE_CHECKING:
    from dspy.adapters.call.policies.parse_fallback import NoOpParseFallbackPolicy
    from dspy.adapters.types.field_type import NativeResponseFieldType
    from dspy.core.types import UserMessageContent
    from dspy.runtime.callback import Callback


class XMLAdapter(Adapter):
    capabilities = AdapterCapabilities(
        supports_finetune=True,
        field_value_role="none",
        default_native_fc=False,
        supports_structured_output=False,
    )
    field_pattern = re.compile(rf"<(?P<name>{FIELD_NAME_BODY})>((?P<content>.*?))</\1>", re.DOTALL)

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[NativeResponseFieldType]] | None = None,
        parallel_tool_calls: bool | None = None,
        allow_json_repair: bool = False,
        json_fallback: JSONAdapter | None = None,
        parse_fallback_policy: JSONParseFallbackPolicy | NoOpParseFallbackPolicy | None = None,
    ) -> None:
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
            native_response_types=native_response_types,
            allow_json_repair=allow_json_repair,
        )
        self.field_formatter = XmlFieldFormatter()
        self._json_fallback = json_fallback
        if parse_fallback_policy is None:
            self.parse_fallback_policy = JSONParseFallbackPolicy(fallback_factory=self._json_adapter_fallback)
        else:
            self.parse_fallback_policy = parse_fallback_policy

    def _json_adapter_fallback(self) -> JSONAdapter:
        if self._json_fallback is not None:
            return self._json_fallback
        return JSONAdapter(
            callbacks=self.callbacks,
            use_native_function_calling=self.use_native_function_calling,
            parallel_tool_calls=self.parallel_tool_calls,
            native_response_types=self.native_response_types,
            allow_json_repair=self.allow_json_repair,
        )

    @override
    def format_field_description(self, task_spec: TaskSpec) -> str:
        return format_field_description(task_spec)

    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        return build_field_structure_instructions(
            input_section=build_role_field_sections(self._require_field_formatter(), task_spec, FieldRole.INPUT),
            output_section=build_role_field_sections(self._require_field_formatter(), task_spec, FieldRole.OUTPUT),
        )

    @override
    def format_task_description(self, task_spec: TaskSpec) -> str:
        return format_task_description(task_spec)

    @override
    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> UserMessageContent:
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
            self._require_field_formatter().format_field_with_value(
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
        return self._require_field_formatter().format_field_with_value(
            {
                FieldBinding(name=field_name, field=field): outputs.get(field_name, missing_field_message)
                for field_name, field in task_spec.output_fields.items()
            }
        )

    def user_message_output_requirements(self, task_spec: TaskSpec) -> str:
        message = "Respond with the corresponding output fields wrapped in XML tags "
        message += ", then ".join(f"`<{f}>`" for f in task_spec.output_fields)
        message += "."
        return message

    @override
    def format_finetune_data(
        self,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        return format_header_finetune_data(self, self._require_field_formatter(), task_spec, demos, inputs, outputs)

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
