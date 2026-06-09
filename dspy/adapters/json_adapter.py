from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import regex
from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.policies.response_format import StructuredOutputPolicy
from dspy.adapters.format.json_formatter import ASSISTANT_ROLE_LABEL, JsonFieldFormatter
from dspy.adapters.format.prompt_sections import (
    format_field_description,
    format_header_user_message_content,
    format_task_description,
    output_field_type_hint,
)
from dspy.adapters.format_field_structure import build_field_structure_instructions, build_role_field_sections
from dspy.adapters.utils import load_json, parse_output_field, validate_parsed_fields
from dspy.errors import AdapterParseError
from dspy.task_spec import FieldBinding
from dspy.task_spec.field_spec import FieldRole

if TYPE_CHECKING:
    from dspy.adapters.types.field_type import NativeResponseFieldType
    from dspy.runtime.callback import Callback
    from dspy.task_spec import TaskSpec


class JSONAdapter(Adapter):
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
        self.field_formatter = JsonFieldFormatter()
        self.response_format_policy = StructuredOutputPolicy()

    @override
    def format_field_description(self, task_spec: TaskSpec) -> str:
        return format_field_description(task_spec)

    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        return build_field_structure_instructions(
            input_preamble="Inputs will have the following structure:",
            input_section=build_role_field_sections(
                self._require_field_formatter(), task_spec, FieldRole.INPUT, role_label="user"
            ),
            output_preamble="Outputs will be a JSON object with the following fields.",
            output_section=build_role_field_sections(
                self._require_field_formatter(), task_spec, FieldRole.OUTPUT, role_label=ASSISTANT_ROLE_LABEL
            ),
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
    ) -> str | list[dict[str, Any]]:
        return format_header_user_message_content(
            self._require_field_formatter(),
            task_spec,
            inputs,
            prefix=prefix,
            suffix=suffix,
            main_request=main_request,
            output_requirements_fn=self.user_message_output_requirements,
        )

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
        return self._require_field_formatter().format_field_with_value(
            fields_with_values=fields_with_values,
            role_label=ASSISTANT_ROLE_LABEL,
        )

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        try:
            fields = load_json(completion, repair=False)
        except json.JSONDecodeError as strict_error:
            if not self.allow_json_repair:
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    task_spec=task_spec,
                    lm_response=completion,
                    message="LM response cannot be serialized to a JSON object.",
                ) from strict_error
            fenced = _extract_fenced_json(completion)
            if fenced is not None:
                try:
                    fields = load_json(fenced, repair=False)
                    completion = fenced
                except json.JSONDecodeError:
                    try:
                        fields = load_json(fenced, repair=True)
                        completion = fenced
                    except json.JSONDecodeError:
                        fields = None
            else:
                fields = None
            if fields is None:
                try:
                    fields = load_json(completion, repair=True)
                except json.JSONDecodeError as exc:
                    raise AdapterParseError(
                        adapter_name="JSONAdapter",
                        task_spec=task_spec,
                        lm_response=completion,
                        message="LM response cannot be serialized to a JSON object.",
                    ) from exc
        if not isinstance(fields, dict):
            object_match = _extract_json_object(completion)
            if object_match:
                completion = object_match
                fields = load_json(completion, repair=self.allow_json_repair)
        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                task_spec=task_spec,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            )
        expected_keys = set(task_spec.output_fields.keys())
        extra_keys = sorted(set(fields.keys()) - expected_keys)
        if extra_keys:
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                task_spec=task_spec,
                lm_response=completion,
                parsed_result=fields,
                message=f"unexpected field(s): {extra_keys}",
            )
        parsed_fields: dict[str, Any] = {}
        for k in expected_keys:
            if k not in fields:
                continue
            parsed_fields[k] = parse_output_field(
                adapter_name="JSONAdapter",
                task_spec=task_spec,
                field_name=k,
                raw_value=fields[k],
                lm_response=completion,
                field=task_spec.output_fields[k],
                repair=self.allow_json_repair,
            )
        validate_parsed_fields(
            adapter_name="JSONAdapter", task_spec=task_spec, lm_response=completion, fields=parsed_fields
        )
        return parsed_fields


def _extract_fenced_json(completion: str) -> str | None:
    match = regex.search(r"```(?:json)?\s*(.*?)\s*```", completion, regex.DOTALL | regex.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _extract_json_object(completion: str) -> str | None:
    pattern = r"\{(?:[^{}]|(?R))*\}"
    match = regex.search(pattern, completion, regex.DOTALL)
    if not match:
        return None
    return match.group(0)
