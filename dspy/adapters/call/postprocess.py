from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from dspy.adapters.base.tool_calls import attach_tool_calls_to_parsed_value
from dspy.adapters.types.base_type import Type

if TYPE_CHECKING:
    from dspy.core.types import LMOutput
    from dspy.task_spec import TaskSpec


class PostprocessEnrichmentAdapter(Protocol):
    native_response_types: list[type[Type]]

    def _get_tool_call_output_field_name(self, task_spec: TaskSpec) -> str | None: ...


def strip_native_response_output_fields(task_spec: TaskSpec, native_response_types: list[type[Type]]) -> TaskSpec:
    for name, field in task_spec.output_fields.items():
        field_type = field.type_
        if (
            isinstance(field_type, type)
            and field_type in native_response_types
            and issubclass(field_type, Type)
        ):
            task_spec = task_spec.delete(name)
    return task_spec


def enrich_parsed_value_from_lm_output(
    adapter: PostprocessEnrichmentAdapter,
    *,
    value: dict[str, Any],
    output: LMOutput,
    original_task_spec: TaskSpec,
) -> dict[str, Any]:
    tool_call_output_field_name = adapter._get_tool_call_output_field_name(original_task_spec)
    for field_name in original_task_spec.output_fields:
        value.setdefault(field_name, None)
    value = attach_tool_calls_to_parsed_value(
        value=value,
        output=output,
        tool_call_output_field_name=tool_call_output_field_name,
    )
    for name, field in original_task_spec.output_fields.items():
        field_type = field.type_
        if (
            isinstance(field_type, type)
            and field_type in adapter.native_response_types
            and issubclass(field_type, Type)
        ):
            parsed_value = field_type.parse_lm_output(output)
            if parsed_value is not None:
                value[name] = parsed_value
    if output.logprobs is not None:
        value["logprobs"] = output.logprobs
    return value
