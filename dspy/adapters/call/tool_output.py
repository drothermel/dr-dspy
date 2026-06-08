from __future__ import annotations

from typing import Any

import json_repair

from dspy.adapters.base.tool_calls import _provider_tool_call_to_tool_call_dict
from dspy.adapters.types.tool import ToolCalls
from dspy.core.types import LMOutput, LMToolCallPart
from dspy.task_spec import TaskSpec


def attach_tool_calls_to_value(
    *,
    value: dict[str, Any],
    output: LMOutput,
    original_task_spec: TaskSpec,
    get_tool_call_output_field_name: Any,
) -> dict[str, Any]:
    tool_call_output_field_name = get_tool_call_output_field_name(original_task_spec)
    tool_calls = output.tool_calls
    if not tool_calls or not tool_call_output_field_name:
        return value

    normalized_tool_calls = []
    for tool_call in tool_calls:
        if isinstance(tool_call, LMToolCallPart):
            normalized_tool_calls.append({"name": tool_call.name, "args": dict(tool_call.args), "id": tool_call.id})
        else:
            normalized_tool_calls.append(
                {
                    "name": tool_call["function"]["name"],
                    "args": json_repair.loads(tool_call["function"]["arguments"]),
                }
            )
    value[tool_call_output_field_name] = ToolCalls.from_dict_list(normalized_tool_calls)
    return value


def attach_provider_tool_calls_to_value(
    *,
    value: dict[str, Any],
    output: LMOutput,
    original_task_spec: TaskSpec,
    get_tool_call_output_field_name: Any,
) -> dict[str, Any]:
    tool_call_output_field_name = get_tool_call_output_field_name(original_task_spec)
    tool_calls = output.tool_calls
    if not tool_calls or not tool_call_output_field_name:
        return value
    normalized = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
    value[tool_call_output_field_name] = ToolCalls.from_dict_list(normalized)
    return value
