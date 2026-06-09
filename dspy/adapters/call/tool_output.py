from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.base.tool_calls import _provider_tool_call_to_tool_call_dict
from dspy.adapters.types.tool import ToolCalls

if TYPE_CHECKING:
    from dspy.core.types import LMOutput


def attach_tool_calls_to_value(
    *,
    value: dict[str, Any],
    output: LMOutput,
    tool_call_output_field_name: str | None,
) -> dict[str, Any]:
    tool_calls = output.tool_calls
    if not tool_calls or not tool_call_output_field_name:
        return value
    normalized = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
    value[tool_call_output_field_name] = ToolCalls.from_dict_list(normalized)
    return value
