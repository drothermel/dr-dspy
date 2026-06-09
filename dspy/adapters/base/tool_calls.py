from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils.json_loads import load_json
from dspy.core.types import LMToolCallPart
from dspy.task_spec.json_serialize import serialize_for_json

if TYPE_CHECKING:
    from dspy.core.types import LMOutput


def _load_provider_tool_arguments(arguments: str) -> dict[str, Any]:
    """Parse provider tool-call argument strings (external wire data; repair enabled)."""
    try:
        parsed = load_json(arguments, repair=False)
    except json.JSONDecodeError:
        parsed = load_json(arguments, repair=True)
    return parsed if isinstance(parsed, dict) else {}


def read_provider_field(value: object, key: str, default: object = None) -> object:
    """Read a field from provider tool-call payloads (OpenAI dict or object shapes)."""
    if isinstance(value, dict):
        value = cast("dict[str, object]", value)
        return value.get(key, default)
    return getattr(value, key, default)


def _provider_tool_call_to_tool_call_dict(tool_call: object) -> dict[str, Any]:
    if isinstance(tool_call, LMToolCallPart):
        args = dict(tool_call.args)
        if not args:
            raw_arguments = tool_call.provider_data.get("raw_arguments") or tool_call.provider_data.get("arguments")
            if isinstance(raw_arguments, str):
                args = _load_provider_tool_arguments(raw_arguments)
        return {"id": tool_call.id, "name": tool_call.name, "args": args}
    function = read_provider_field(value=tool_call, key="function", default={}) or {}
    arguments = read_provider_field(value=function, key="arguments", default={})
    if isinstance(arguments, str):
        parsed_arguments = _load_provider_tool_arguments(arguments)
    elif isinstance(arguments, dict):
        parsed_arguments = arguments
    else:
        parsed_arguments = {}
    return {
        "id": read_provider_field(value=tool_call, key="id") or read_provider_field(value=tool_call, key="call_id"),
        "name": read_provider_field(value=function, key="name") or read_provider_field(value=tool_call, key="name"),
        "args": parsed_arguments,
    }


def _tool_calls_from_message(message: dict[str, Any]) -> tuple[str | None, ToolCalls | None]:
    for name, value in message.items():
        if isinstance(value, ToolCalls) or (isinstance(value, dict) and "tool_calls" in value):
            return (name, ToolCalls.model_validate(value))
    return (None, None)


def _tool_result_content(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(serialize_for_json(cast("Any", value)), ensure_ascii=False)


def _tool_call_as_openai_message_tool_call(tool_call: ToolCalls.ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(serialize_for_json(tool_call.args), ensure_ascii=False),
        },
    }


def attach_tool_calls_to_parsed_value(
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
