from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.types.tool.tool_calls import normalize_tool_call_dict
from dspy.adapters.utils.json_loads import load_json
from dspy.core.types import LMToolCallPart
from dspy.serialization.json import to_jsonable

if TYPE_CHECKING:
    from dspy.core.types import LMOutput


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
                args = load_json(raw_arguments, repair=True)
                if not isinstance(args, dict):
                    args = {}
        return normalize_tool_call_dict(
            {"id": tool_call.id, "name": tool_call.name, "args": args},
            repair=True,
        )
    function = read_provider_field(value=tool_call, key="function", default={}) or {}
    arguments = read_provider_field(value=function, key="arguments", default={})
    parsed_arguments = arguments if isinstance(arguments, (str, dict)) else {}
    return normalize_tool_call_dict(
        {
            "id": read_provider_field(value=tool_call, key="id") or read_provider_field(value=tool_call, key="call_id"),
            "name": read_provider_field(value=function, key="name") or read_provider_field(value=tool_call, key="name"),
            "args": parsed_arguments,
        },
        repair=True,
    )


def _tool_calls_from_message(message: dict[str, Any]) -> tuple[str | None, ToolCalls | None]:
    for name, value in message.items():
        if isinstance(value, ToolCalls) or (isinstance(value, dict) and "tool_calls" in value):
            return (name, ToolCalls.model_validate(value))
    return (None, None)


def _tool_result_content(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(to_jsonable(cast("Any", value)), ensure_ascii=False)


def _tool_call_as_openai_message_tool_call(tool_call: ToolCalls.ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(to_jsonable(tool_call.args), ensure_ascii=False),
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
