from __future__ import annotations

import json
from typing import Any, cast

import json_repair

from dspy.adapters.types.tool import ToolCalls
from dspy.core.types import LMToolCallPart
from dspy.task_spec.json_serialize import serialize_for_json


def _provider_value(value: object, key: str, default: object = None) -> object:
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
                args = json_repair.loads(raw_arguments)
        return {"id": tool_call.id, "name": tool_call.name, "args": args}
    function = _provider_value(value=tool_call, key="function", default={}) or {}
    arguments = _provider_value(value=function, key="arguments", default={})
    if isinstance(arguments, str):
        parsed_arguments = json_repair.loads(arguments)
    elif isinstance(arguments, dict):
        parsed_arguments = arguments
    else:
        parsed_arguments = {}
    return {
        "id": _provider_value(value=tool_call, key="id") or _provider_value(value=tool_call, key="call_id"),
        "name": _provider_value(value=function, key="name") or _provider_value(value=tool_call, key="name"),
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
