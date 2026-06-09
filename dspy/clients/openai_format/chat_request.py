from __future__ import annotations

from typing import Any

from dspy.clients.openai_format.serialize import (
    assistant_tool_call_to_openai,
    common_config_kwargs,
    parts_to_openai_content,
    tool_choice_to_openai,
    tool_result_to_openai,
    tool_to_openai,
)
from dspy.core.types.messages import LMMessage
from dspy.core.types.parts import LMToolCallPart, LMToolResultPart
from dspy.core.types.request import LMRequest


def message_to_openai_chat(message: LMMessage) -> dict[str, Any]:
    output: dict[str, Any] = {"role": message.role.value}
    if message.name is not None:
        output["name"] = message.name
    if message.role == "assistant":
        tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
        content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
        output["content"] = None if tool_calls and (not content_parts) else parts_to_openai_content(content_parts)
        if tool_calls:
            output["tool_calls"] = [assistant_tool_call_to_openai(part) for part in tool_calls]
        return output
    if message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
        result = message.parts[0]
        output.update(tool_result_to_openai(result))
        if result.call_id is not None:
            output["tool_call_id"] = result.call_id
        if result.name is not None:
            output["name"] = result.name
        return output
    output["content"] = parts_to_openai_content(message.parts)
    return output


def request_messages_as_openai(request: LMRequest) -> list[dict[str, Any]]:
    return [message_to_openai_chat(message) for message in request.messages]


def to_openai_chat_request(request: LMRequest) -> dict[str, Any]:
    data = {"model": request.model, "messages": request_messages_as_openai(request)}
    data.update(common_config_kwargs(request.config, model=request.model, endpoint="chat"))
    if request.config.tool_choice is not None:
        data.update(tool_choice_to_openai(request.config.tool_choice))
    if request.tools:
        data["tools"] = [tool_to_openai(tool) for tool in request.tools]
    return data
