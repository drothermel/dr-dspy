from __future__ import annotations

from typing import Any

from dspy.clients.openai_format.serialize import (
    assistant_tool_call_to_openai,
    parts_to_openai_content,
    responses_config_kwargs,
    responses_tool_output_text,
    tool_choice_to_openai,
    tool_result_to_openai,
    tool_to_openai,
)
from dspy.core.types import LMMessage, LMRequest, LMToolCallPart, LMToolResultPart


def to_openai_responses_request(request: LMRequest) -> dict[str, Any]:
    """Convert a normalized DSPy request into Responses API kwargs."""
    config = request.config
    data: dict[str, Any] = {
        "model": request.model,
        "input": [item for message in request.messages for item in message_to_responses_input_items(message)],
    }
    data.update(responses_config_kwargs(config, model=request.model))
    if config.tool_choice is not None:
        data.update(tool_choice_to_openai(config.tool_choice))
    if request.tools:
        data["tools"] = [tool_to_openai(tool) for tool in request.tools]
    return data


def message_to_responses_input_items(message: LMMessage) -> list[dict[str, Any]]:
    """Convert one DSPy message into one or more Responses input items."""
    if message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
        result = message.parts[0]
        item = {
            "type": "function_call_output",
            "output": responses_tool_output_text(tool_result_to_openai(result)["content"]),
        }
        if result.call_id is not None:
            item["call_id"] = result.call_id
        return [item]

    tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
    content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
    content = parts_to_responses_content(content_parts)
    items: list[dict[str, Any]] = []

    if content or message.role != "assistant" or not tool_calls:
        item: dict[str, Any] = {"role": message.role, "content": content}
        if message.name is not None:
            item["name"] = message.name
        items.append(item)

    if message.role == "assistant":
        items.extend(tool_call_to_responses_input(tool_call) for tool_call in tool_calls)
    return items


def parts_to_responses_content(parts: list[Any]) -> list[dict[str, Any]]:
    blocks = parts_to_openai_content(parts)
    if isinstance(blocks, str):
        return [{"type": "input_text", "text": blocks}]
    return [content_block_to_responses(block) for block in blocks]


def tool_call_to_responses_input(tool_call_part: LMToolCallPart) -> dict[str, Any]:
    tool_call = assistant_tool_call_to_openai(tool_call_part)
    function = tool_call.get("function", {})
    item = {"type": "function_call", "name": function.get("name", ""), "arguments": function.get("arguments", "{}")}
    call_id = tool_call.get("id") or tool_call.get("call_id")
    if call_id is not None:
        item["call_id"] = call_id
    return item


def content_block_to_responses(block: dict[str, Any]) -> dict[str, Any]:
    block_type = block.get("type")
    if block_type == "text":
        return {"type": "input_text", "text": block.get("text", "")}
    if block_type == "image_url":
        image_url = block.get("image_url", {})
        out = {"type": "input_image", "image_url": image_url.get("url", "")}
        if image_url.get("detail") is not None:
            out["detail"] = image_url["detail"]
        return out
    if block_type == "input_audio":
        return block
    if block_type == "file":
        file = block.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }
    return block
