from __future__ import annotations

import json
from typing import Any, cast

from dspy.clients.openai_binary import binary_to_openai
from dspy.core.types.parts import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMImagePart,
    LMOpaquePart,
    LMPart,
    LMTextPart,
    LMToolCallPart,
    LMToolResultPart,
    LMVideoPart,
    _split_data_uri,
)
from dspy.core.types.request import LMRequest


def _history_request_prompt(request: LMRequest) -> str | None:
    if len(request.messages) != 1:
        return None
    message = request.messages[0]
    if message.role != "user" or len(message.parts) != 1:
        return None
    part = message.parts[0]
    return part.text if isinstance(part, LMTextPart) else None


def _history_request_messages_as_openai(request: LMRequest) -> list[dict[str, Any]]:
    messages = []
    for message in request.messages:
        if message.role == "assistant":
            tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
            content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
            item: dict[str, Any] = {
                "role": "assistant",
                "content": _history_message_parts_as_openai_content(cast("list[LMPart]", content_parts))
                if content_parts
                else None,
            }
            if tool_calls:
                item["tool_calls"] = [_history_tool_call_as_openai(call) for call in tool_calls]
        elif message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
            result = message.parts[0]
            item = {"role": "tool", "content": _history_tool_result_content(result)}
            if result.call_id is not None:
                item["tool_call_id"] = result.call_id
            if result.name is not None:
                item["name"] = result.name
        else:
            item = {"role": message.role, "content": _history_message_parts_as_openai_content(message.parts)}
        if message.name is not None and "name" not in item:
            item["name"] = message.name
        messages.append(item)
    return messages


def _history_tool_call_as_openai(call: LMToolCallPart) -> dict[str, Any]:
    data: dict[str, Any] = {"type": "function", "function": {"name": call.name, "arguments": json.dumps(call.args)}}
    if call.id is not None:
        data["id"] = call.id
    return data


def _history_tool_result_content(result: LMToolResultPart) -> str:
    chunks = []
    for part in result.content:
        if isinstance(part, LMTextPart):
            chunks.append(part.text)
        else:
            chunks.append(json.dumps(part.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
    return "".join(chunks)


def _history_message_parts_as_openai_content(parts: list[LMPart]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and isinstance(parts[0], LMTextPart):
        return parts[0].text
    return [_history_part_as_openai_content(part) for part in parts]


def _history_part_as_openai_content(part: LMPart) -> dict[str, Any]:
    if isinstance(part, LMOpaquePart):
        return dict(part.block)
    if isinstance(part, LMTextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, LMImagePart):
        return {"type": "image_url", "image_url": {"url": _history_part_source(part)}}
    if isinstance(part, LMAudioPart):
        input_audio: dict[str, Any] = {"format": _history_media_format(part.media_type)}
        if part.data is not None:
            if part.data.startswith("data:"):
                media_type, data = _split_data_uri(part.data)
                input_audio["format"] = _history_media_format(media_type)
                input_audio["data"] = data
            else:
                input_audio["data"] = part.data
        elif part.url is not None:
            input_audio["url"] = part.url
        elif part.file_id is not None:
            input_audio["file_id"] = part.file_id
        elif part.path is not None:
            input_audio["path"] = part.path
        return {"type": "input_audio", "input_audio": input_audio}
    if isinstance(part, LMVideoPart):
        video: dict[str, Any] = {"media_type": part.media_type}
        if part.data is not None:
            video["data"] = _history_part_source(part)
        elif part.url is not None:
            video["url"] = part.url
        elif part.file_id is not None:
            video["file_id"] = part.file_id
        elif part.path is not None:
            video["path"] = part.path
        return {"type": "video", "video": video}
    if isinstance(part, LMDocumentPart):
        data: dict[str, Any] = {"type": "document"}
        if part.source is not None:
            data["source"] = part.source
        else:
            data["source"] = _history_part_source(part)
            data["media_type"] = part.media_type
        if part.citations:
            data["citations"] = part.citations
        if part.title is not None:
            data["title"] = part.title
        if part.context is not None:
            data["context"] = part.context
        return data
    if isinstance(part, LMBinaryPart):
        return binary_to_openai(part)
    return part.model_dump(exclude_none=True)


def _history_part_source(part: LMImagePart | LMAudioPart | LMVideoPart | LMDocumentPart | LMBinaryPart) -> str | None:
    if part.data is not None:
        return part.data if part.data.startswith("data:") else f"data:{part.media_type};base64,{part.data}"
    return part.url or part.file_id or part.path


def _history_media_format(media_type: str) -> str:
    return media_type.split("/", 1)[1] if "/" in media_type else media_type


def _history_request_kwargs(request: LMRequest) -> dict[str, Any]:
    data = request.config.model_dump(exclude_none=True)
    extensions = data.pop("extensions", {}) or {}
    return {**extensions, **data}
