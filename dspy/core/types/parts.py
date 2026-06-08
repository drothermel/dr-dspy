"""Normalized LM types — content parts and part-level coercion."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMBasePart(BaseModel):
    """A single content item in an LM message or output."""

    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class LMSourcePart(LMBasePart):
    """Content from exactly one provider-addressable source."""

    media_type: str
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> LMSourcePart:
        _validate_one_source(self)
        return self


class LMTextPart(LMBasePart):
    """Text content."""

    type: Literal["text"] = "text"
    text: str


class LMImagePart(LMSourcePart):
    """Image content from data, a URL, a file ID, or a local path."""

    type: Literal["image"] = "image"
    media_type: str = "image/png"
    detail: Literal["low", "high", "auto"] | None = None


class LMAudioPart(LMSourcePart):
    """Audio content from data, a URL, a file ID, or a local path."""

    type: Literal["audio"] = "audio"
    media_type: str = "audio/wav"


class LMVideoPart(LMSourcePart):
    """Video content from data, a URL, a file ID, or a local path."""

    type: Literal["video"] = "video"
    media_type: str = "video/mp4"


class LMDocumentPart(LMBasePart):
    """Semantic source/document content, optionally citation-enabled.

    Documents are source material: text, PDFs, reports, contracts, or other
    provider-addressable evidence that benefits from title/context/citation
    semantics. Use `LMBinaryPart` for opaque attachments or arbitrary bytes.
    """

    type: Literal["document"] = "document"
    media_type: str = "application/pdf"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None
    source: dict[str, Any] | None = None
    citations: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    context: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> LMDocumentPart:
        has_media_source = any(value is not None for value in (self.data, self.url, self.file_id, self.path))
        if self.source is not None and has_media_source:
            raise ValueError("LMDocumentPart accepts either source or one of data, url, file_id, or path, not both.")
        if self.source is None:
            _validate_one_source(self)
        elif not self.source:
            raise ValueError("LMDocumentPart.source must be non-empty when provided.")
        return self


class LMBinaryPart(LMSourcePart):
    """Opaque binary content from data, a URL, a file ID, or a local path."""

    type: Literal["binary"] = "binary"
    media_type: str = "application/octet-stream"
    filename: str | None = None


class LMToolCallPart(LMBasePart):
    """A model request to call a tool.

    Use `ToolCall(...)` as a shorter alias when constructing
    assistant messages by hand.

    Args:
        id: Provider call ID, when the backend uses one.
        name: Name of the tool to call.
        args: JSON-like arguments for the tool.
        provider_data: Raw provider fields to keep with the tool call.

    Examples:
        ```python
        from dspy.core.types import Assistant, ToolCall

        assistant = Assistant(
            ToolCall(
                id="call_1",
                name="search",
                args={"query": "DSPy"},
            )
        )
        ```
    """

    type: Literal["tool_call"] = "tool_call"
    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)


class LMToolResultPart(LMBasePart):
    """A tool execution result sent back to a model."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str | None = None
    name: str | None = None
    content: list[LMPart] = Field(default_factory=list)
    is_error: bool = False
    provider_data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_content(cls, data: Any) -> Any:
        if isinstance(data, dict) and "content" in data:
            data = dict(data)
            content = data["content"]
            if content is None:
                data["content"] = []
            elif isinstance(content, list):
                data["content"] = [_coerce_part(item) for item in content]
            else:
                data["content"] = [_coerce_part(content)]
        return data


class LMThinkingPart(LMBasePart):
    """Reasoning or thinking content returned by a model."""

    type: Literal["thinking"] = "thinking"
    text: str
    redacted: bool = False


class LMCitationPart(LMBasePart):
    """A source citation returned by a model."""

    type: Literal["citation"] = "citation"
    text: str | None = None
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_has_content(self) -> LMCitationPart:
        if self.text is None and self.title is None and self.url is None:
            raise ValueError("LMCitationPart requires at least one of text, title, or url.")
        return self


class LMRefusalPart(LMBasePart):
    """A model refusal."""

    type: Literal["refusal"] = "refusal"
    text: str


class LMOpaquePart(LMBasePart):
    """Provider-native content block preserved verbatim for round-trip."""

    type: Literal["opaque"] = "opaque"
    block: dict[str, Any]


LMPart = Annotated[
    LMTextPart
    | LMImagePart
    | LMAudioPart
    | LMVideoPart
    | LMDocumentPart
    | LMBinaryPart
    | LMToolCallPart
    | LMToolResultPart
    | LMThinkingPart
    | LMCitationPart
    | LMRefusalPart
    | LMOpaquePart,
    Field(discriminator="type"),
]


def _validate_one_source(part: Any) -> None:
    sources = {
        name: getattr(part, name) for name in ("data", "url", "file_id", "path") if getattr(part, name) is not None
    }
    class_name = type(part).__name__
    if len(sources) != 1:
        raise ValueError(f"{class_name} requires exactly one of data, url, file_id, or path.")
    name, value = next(iter(sources.items()))
    if isinstance(value, str) and not value:
        raise ValueError(f"{class_name}.{name} must be non-empty.")


def _coerce_part(value: Any) -> LMPart:
    if isinstance(
        value,
        (
            LMTextPart,
            LMImagePart,
            LMAudioPart,
            LMVideoPart,
            LMDocumentPart,
            LMBinaryPart,
            LMToolCallPart,
            LMToolResultPart,
            LMThinkingPart,
            LMCitationPart,
            LMRefusalPart,
            LMOpaquePart,
        ),
    ):
        return value
    if isinstance(value, str):
        return LMTextPart(text=value)
    if isinstance(value, dict) and "type" in value:
        return pydantic.TypeAdapter(LMPart).validate_python(value)
    raise TypeError(f"Cannot convert {type(value)!r} to an LMPart.")


def _parts_from_openai_content(content: Any) -> list[LMPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [LMTextPart(text=content)]
    if not isinstance(content, list):
        return [_coerce_part(content)]

    parts = []
    for item in content:
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type == "text":
            parts.append(LMTextPart(text=item.get("text", ""), metadata=item.get("metadata", {}) or {}))
        elif item_type == "image_url":
            image = item.get("image_url", {})
            if not isinstance(image, dict):
                raise TypeError("Image content block must be a mapping.")
            url = image.get("url")
            if url is None:
                raise ValueError("Image content block requires url.")
            parts.append(_image_source_to_part(url))
        elif item_type == "input_audio":
            audio = item.get("input_audio", {})
            parts.append(_audio_dict_to_part(audio))
        elif item_type == "file":
            parts.append(_binary_dict_to_part(item.get("file", {})))
        elif item_type == "document":
            parts.append(_document_dict_to_part(item))
        elif item_type == "video":
            video = item.get("video", {})
            parts.append(_media_dict_to_video_part(video))
        elif isinstance(item, dict):
            parts.append(LMOpaquePart(block=item))
        else:
            parts.append(_coerce_part(item))
    return parts


def _tool_calls_from_openai(tool_calls: list[Any]) -> list[LMToolCallPart]:
    return [_tool_call_from_openai(tool_call) for tool_call in tool_calls]


def _tool_call_from_openai(tool_call: Any) -> LMToolCallPart:
    if not isinstance(tool_call, Mapping):
        part = _coerce_part(tool_call)
        if isinstance(part, LMToolCallPart):
            return part
        raise TypeError(f"Cannot convert {type(tool_call)!r} to an LMToolCallPart.")

    function = tool_call.get("function", {})
    if not isinstance(function, Mapping):
        function = {}

    args = function.get("arguments", {})
    if isinstance(args, str):
        args = _parse_json_object(args)
    elif isinstance(args, Mapping):
        args = dict(args)
    else:
        args = {}

    return LMToolCallPart(
        id=tool_call.get("id"),
        name=function.get("name") or tool_call.get("name") or "",
        args=args,
    )


def _image_source_to_part(source: str) -> LMImagePart:
    if not isinstance(source, str):
        raise TypeError("Image URL must be a string.")
    if source.startswith("data:"):
        media_type, data = _split_data_uri(source)
        return LMImagePart(data=data, media_type=media_type)
    media_type = mimetypes.guess_type(urlparse(source).path)[0] or "image/png"
    return LMImagePart(url=source, media_type=media_type)


def _audio_dict_to_part(audio: dict[str, Any]) -> LMAudioPart:
    if not isinstance(audio, dict):
        raise TypeError("Audio content block must be a mapping.")

    audio_format = audio.get("format") or "wav"
    if not isinstance(audio_format, str):
        raise TypeError("Audio format must be a string.")
    media_type = audio_format if "/" in audio_format else f"audio/{audio_format}"

    if audio.get("data") is not None:
        data = audio["data"]
        if isinstance(data, str) and data.startswith("data:"):
            media_type, data = _split_data_uri(data)
        return LMAudioPart(data=data, media_type=media_type)
    if audio.get("url") is not None:
        return LMAudioPart(url=audio["url"], media_type=media_type)
    if audio.get("file_id") is not None:
        return LMAudioPart(file_id=audio["file_id"], media_type=media_type)
    if audio.get("path") is not None:
        return LMAudioPart(path=audio["path"], media_type=media_type)
    raise ValueError("Audio content block requires data, url, file_id, or path.")


def _binary_dict_to_part(file: dict[str, Any]) -> LMBinaryPart:
    if file.get("file_data") is not None:
        media_type, data = _split_data_uri(file["file_data"])
        return LMBinaryPart(data=data, media_type=media_type, filename=file.get("filename"))
    if file.get("data") is not None:
        media_type, data = _split_data_uri(file["data"])
        return LMBinaryPart(data=data, media_type=media_type, filename=file.get("filename"))
    if file.get("file_id") is not None:
        return LMBinaryPart(file_id=file["file_id"], filename=file.get("filename"))
    raise ValueError("Binary content block requires data, file_data, or file_id.")


def _document_dict_to_part(item: dict[str, Any]) -> LMDocumentPart:
    common = {"title": item.get("title"), "context": item.get("context")}
    media_type = item.get("media_type") or "application/pdf"
    for source_key in ("data", "url", "file_id", "path"):
        if item.get(source_key) is not None:
            return LMDocumentPart(**{source_key: item[source_key]}, media_type=media_type, **common)

    source = item.get("source")
    if isinstance(source, dict):
        return LMDocumentPart(
            source=source,
            citations=item.get("citations") or {},
            **common,  # ty:ignore[invalid-argument-type]
        )
    if isinstance(source, str):
        kwargs = _media_source_kwargs(source, default_media_type=media_type)
        return LMDocumentPart(**kwargs, **common)  # ty:ignore[invalid-argument-type]
    raise ValueError("Document content block requires source.")


def _media_dict_to_video_part(video: dict[str, Any]) -> LMVideoPart:
    if video.get("data") is not None:
        data = video["data"]
        if isinstance(data, str) and data.startswith("data:"):
            media_type, data = _split_data_uri(data)
        else:
            media_type = video.get("media_type") or "video/mp4"
        return LMVideoPart(data=data, media_type=media_type)
    if video.get("url") is not None:
        return LMVideoPart(url=video["url"], media_type=video.get("media_type") or "video/mp4")
    if video.get("file_id") is not None:
        return LMVideoPart(file_id=video["file_id"], media_type=video.get("media_type") or "video/mp4")
    if video.get("path") is not None:
        return LMVideoPart(path=video["path"], media_type=video.get("media_type") or "video/mp4")
    raise ValueError("Video content block requires data, url, file_id, or path.")


def _split_data_uri(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        return "application/octet-stream", value
    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type, data


def _media_source_kwargs(source: str, *, default_media_type: str) -> dict[str, str]:
    if source.startswith("data:"):
        media_type, data = _split_data_uri(source)
        return {"data": data, "media_type": media_type}

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        media_type = mimetypes.guess_type(parsed.path)[0] or default_media_type
        return {"url": source, "media_type": media_type}

    return {"file_id": source, "media_type": default_media_type}


def _part_to_value(part: LMPart) -> Any:
    if isinstance(part, LMTextPart):
        return part.text
    if isinstance(part, LMThinkingPart):
        return part
    if isinstance(part, LMToolCallPart):
        return part
    if isinstance(part, LMRefusalPart):
        return part.text
    return part


def _finalize_stream_part(part: LMPart) -> LMPart:
    if isinstance(part, LMToolCallPart) and "args_buffer" in part.provider_data:
        return part.model_copy(update={"args": _parse_json_object_strict(part.provider_data["args_buffer"])})
    return part


def _tool_call_to_provider_dict(call: LMToolCallPart) -> dict[str, Any]:
    data = {
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.args),
        },
    }
    if call.id is not None:
        data["id"] = call.id
    return data


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_object_strict(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Streamed tool-call arguments must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Streamed tool-call arguments must be a JSON object.")
    return parsed
