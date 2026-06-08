"""OpenAI content-block coercion for LM parts."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from dspy.core.types.parts.models import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMImagePart,
    LMOpaquePart,
    LMPart,
    LMTextPart,
    LMToolCallPart,
    LMVideoPart,
    _coerce_part,
)
from dspy.core.types.parts.serialize import _parse_json_object


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
