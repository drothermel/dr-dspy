from __future__ import annotations

import base64
import mimetypes
from typing import Any

import pydantic

from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMImagePart,
    LMTextPart,
)


def media_source(part: LMImagePart | LMAudioPart | LMDocumentPart | LMBinaryPart) -> str:
    if part.data is not None:
        return data_uri(media_type=part.media_type, data=part.data)
    if part.url is not None:
        return part.url
    if part.file_id is not None:
        return part.file_id
    if part.path is not None:
        return data_uri_from_path(part.path, fallback_media_type=part.media_type)
    raise ValueError(f"{type(part).__name__} has no media source.")


def read_path_base64(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def media_type_for_path(path: str, *, fallback: str) -> str:
    return mimetypes.guess_type(path)[0] or fallback


def data_uri_from_path(path: str, *, fallback_media_type: str) -> str:
    return data_uri(
        media_type=media_type_for_path(path, fallback=fallback_media_type),
        data=read_path_base64(path),
    )


def data_uri(media_type: str, data: str) -> str:
    if data.startswith("data:"):
        return data
    return f"data:{media_type};base64,{data}"


def split_data_uri(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        return "application/octet-stream", value
    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type, data


def media_format(media_type: str) -> str:
    format_ = media_type.split("/", 1)[1] if "/" in media_type else media_type
    return {"x-wav": "wav", "mpeg": "mp3"}.get(format_, format_)


def part_text(value: Any) -> str:
    return value.text if isinstance(value, LMTextPart) else str(value)


def get_value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _rebuild_pydantic_serializers(value: Any) -> None:
    stack = [value]
    seen = set()
    for item in stack:
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, pydantic.BaseModel):
            type(item).model_rebuild(force=True, raise_errors=False)
            stack.extend([*item.__dict__.values(), *((item.__pydantic_extra__ or {}).values())])
        elif isinstance(item, (dict, list, tuple)):
            stack.extend(item.values() if isinstance(item, dict) else item)


def model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=True)
        except TypeError as error:
            if "MockValSer" not in str(error):
                raise
            _rebuild_pydantic_serializers(value)
            return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    data = {}
    for key in (
        "id",
        "call_id",
        "type",
        "name",
        "arguments",
        "status",
        "text",
        "refusal",
        "url",
        "data",
        "file_id",
        "filename",
        "media_type",
        "mime_type",
    ):
        item = getattr(value, key, None)
        if item is not None:
            data[key] = item
    return data
