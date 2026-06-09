from __future__ import annotations

from typing import Any

import pydantic

from dspy.clients.media_uri import (
    data_uri,
    data_uri_from_path,
    media_type_for_path,
    read_path_base64,
    split_data_uri,
)
from dspy.core.types import LMAudioPart, LMBinaryPart, LMDocumentPart, LMImagePart, LMTextPart

__all__ = [
    "data_uri",
    "data_uri_from_path",
    "get_value",
    "media_format",
    "media_source",
    "media_type_for_path",
    "model_dump",
    "part_text",
    "read_path_base64",
    "split_data_uri",
]


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
            stack.extend([*item.__dict__.values(), *(item.__pydantic_extra__ or {}).values()])
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
