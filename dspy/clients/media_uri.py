from __future__ import annotations

import base64
import mimetypes


def data_uri(media_type: str, data: str) -> str:
    if data.startswith("data:"):
        return data
    return f"data:{media_type};base64,{data}"


def read_path_base64(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def media_type_for_path(path: str, *, fallback: str) -> str:
    return mimetypes.guess_type(path)[0] or fallback


def data_uri_from_path(path: str, *, fallback_media_type: str) -> str:
    return data_uri(media_type=media_type_for_path(path, fallback=fallback_media_type), data=read_path_base64(path))


def split_data_uri(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        return ("application/octet-stream", value)
    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return (media_type, data)
