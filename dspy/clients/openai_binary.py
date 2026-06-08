from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any

from dspy.core.types.parts import LMBinaryPart


def _data_uri(media_type: str, data: str) -> str:
    if data.startswith("data:"):
        return data
    return f"data:{media_type};base64,{data}"


def _data_uri_from_path(path: str, *, fallback_media_type: str) -> str:
    media_type = mimetypes.guess_type(path)[0] or fallback_media_type
    with open(path, "rb") as file:
        data = base64.b64encode(file.read()).decode("ascii")
    return _data_uri(media_type=media_type, data=data)


def binary_to_openai(binary: LMBinaryPart) -> dict[str, Any]:
    file_data: dict[str, Any] = {}
    if binary.data is not None:
        file_data["file_data"] = _data_uri(media_type=binary.media_type, data=binary.data)
    elif binary.path is not None:
        file_data["file_data"] = _data_uri_from_path(binary.path, fallback_media_type=binary.media_type)
        file_data["filename"] = binary.filename or os.path.basename(binary.path)
    elif binary.url is not None:
        file_data["file_data"] = binary.url
    if binary.file_id is not None:
        file_data["file_id"] = binary.file_id
    if binary.filename is not None:
        file_data["filename"] = binary.filename
    return {"type": "file", "file": file_data}
