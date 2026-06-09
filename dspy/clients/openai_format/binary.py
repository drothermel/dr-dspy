from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from dspy.clients.media_uri import data_uri, data_uri_from_path

if TYPE_CHECKING:
    from dspy.core.types.parts import LMBinaryPart


def binary_to_openai(binary: LMBinaryPart) -> dict[str, Any]:
    file_data: dict[str, Any] = {}
    if binary.data is not None:
        file_data["file_data"] = data_uri(media_type=binary.media_type, data=binary.data)
    elif binary.path is not None:
        file_data["file_data"] = data_uri_from_path(binary.path, fallback_media_type=binary.media_type)
        file_data["filename"] = binary.filename or os.path.basename(binary.path)
    elif binary.url is not None:
        file_data["file_data"] = binary.url
    if binary.file_id is not None:
        file_data["file_id"] = binary.file_id
    if binary.filename is not None:
        file_data["filename"] = binary.filename
    return {"type": "file", "file": file_data}
