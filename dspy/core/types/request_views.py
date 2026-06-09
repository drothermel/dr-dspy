from __future__ import annotations

from typing import Any

from dspy.core.types.parts.models import LMTextPart
from dspy.core.types.request import LMRequest


def request_prompt(request: LMRequest) -> str | None:
    if len(request.messages) != 1:
        return None
    message = request.messages[0]
    if message.role != "user" or len(message.parts) != 1:
        return None
    part = message.parts[0]
    return part.text if isinstance(part, LMTextPart) else None


def request_kwargs(request: LMRequest) -> dict[str, Any]:
    data = request.config.model_dump(exclude_none=True)
    extensions = data.pop("extensions", {}) or {}
    return {**extensions, **data}
