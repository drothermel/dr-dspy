from __future__ import annotations

from typing import Any

from dspy.core.types.messages import LMMessage, LMMessageRole
from dspy.core.types.parts import LMToolCallPart, LMToolResultPart
from dspy.core.types.parts.models import _coerce_part


def System(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    return LMMessage(
        role=LMMessageRole.SYSTEM, parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {}
    )


def Developer(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    return LMMessage(
        role=LMMessageRole.DEVELOPER,
        parts=[_coerce_part(part) for part in parts],
        name=name,
        metadata=metadata or {},
    )


def User(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    return LMMessage(
        role=LMMessageRole.USER, parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {}
    )


def Assistant(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    return LMMessage(
        role=LMMessageRole.ASSISTANT,
        parts=[_coerce_part(part) for part in parts],
        name=name,
        metadata=metadata or {},
    )


ToolCall = LMToolCallPart


def ToolResult(
    *parts: Any, call_id: str | None = None, name: str | None = None, content: Any | None = None, is_error: bool = False
) -> LMMessage:
    if content is not None:
        if parts:
            raise TypeError("Pass tool output either as positional parts or as `content=...`, not both.")
        parts = tuple(content if isinstance(content, list) else [content])
    if len(parts) == 1 and isinstance(parts[0], LMToolResultPart):
        result = parts[0]
    else:
        result = LMToolResultPart(
            call_id=call_id, name=name, content=[_coerce_part(part) for part in parts], is_error=is_error
        )
    return LMMessage(role=LMMessageRole.TOOL, parts=[result])
