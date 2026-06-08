"""Normalized LM types — message assembly helpers for requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.core.types.config import LMToolSpec
from dspy.core.types.messages import LMMessage
from dspy.core.types.parts import _coerce_part

if TYPE_CHECKING:
    from dspy.core.types.response import LMResponse


def _coerce_message(value: dict[str, Any] | LMMessage) -> LMMessage:
    if isinstance(value, LMMessage):
        return value
    return LMMessage(**value)


def _messages_from_items(items: tuple[Any, ...], *, prompt: str | None = None) -> tuple[list[LMMessage], list[Any]]:
    # TODO: Normalize DSPy-specific LM(...) objects in the LM call layer before building LMRequest.
    if prompt is not None:
        items = (prompt, *items)
    if not items:
        items = ("",)

    if len(items) == 1 and _is_message_sequence(items[0]):
        items = tuple(items[0])

    from dspy.core.types.response import LMResponse

    if all(isinstance(item, (LMMessage, LMResponse)) for item in items):
        messages: list[LMMessage] = []
        for item in items:
            if isinstance(item, LMMessage):
                messages.append(item)
            else:
                messages.extend(_messages_from_response(item))  # ty:ignore[invalid-argument-type]
        return messages, []

    parts = [_coerce_part(item) for item in items]
    return [LMMessage(role="user", parts=parts)], []


def _messages_from_response(response: LMResponse) -> list[LMMessage]:
    return [LMMessage(role="assistant", parts=output.parts) for output in response.outputs]


def _is_message_sequence(value: Any) -> bool:
    from dspy.core.types.response import LMResponse

    return isinstance(value, (list, tuple)) and all(isinstance(item, (LMMessage, LMResponse)) for item in value)


def _coerce_tool_spec(tool: Any) -> LMToolSpec:
    if isinstance(tool, LMToolSpec):
        return tool
    if hasattr(tool, "to_lm_tool_spec"):
        return tool.to_lm_tool_spec()
    if isinstance(tool, dict):
        if "function" in tool:
            function = tool["function"]
            provider_data = {key: value for key, value in tool.items() if key not in {"type", "function"}}
            return LMToolSpec(
                name=function.get("name"),
                description=function.get("description"),
                parameters=function.get("parameters", {}),
                provider_data=provider_data,
            )
        return LMToolSpec(**tool)
    raise TypeError(f"Cannot convert {type(tool)!r} to LMToolSpec.")
