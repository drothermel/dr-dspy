from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from dspy.core.types.messages import LMMessage, LMMessageRole
from dspy.core.types.parts.models import _coerce_part


@runtime_checkable
class LMResponseLike(Protocol):
    outputs: list[Any]


def _coerce_message(value: dict[str, Any] | LMMessage) -> LMMessage:
    if isinstance(value, LMMessage):
        return value
    from dspy.clients.openai_format.chat_request import message_from_openai_chat

    return message_from_openai_chat(value)


def _is_lm_response(value: Any) -> bool:
    return isinstance(value, LMResponseLike)


def _messages_from_response(response: LMResponseLike) -> list[LMMessage]:
    return [LMMessage(role=LMMessageRole.ASSISTANT, parts=output.parts) for output in response.outputs]


def _is_message_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, LMMessage) or _is_lm_response(item) for item in value
    )


def _messages_from_items(items: tuple[Any, ...], *, prompt: str | None = None) -> list[LMMessage]:
    if prompt is not None:
        items = (prompt, *items)
    if not items:
        items = ("",)
    if len(items) == 1 and _is_message_sequence(items[0]):
        items = tuple(items[0])

    if all(isinstance(item, LMMessage) or _is_lm_response(item) for item in items):
        messages: list[LMMessage] = []
        for item in items:
            if isinstance(item, LMMessage):
                messages.append(item)
            else:
                messages.extend(_messages_from_response(cast("LMResponseLike", item)))
        return messages

    parts = [_coerce_part(item) for item in items]
    return [LMMessage(role=LMMessageRole.USER, parts=parts)]
