from typing import Any

from dspy.clients.openai_format.chat_request import message_from_openai_chat
from dspy.core.types import LMMessage


def build_lm_message(role: str, content: str | list[dict[str, Any]] | None = None, **extra: Any) -> LMMessage:
    payload: dict[str, Any] = {"role": role}
    if content is not None:
        payload["content"] = content
    payload.update(extra)
    return message_from_openai_chat(payload)
