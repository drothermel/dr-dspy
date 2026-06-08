from typing import Any

from dspy.core.types import LMMessage


def build_lm_message(
    role: str,
    content: str | list[dict[str, Any]] | None = None,
    **extra: Any,
) -> LMMessage:
    payload: dict[str, Any] = {"role": role}
    if content is not None:
        payload["content"] = content
    payload.update(extra)
    return LMMessage(**payload)
