from __future__ import annotations

from typing import Any

from dspy.clients.openai_format.serialize import text_config_kwargs
from dspy.core.types import LMMessage, LMRequest, LMTextPart


def to_openai_text_request(request: LMRequest) -> dict[str, Any]:
    data = {"model": request.model, "prompt": messages_to_text_prompt(request.messages)}
    data.update(text_config_kwargs(request.config))
    return data


def messages_to_text_prompt(messages: list[LMMessage]) -> str:
    chunks = []
    for message in messages:
        texts = []
        for part in message.parts:
            if not isinstance(part, LMTextPart):
                raise ValueError(
                    f"OpenAI text completions only support text parts, but received {type(part).__name__}."
                )
            texts.append(part.text)
        chunks.append("".join(texts))
    return "\n\n".join(chunks + ["BEGIN RESPONSE:"])
