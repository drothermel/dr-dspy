from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dspy.clients.lm import LM
from dspy.clients.openai_format import message_to_openai_chat
from dspy.core.types import LMRequest, LMResponse


def request_prompt(request: LMRequest) -> str | None:
    if len(request.messages) != 1:
        return None
    message = request.messages[0]
    return message.text


class SpyLM(LM):
    """LM test double that records typed requests and returns canned text."""

    def __init__(
        self,
        *args: Any,
        return_json: bool = False,
        response_text: str | Callable[[LMRequest], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []
        self.return_json = return_json
        self.response_text = response_text

    def forward(self, request: LMRequest) -> LMResponse:
        messages = [message_to_openai_chat(message) for message in request.messages]
        kwargs = {**self.kwargs, **request.config.model_dump(exclude_none=True)}
        self.calls.append({"prompt": request_prompt(request), "messages": messages, "kwargs": kwargs})

        if self.response_text is not None:
            text = self.response_text(request) if callable(self.response_text) else self.response_text
        elif self.return_json:
            text = "{'answer':'100%'}"
        else:
            text = "[[ ## answer ## ]]\n100%!"

        return LMResponse.from_text(text, model=request.model)
