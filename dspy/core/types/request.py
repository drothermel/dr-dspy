from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types.lm_config import LMConfig
from dspy.core.types.message_coercion import _coerce_message, _messages_from_items
from dspy.core.types.messages import LMMessage
from dspy.core.types.tool_spec import LMToolSpec, coerce_tool_spec


class LMRequest(BaseModel):
    """Provider request payload."""

    model: str
    messages: list[LMMessage]
    tools: list[LMToolSpec] = Field(default_factory=list)
    config: LMConfig = Field(default_factory=LMConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def from_call(
        cls,
        *,
        model: str,
        items: tuple[Any, ...] = (),
        prompt: str | None = None,
        messages: list[dict[str, Any] | LMMessage] | None = None,
        tools: list[Any] | None = None,
        config: LMConfig | None = None,
    ) -> LMRequest:
        if messages is not None and (items or prompt is not None):
            raise ValueError("Pass messages or direct-call inputs, not both.")
        collected_tools: list[Any] = list(tools or [])
        if messages is not None:
            normalized_messages = [_coerce_message(message) for message in messages]
        else:
            normalized_messages = _messages_from_items(items, prompt=prompt)
        return cls(
            model=model,
            messages=normalized_messages,
            tools=[coerce_tool_spec(tool) for tool in collected_tools],
            config=config or LMConfig(),
        )
