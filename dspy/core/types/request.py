from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types.coercion import _coerce_message, _coerce_tool_spec, _messages_from_items
from dspy.core.types.config import (
    LMConfig,
    LMToolSpec,
    _merge_config_overrides,
    _merge_lm_config,
)
from dspy.core.types.messages import LMMessage
from dspy.core.types.parts import LMPart


@dataclass
class LMRequestPatch:
    messages: list[LMMessage] = dataclass_field(default_factory=list)
    system_parts: list[LMPart] = dataclass_field(default_factory=list)
    user_parts: list[LMPart] = dataclass_field(default_factory=list)
    assistant_parts: list[LMPart] = dataclass_field(default_factory=list)
    tools: list[LMToolSpec] = dataclass_field(default_factory=list)
    config: LMConfig | None = None
    delete_input_fields: tuple[str, ...] = ()
    delete_output_fields: tuple[str, ...] = ()
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def merge(self, other: LMRequestPatch) -> LMRequestPatch:
        return LMRequestPatch(
            messages=[*self.messages, *other.messages],
            system_parts=[*self.system_parts, *other.system_parts],
            user_parts=[*self.user_parts, *other.user_parts],
            assistant_parts=[*self.assistant_parts, *other.assistant_parts],
            tools=[*self.tools, *other.tools],
            config=_merge_lm_config(self.config, other.config),
            delete_input_fields=(*self.delete_input_fields, *other.delete_input_fields),
            delete_output_fields=(*self.delete_output_fields, *other.delete_output_fields),
            metadata={**self.metadata, **other.metadata},
        )


class LMRequest(BaseModel):
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
            normalized_messages, positional_tools = _messages_from_items(items, prompt=prompt)
            collected_tools.extend(positional_tools)
        return cls(
            model=model,
            messages=normalized_messages,
            tools=[_coerce_tool_spec(tool) for tool in collected_tools],
            config=config or LMConfig(),
        )

    @classmethod
    def from_prompt_or_messages(
        cls,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict[str, Any] | LMMessage] | None = None,
        config: LMConfig | None = None,
    ) -> LMRequest:
        return cls.from_call(model=model, prompt=prompt, messages=messages, config=config)

    def with_config_overrides(self, config: LMConfig) -> LMRequest:
        merged = _merge_config_overrides(self.config, config.model_dump(exclude_none=True))
        return self.model_copy(update={"config": merged}, deep=True)
