"""Normalized LM types — normalized LM requests."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types.config import (
    LMConfig,
    LMToolSpec,
    _lm_config_data_from_kwargs,
    _merge_config_overrides,
    _merge_lm_config,
)
from dspy.core.types.conversation import _coerce_message, _coerce_tool_spec, _messages_from_items
from dspy.core.types.messages import LMMessage
from dspy.core.types.parts import LMPart


@dataclass
class LMRequestPatch:
    """A partial normalized LM request contributed while rendering a DSPy call.

    `LMRequest` is the complete object a `LanguageModel` receives. A patch is
    the smaller, composable unit that DSPy type strategies can contribute while
    an adapter is still building that request: extra messages, extra parts,
    native tools, native config, or signature fields that should be hidden from
    the outer adapter's ordinary text/JSON/XML rendering.
    """

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
        """Return a new patch containing this patch followed by `other`."""
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
    """A normalized request passed to a `LanguageModel`."""

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
        **kwargs: Any,
    ) -> LMRequest:
        if messages is not None and (items or prompt is not None):
            raise ValueError("Pass messages or direct-call inputs, not both.")

        collected_tools: list[Any] = list(tools or [])
        if messages is not None:
            normalized_messages = [_coerce_message(message) for message in messages]
        else:
            normalized_messages, positional_tools = _messages_from_items(items, prompt=prompt)
            collected_tools.extend(positional_tools)

        config = LMConfig(**_lm_config_data_from_kwargs(kwargs))
        return cls(
            model=model,
            messages=normalized_messages,
            tools=[_coerce_tool_spec(tool) for tool in collected_tools],
            config=config,
        )

    @classmethod
    def from_prompt_or_messages(
        cls,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict[str, Any] | LMMessage] | None = None,
        **kwargs: Any,
    ) -> LMRequest:
        return cls.from_call(model=model, prompt=prompt, messages=messages, **kwargs)

    def with_config_overrides(self, **kwargs: Any) -> LMRequest:
        """Return a copy with explicit request config overrides applied.

        Only fields implied by the supplied keyword arguments are changed. This
        preserves existing grouped config such as `cache`, `prompt_cache`,
        `tool_choice`, `reasoning`, `stop`, and provider-specific
        `extensions` when an unrelated setting is overridden.
        """
        if not kwargs:
            return self
        merged = _merge_config_overrides(self.config, kwargs)
        return self.model_copy(update={"config": merged}, deep=True)
