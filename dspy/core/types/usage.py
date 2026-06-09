from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def fill_aliases(self) -> LMUsage:
        if self.input_tokens is None and self.prompt_tokens is not None:
            self.input_tokens = self.prompt_tokens
        if self.output_tokens is None and self.completion_tokens is not None:
            self.output_tokens = self.completion_tokens
        if self.prompt_tokens is None and self.input_tokens is not None:
            self.prompt_tokens = self.input_tokens
        if self.completion_tokens is None and self.output_tokens is not None:
            self.completion_tokens = self.output_tokens
        if self.total_tokens is None and self.input_tokens is not None and (self.output_tokens is not None):
            self.total_tokens = self.input_tokens + self.output_tokens
        return self
