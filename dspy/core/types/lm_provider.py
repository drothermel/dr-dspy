from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMProviderOptions(BaseModel):
    """LiteLLM / provider connection and passthrough options for BaseLM construction."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    api_base: str | None = None
    base_url: str | None = None
    cache: bool | None = None
    timeout: float | None = None
    max_retries: int | None = None
    custom_llm_provider: str | None = None
    model_list: list[dict[str, Any]] | None = None
    response_format: Any | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_base_url(self) -> LMProviderOptions:
        if self.base_url is not None and self.api_base is None:
            self.api_base = self.base_url
        return self

    def to_kwargs(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True, exclude={"extensions"})
        data.update(self.extensions)
        return data


def merge_provider_options(
    left: LMProviderOptions | None,
    right: LMProviderOptions | None,
) -> LMProviderOptions | None:
    if left is None:
        return right
    if right is None:
        return left
    merged = left.model_dump()
    merged.update(right.model_dump(exclude_none=True))
    return LMProviderOptions(**merged)
