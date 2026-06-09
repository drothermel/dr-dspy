from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMProviderOptions(BaseModel):
    """LiteLLM / provider connection and passthrough options for BaseLM construction.

    ``api_base`` is the canonical endpoint override. ``base_url`` is accepted as an
    alias and normalized to ``api_base`` when only ``base_url`` is set.
    """

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
        data = self.model_dump(exclude_none=True, exclude={"extensions", "cache"})
        data.update(self.extensions)
        return data


def merge_provider_options(
    left: LMProviderOptions | None,
    right: LMProviderOptions | None,
) -> LMProviderOptions | None:
    """Merge provider options using the same overlay rules as ``merge_lm_config``.

    ``extensions`` are union-merged (right keys override left); explicit ``None`` on
    right clears all extensions; an empty mapping on right preserves left keys.
    """
    if left is None:
        return right
    if right is None:
        return left
    data = left.model_dump(exclude_none=True)
    extensions = {**left.extensions}
    for key in right.model_fields_set:
        value = getattr(right, key)
        if key == "extensions":
            if value is None:
                extensions = {}
            elif isinstance(value, Mapping):
                extensions.update(value)
            else:
                extensions = dict(value)
            continue
        if isinstance(value, BaseModel):
            data[key] = value.model_dump(exclude_none=True)
        else:
            data[key] = value
    data["extensions"] = extensions
    return LMProviderOptions(**data)
