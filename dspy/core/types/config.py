"""Normalized LM types — request configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LMToolSpec(BaseModel):
    """Provider-independent schema for a tool available to an LM."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class LMReasoningConfig(BaseModel):
    """Reasoning controls for models with native reasoning support."""

    effort: str | None = None
    max_tokens: int | None = None
    summary: str | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_value(cls, value: Any = None, **overrides: Any) -> LMReasoningConfig:
        data = _config_data(value, str_field="effort")
        data.update({key: value for key, value in overrides.items() if value is not _MISSING})
        return cls(**data)


class LMToolChoice(BaseModel):
    """Tool-choice controls for native tool-capable models."""

    mode: Literal["auto", "required", "none"] = "auto"
    # Tool names that are allowed for a model request.
    allowed: list[str] | None = None
    parallel: bool | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_value(cls, value: Any = None, **overrides: Any) -> LMToolChoice:
        data = _config_data(value, str_field="mode")
        data.update({key: value for key, value in overrides.items() if value is not _MISSING})
        return cls(**data)


class LMPromptCacheConfig(BaseModel):
    """Provider-side prompt/token cache controls.

    Prompt caching is not DSPy memoization. The provider call still happens,
    but the backend may reuse cached prompt prefixes or KV state for lower
    latency or lower input-token cost.
    """

    enabled: bool | None = None
    key: str | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_value(cls, value: Any = None, **overrides: Any) -> LMPromptCacheConfig:
        data = _config_data(value, bool_field="enabled")
        data.update({key: value for key, value in overrides.items() if value is not _MISSING})
        return cls(**data)


_MISSING = object()


def _config_data(value: Any, *, str_field: str | None = None, bool_field: str | None = None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    if str_field is not None and isinstance(value, str):
        return {str_field: value}
    if bool_field is not None and isinstance(value, bool):
        return {bool_field: value}
    raise TypeError(f"Cannot convert {type(value)!r} to a config object.")


class LMConfig(BaseModel):
    """Common generation controls for an LM request."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    n: int | None = None
    logprobs: bool | int | None = None
    response_format: Any | None = None
    reasoning: LMReasoningConfig | None = None
    tool_choice: LMToolChoice | None = None
    prompt_cache: LMPromptCacheConfig | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> LMConfig:
        return cls(**kwargs)


def _merge_lm_config(left: LMConfig | None, right: LMConfig | None) -> LMConfig | None:
    if left is None:
        return right
    if right is None:
        return left

    data = left.model_dump()
    extensions = {**left.extensions}
    for key in right.model_fields_set:
        value = getattr(right, key)
        if key == "extensions":
            extensions = dict(value) if value is not None else {}
            continue
        if key in ("reasoning", "tool_choice", "prompt_cache") and value is not None:
            left_value = data.get(key)
            right_value = value.model_dump(exclude_none=True)
            if isinstance(left_value, dict) and right_value:
                data[key] = {**left_value, **right_value}
            else:
                data[key] = right_value
            continue
        if isinstance(value, BaseModel):
            data[key] = value.model_dump(exclude_none=True)
        else:
            data[key] = value
    data["extensions"] = extensions
    return LMConfig(**data)


def _merge_config_overrides(config: LMConfig, kwargs: dict[str, Any]) -> LMConfig:
    if not kwargs:
        return config

    data = config.model_dump()
    extensions = dict(config.extensions)
    field_names = set(LMConfig.model_fields)

    for key, value in kwargs.items():
        if key == "extensions":
            if value is None:
                extensions = {}
            elif isinstance(value, Mapping):
                extensions.update(value)
            else:
                raise TypeError("`extensions` override must be a mapping or None.")
        elif key in field_names and key != "extensions":
            data[key] = value
        else:
            raise ValueError(f"Unknown LM config override: {key!r}")

    data["extensions"] = extensions
    return LMConfig(**data)


def _coerce_from_call_config_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(kwargs)
    prompt_cache = data.get("prompt_cache")
    if isinstance(prompt_cache, bool):
        data["prompt_cache"] = LMPromptCacheConfig(enabled=prompt_cache)
    return data


def _lm_config_data_from_kwargs(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Extract LMConfig fields from a loose kwargs mapping.

    Provider-only settings such as ``api_base`` and ``api_key`` are ignored so
    callers can spread ``lm.kwargs`` into ``LMRequest.from_call`` safely.
    """
    if not raw:
        return {}

    data = dict(raw)
    if "max_completion_tokens" in data and "max_tokens" not in data:
        data["max_tokens"] = data.pop("max_completion_tokens")

    data = _coerce_from_call_config_kwargs(data)
    field_names = set(LMConfig.model_fields) - {"extensions"}
    filtered = {key: value for key, value in data.items() if key in field_names and value is not None}
    extensions = data.get("extensions")
    if isinstance(extensions, Mapping):
        filtered["extensions"] = dict(extensions)
    return filtered


def coerce_lm_config(value: LMConfig | Mapping[str, Any] | None = None) -> LMConfig:
    if value is None:
        return LMConfig()
    if isinstance(value, LMConfig):
        return value
    return LMConfig(**dict(value))


def lm_defaults_config(lm: Any) -> LMConfig:
    return LMConfig(**_lm_config_data_from_kwargs(getattr(lm, "kwargs", None) or {}))


def merge_lm_request_config(lm: Any, config: LMConfig | None = None) -> LMConfig:
    return _merge_lm_config(lm_defaults_config(lm), config or LMConfig()) or (config or LMConfig())
