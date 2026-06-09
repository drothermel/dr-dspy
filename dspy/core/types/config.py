from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NativeAdaptationMode(StrEnum):
    ADAPT = "adapt"
    SKIP = "skip"


class LMToolSpec(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class LMReasoningConfig(BaseModel):
    effort: ReasoningEffort | None = None
    max_tokens: int | None = None
    summary: str | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def coerce_effort(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("effort"), str):
            data = dict(data)
            data["effort"] = ReasoningEffort(data["effort"])
        return data

    @classmethod
    def from_value(cls, value: Any = None, **overrides: Any) -> LMReasoningConfig:
        data = _config_data(value, str_field="effort")
        if isinstance(data.get("effort"), str):
            data["effort"] = ReasoningEffort(data["effort"])
        data.update({key: value for key, value in overrides.items() if value is not _MISSING})
        return cls(**data)


class LMToolChoice(BaseModel):
    mode: Literal["auto", "required", "none"] = "auto"
    allowed: list[str] | None = None
    parallel: bool | None = None
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_value(cls, value: Any = None, **overrides: Any) -> LMToolChoice:
        data = _config_data(value, str_field="mode")
        data.update({key: value for key, value in overrides.items() if value is not _MISSING})
        return cls(**data)


class LMPromptCacheConfig(BaseModel):
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
    """Per-call LM generation config merged over LM defaults at request time.

    ``response_format`` set here (via ``PredictOptions.config`` or ``LMRequest.config``)
    overrides ``LMProviderOptions.response_format`` from the LM's default kwargs when
    explicitly provided on this config instance. Adapter-layer structured-output
    policies may apply a further override at the adapter boundary.
    """

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


_NESTED_CONFIG_FIELDS = frozenset({"reasoning", "tool_choice", "prompt_cache"})


def merge_lm_config(left: LMConfig | None, right: LMConfig | None) -> LMConfig | None:
    """Merge two ``LMConfig`` values with right overriding left.

    Semantics:
    - Base state comes from ``left`` with ``exclude_none=True``.
    - Only fields present in ``right.model_fields_set`` are applied.
    - Scalars: right wins, including explicit ``None`` (clears the field).
    - Nested configs (``reasoning``, ``tool_choice``, ``prompt_cache``): shallow
      dict merge when both sides are non-``None``; explicit ``None`` on right clears.
    - ``extensions``: union-merge; ``None`` on right clears all extensions; an empty
      mapping on right is a no-op that preserves left keys; colliding keys use right.
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
        if key in _NESTED_CONFIG_FIELDS:
            if value is None:
                data[key] = None
            else:
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


def _coerce_from_call_config_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(kwargs)
    prompt_cache = data.get("prompt_cache")
    if isinstance(prompt_cache, bool):
        raise TypeError(
            "bool prompt_cache is not supported. Use LMPromptCacheConfig(enabled=...) or prompt_cache={'enabled': ...}."
        )
    if "max_completion_tokens" in data:
        raise ValueError("max_completion_tokens is not supported in LMConfig. Use max_tokens instead.")
    if "reasoning_effort" in data:
        raise ValueError("reasoning_effort is not supported in LMConfig. Use reasoning={'effort': ...} instead.")
    return data


def _lm_config_data_from_kwargs(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    data = dict(raw)
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
    return LMConfig(**_coerce_from_call_config_kwargs(value))


def lm_defaults_config(lm: Any) -> LMConfig:
    return LMConfig(**_lm_config_data_from_kwargs(getattr(lm, "kwargs", None) or {}))


def merge_lm_request_config(lm: Any, config: LMConfig | None = None) -> LMConfig:
    """Merge per-call ``config`` over LM default kwargs (including provider options).

    Fields explicitly set on ``config`` win over LM defaults. In particular,
    ``response_format`` on ``config`` overrides ``LMProviderOptions.response_format``
    seeded into ``lm.kwargs``.
    """
    return merge_lm_config(lm_defaults_config(lm), config or LMConfig()) or LMConfig()
