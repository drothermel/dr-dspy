from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from dspy.clients.dr_llm.controls import DR_LLM_EXTENSION_KEY, parse_dr_llm_controls
from dspy.core.types.lm_provider import LMProviderOptions
from dspy.errors import LMConfigurationError, LMUnsupportedFeatureError

if TYPE_CHECKING:
    from dspy.core.types import LMConfig


def provider_options_from_serialized_state(
    *,
    provider_data: dict[str, Any] | None,
    remaining: dict[str, Any],
) -> LMProviderOptions:
    merged = dict(remaining)
    if provider_data:
        merged = {**provider_data, **merged}
    fields = set(LMProviderOptions.model_fields)
    data = {key: value for key, value in merged.items() if key in fields}
    return LMProviderOptions(**data)


def _provider_options_non_empty(provider_options: LMProviderOptions | None) -> bool:
    if provider_options is None:
        return False
    data = provider_options.model_dump(exclude_none=True)
    extensions = data.pop("extensions", {})
    if extensions:
        return True
    return bool(data)


def validate_dr_llm_ctor(
    *,
    model: str,
    provider_options: LMProviderOptions | None = None,
    **kwargs: Any,
) -> None:
    """Reject misleading BaseLM-style options that dr-llm v1 does not wire through."""
    if kwargs:
        unknown = ", ".join(repr(key) for key in sorted(kwargs))
        raise TypeError(
            f"DrLlm LM for {model!r} does not accept keyword argument(s) {unknown}. "
            "Provider auth and routing are configured via the dr-llm registry and environment, "
            "not LMProviderOptions or LiteLLM-style passthrough kwargs."
        )
    if _provider_options_non_empty(provider_options):
        raise LMConfigurationError(
            f"DrLlm LM for {model!r} does not accept LMProviderOptions. "
            "Configure provider auth and routing via the dr-llm registry and environment variables "
            "(for example OPENAI_API_KEY), not provider_options.",
            model=model,
        )


def reject_unsupported_merged_config(config: LMConfig, *, model: str) -> None:
    """Reject per-request LMConfig fields that dr-llm v1 does not map to BackendRequest."""
    if config.response_format is not None:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support structured response_format.",
            model=model,
            features=["response_format"],
        )
    if config.stop is not None:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support stop sequences.",
            model=model,
            features=["stop"],
        )
    if config.n is not None and config.n != 1:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support n>1 sampling.",
            model=model,
            features=["n"],
        )
    if config.logprobs is not None:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support logprobs.",
            model=model,
            features=["logprobs"],
        )
    if config.tool_choice is not None:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support tool_choice.",
            model=model,
            features=["tool_choice"],
        )
    if config.prompt_cache is not None:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support prompt_cache.",
            model=model,
            features=["prompt_cache"],
        )
    unknown_extensions = set(config.extensions) - {DR_LLM_EXTENSION_KEY}
    if unknown_extensions:
        joined = ", ".join(sorted(unknown_extensions))
        raise LMUnsupportedFeatureError(
            f"dr-llm backends v1 do not support LMConfig.extensions key(s): {joined}.",
            model=model,
            features=["extensions"],
        )
    try:
        controls = parse_dr_llm_controls(config.extensions.get(DR_LLM_EXTENSION_KEY))
    except ValidationError as exc:
        raise LMConfigurationError("Invalid dr_llm provider controls.", model=model) from exc
    reasoning = config.reasoning
    if reasoning is not None:
        unsupported_reasoning_fields: list[str] = []
        if reasoning.max_tokens is not None:
            unsupported_reasoning_fields.append("reasoning.max_tokens")
        if reasoning.summary is not None:
            unsupported_reasoning_fields.append("reasoning.summary")
        if unsupported_reasoning_fields:
            joined = ", ".join(unsupported_reasoning_fields)
            raise LMUnsupportedFeatureError(
                f"dr-llm backends v1 only support reasoning.effort via BackendRequest.effort; "
                f"unsupported field(s): {joined}.",
                model=model,
                features=["reasoning"],
            )
        if reasoning.effort is not None and (controls.effort is not None or controls.reasoning is not None):
            raise LMUnsupportedFeatureError(
                "dr-llm backends v1 do not support both generic reasoning.effort and "
                "dr_llm provider reasoning controls on the same request.",
                model=model,
                features=["reasoning", "extensions"],
            )
