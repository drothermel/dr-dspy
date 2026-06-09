from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.errors import LMConfigurationError

if TYPE_CHECKING:
    from dspy.core.types.config import LMConfig
    from dspy.core.types.lm_provider import LMProviderOptions


def provider_options_from_serialized_state(
    *,
    provider_data: dict[str, Any] | None,
    remaining: dict[str, Any],
) -> LMProviderOptions:
    from dspy.core.types.lm_provider import LMProviderOptions as ProviderOptions

    merged = dict(remaining)
    if provider_data:
        merged = {**provider_data, **merged}
    fields = set(ProviderOptions.model_fields)
    data = {key: value for key, value in merged.items() if key in fields}
    return ProviderOptions(**data)


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
    from dspy.errors import LMUnsupportedFeatureError

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
    if config.n is not None:
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
    if config.extensions:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support LMConfig.extensions.",
            model=model,
            features=["extensions"],
        )
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
