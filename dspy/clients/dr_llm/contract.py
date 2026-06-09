from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.errors import LMConfigurationError

if TYPE_CHECKING:
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
