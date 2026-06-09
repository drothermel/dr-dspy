from __future__ import annotations

from dr_llm.llm import ProviderName

from dspy.errors import LMUnsupportedFeatureError


def parse_dr_llm_provider(provider_name: str, *, model: str) -> ProviderName:
    try:
        return ProviderName(provider_name)
    except ValueError as exc:
        raise LMUnsupportedFeatureError(
            f"Unsupported dr-llm provider {provider_name!r}.",
            model=model,
        ) from exc
