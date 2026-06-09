from __future__ import annotations

DEFAULT_PROVIDER = "openai"


def split_provider_model(model: str, *, default_provider: str = DEFAULT_PROVIDER) -> tuple[str, str]:
    if "/" in model:
        provider, rest = model.split("/", 1)
        return provider, rest
    return default_provider, model
