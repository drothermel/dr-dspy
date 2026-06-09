from __future__ import annotations

from typing import Any

REJECTED_LM_KWARG_KEYS = frozenset({"reasoning_effort", "max_completion_tokens", "cache"})


class LegacyLMKeyError(ValueError):
    """Raised when legacy LM kwargs or serialized state keys are used."""


def _reject_legacy_keys(data: dict[str, Any], *, context: str) -> None:
    legacy_keys = sorted(key for key in REJECTED_LM_KWARG_KEYS if key in data)
    if legacy_keys:
        joined = ", ".join(repr(key) for key in legacy_keys)
        raise LegacyLMKeyError(
            f"Legacy LM {context} key(s) {joined} are not supported. "
            "Use reasoning={{'effort': ...}}, max_tokens, and LMProviderOptions(cache=...) instead."
        )


def validate_lm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate LM kwargs and return a shallow copy without legacy aliases."""
    data = dict(kwargs)
    _reject_legacy_keys(data, context="kwarg")
    return data


def validate_lm_state(state: dict[str, Any]) -> dict[str, Any]:
    """Validate serialized LM state and return a shallow copy without legacy aliases."""
    data = dict(state)
    _reject_legacy_keys(data, context="state")
    return data


def lm_kwargs_max_tokens(kwargs: dict[str, Any]) -> int | None:
    """Return max_tokens from LM kwargs."""
    max_tokens = kwargs.get("max_tokens")
    return max_tokens if isinstance(max_tokens, int) else None
