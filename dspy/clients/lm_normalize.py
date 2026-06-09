from __future__ import annotations

from typing import Any


def normalize_lm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy LM kwarg names to canonical internal shapes."""
    data = dict(kwargs)
    if "max_completion_tokens" in data and "max_tokens" not in data:
        data["max_tokens"] = data.pop("max_completion_tokens")
    elif "max_completion_tokens" in data and "max_tokens" in data:
        data.pop("max_completion_tokens", None)
    if "reasoning_effort" in data:
        effort = data.pop("reasoning_effort")
        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict):
            if reasoning.get("effort") is None:
                data["reasoning"] = {**reasoning, "effort": effort}
        else:
            data["reasoning"] = {"effort": effort}
    return data


def normalize_lm_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize serialized LM state before load_state."""
    data = dict(state)
    max_tokens = data.pop("max_tokens", None)
    max_completion_tokens = data.pop("max_completion_tokens", None)
    if max_tokens is None and max_completion_tokens is not None:
        max_tokens = max_completion_tokens
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if "reasoning_effort" in data:
        effort = data.pop("reasoning_effort")
        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict):
            if reasoning.get("effort") is None:
                data["reasoning"] = {**reasoning, "effort": effort}
        else:
            data["reasoning"] = {"effort": effort}
    return data


def lm_kwargs_max_tokens(kwargs: dict[str, Any]) -> int | None:
    """Return max_tokens from LM kwargs, accepting legacy max_completion_tokens."""
    max_tokens = kwargs.get("max_tokens")
    if max_tokens is not None:
        return max_tokens
    value = kwargs.get("max_completion_tokens")
    return value if isinstance(value, int) else None
