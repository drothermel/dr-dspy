from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_llm.backends.models import BackendCapabilities

_V1_SUPPORTED_PARAMS = frozenset({"temperature", "max_tokens", "top_p", "reasoning"})
_REASONING_CONTROL_MODES = frozenset({"reasoning", "thinking", "effort"})


def supports_reasoning_from_capabilities(capabilities: BackendCapabilities) -> bool:
    mode = capabilities.control_mode.lower()
    if mode in _REASONING_CONTROL_MODES:
        return True
    return bool(capabilities.supported_thinking_levels or capabilities.default_reasoning)


def supported_params_v1() -> set[str]:
    return set(_V1_SUPPORTED_PARAMS)
