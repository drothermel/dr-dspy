from __future__ import annotations

from typing import TYPE_CHECKING, cast

from dspy.errors import LMConfigurationError

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM

BUILTIN_LM_CLASS_PATH = "dspy.clients.lm.LM"

_REGISTRY: dict[str, type] | None = None


def _lm_class_registry() -> dict[str, type]:
    global _REGISTRY
    if _REGISTRY is None:
        from dspy.clients.dr_llm.direct import DrLlmDirectLM
        from dspy.clients.dr_llm.pool import DrLlmPoolLM
        from dspy.clients.lm.client import LM

        _REGISTRY = {
            BUILTIN_LM_CLASS_PATH: LM,
            "dspy.clients.dr_llm.direct.DrLlmDirectLM": DrLlmDirectLM,
            "dspy.clients.dr_llm.pool.DrLlmPoolLM": DrLlmPoolLM,
        }
    return _REGISTRY


def get_lm_class(class_path: str) -> type[BaseLM]:
    registry = _lm_class_registry()
    if class_path not in registry:
        known = ", ".join(sorted(registry))
        raise LMConfigurationError(f"Unknown serialized LM class `{class_path}`. Known builtin class paths: {known}.")
    return cast("type[BaseLM]", registry[class_path])
