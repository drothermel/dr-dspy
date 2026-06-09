from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "GRPO": ("dspy.teleprompt.grpo.optimizer", "GRPO"),
}

__all__ = ["GRPO"]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
