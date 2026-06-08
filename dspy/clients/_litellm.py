from __future__ import annotations

import functools
import sys
from typing import TYPE_CHECKING, Any

from dspy.utils.lazy_import import require

if TYPE_CHECKING:
    import types


@functools.cache
def _configure_litellm_defaults(litellm: types.ModuleType) -> None:
    litellm.telemetry = False
    litellm.cache = None
    if not getattr(litellm, "_dspy_logging_configured", False):
        litellm.suppress_debug_info = True
        litellm._dspy_logging_configured = True


def _materialize_litellm(litellm: types.ModuleType) -> None:
    _completion = litellm.completion


@functools.cache
def get_litellm(*, feature: str) -> Any:
    litellm = require("litellm", extra="litellm", feature=feature)
    _materialize_litellm(litellm)
    _configure_litellm_defaults(litellm)
    return litellm


def is_litellm_context_window_error(error: Exception) -> bool:
    litellm_module = sys.modules.get("litellm")
    context_window_error = getattr(litellm_module, "ContextWindowExceededError", None)
    return context_window_error is not None and isinstance(error, context_window_error)
