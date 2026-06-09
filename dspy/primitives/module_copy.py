from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

from dspy.primitives.module_graph import is_module_instance, predictors

if TYPE_CHECKING:
    from dspy.primitives.module import Module

logger = logging.getLogger(__name__)


def module_getstate(module: Module) -> dict[str, Any]:
    state = module.__dict__.copy()
    state.pop("call_log", None)
    state.pop("callbacks", None)
    return state


def module_setstate(module: Module, state: dict[str, Any]) -> None:
    module.__dict__.update(state)
    if not hasattr(module, "call_log"):
        module.call_log = []
    if not hasattr(module, "callbacks"):
        module.callbacks = []
    if not hasattr(module, "_compiled"):
        module._compiled = False
    if not hasattr(module, "run"):
        module.run = None


def deepcopy_module(module: Module) -> Module:
    try:
        return copy.deepcopy(module)
    except Exception:
        logger.debug(
            "copy.deepcopy failed for %s; falling back to manual deepcopy",
            module.__class__.__name__,
            exc_info=True,
        )
    new_instance = module.__class__.__new__(module.__class__)
    for attr, value in module.__dict__.items():
        if is_module_instance(value):
            setattr(new_instance, attr, deepcopy_module(value))
        else:
            try:
                setattr(new_instance, attr, copy.deepcopy(value))
            except Exception:
                logger.warning(
                    "Failed to deep copy attribute '%s' of %s, falling back to shallow copy or reference copy.",
                    attr,
                    module.__class__.__name__,
                )
                try:
                    setattr(new_instance, attr, copy.copy(value))
                except Exception:
                    setattr(new_instance, attr, value)
    return new_instance


def reset_copy(module: Module) -> Module:
    new_instance = deepcopy_module(module)
    for predictor in predictors(new_instance):
        predictor.reset()
    return new_instance
