from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.primitives.module_graph import named_predictors

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM
    from dspy.primitives.module import Module


def set_lm(module: Module, lm: BaseLM | None) -> None:
    for _, predictor in named_predictors(module):
        predictor.lm = lm


def get_lm(module: Module) -> BaseLM:
    lm = optional_lm(module)
    if lm is None:
        raise ValueError("No LM is configured on this module's predictors.")
    return lm


def optional_lm(module: Module) -> BaseLM | None:
    """Return the module's LM when all predictors share one; otherwise ``None`` or raise."""
    all_used_lms = [predictor.lm for _, predictor in named_predictors(module)]
    if not all_used_lms:
        return None
    if len(set(all_used_lms)) != 1:
        raise ValueError(
            "Multiple LMs are configured on this module. Inspect per-predictor LMs via "
            "named_predictors() and read predictor.lm on each predictor."
        )
    return all_used_lms[0]
