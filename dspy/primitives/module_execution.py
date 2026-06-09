from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dspy.primitives.prediction import Prediction
from dspy.runtime import RunContext, resolve_run, track_usage
from dspy.runtime.active_run import call_scope, get_active_usage_tracker

if TYPE_CHECKING:
    from dspy.primitives.module import Module
    from dspy.runtime.call_options import ModuleCallOptions

logger = logging.getLogger(__name__)

_DIRECT_AFORWARD_WARNED: set[type] = set()


def warn_direct_aforward_once(cls: type) -> None:
    if cls in _DIRECT_AFORWARD_WARNED:
        return
    _DIRECT_AFORWARD_WARNED.add(cls)
    logger.warning(
        "Calling module.aforward(...) on %s directly is discouraged. Please use await module(...) instead.",
        cls.__name__,
    )


async def invoke_module(
    module: Module,
    *,
    run: RunContext,
    options: ModuleCallOptions | None = None,
    **inputs: Any,
) -> Prediction:
    run = resolve_run(run=run, bound_run=module.run)
    async with call_scope(run=run, caller=module):
        if run.telemetry.track_usage and run.usage_tracker is None:
            with track_usage(run) as usage_tracker:
                output = await module._aforward_impl(run=run, options=options, **inputs)
            tokens = usage_tracker.get_total_tokens()
        else:
            output = await module._aforward_impl(run=run, options=options, **inputs)
            usage_tracker = get_active_usage_tracker(run) if run.telemetry.track_usage else None
            tokens = usage_tracker.get_total_tokens() if usage_tracker else None
        if tokens:
            set_lm_usage(module, tokens=tokens, output=output)
        return output


def set_lm_usage(module: Module, *, tokens: dict[str, Any], output: Any) -> None:
    prediction_in_output = None
    if isinstance(output, Prediction):
        prediction_in_output = output
    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Prediction):
        prediction_in_output = output[0]
    if prediction_in_output:
        prediction_in_output.set_lm_usage(tokens)
    else:
        logger.warning(
            "Failed to set LM usage. Please return `dspy.primitives.prediction.Prediction` object from Module to enable usage tracking."
        )
