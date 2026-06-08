from contextlib import contextmanager

from dspy.compile.resolve import resolve_adapter
from dspy.runtime.run_context import RunContext
from dspy.utils.transparency import reset_active_call_metadata, set_active_call_metadata


def optimizer_run_context(
    run: RunContext,
    *,
    lm,
    adapter=None,
    phase: str,
    lm_role: str,
    **extra,
) -> RunContext:
    transparency = run.telemetry.transparency
    resolved_adapter, _notes = resolve_adapter(adapter or run.adapter, transparency=transparency)
    return run.fork(lm=lm, adapter=resolved_adapter, **extra)


@contextmanager
def optimizer_lm_context(run: RunContext, *, lm, adapter=None, phase: str, lm_role: str, **extra):
    metadata_token = set_active_call_metadata(phase=phase, lm_role=lm_role, module=extra.pop("module", "optimizer"))
    try:
        yield optimizer_run_context(run, lm=lm, adapter=adapter, phase=phase, lm_role=lm_role, **extra)
    finally:
        reset_active_call_metadata(metadata_token)
