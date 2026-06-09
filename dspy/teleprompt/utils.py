from contextlib import contextmanager

from dspy.compile.resolve import resolve_adapter
from dspy.runtime.run_context import RunContext
from dspy.utils.transparency import CallSite


def optimizer_run_context(
    run: RunContext,
    *,
    lm,
    adapter=None,
    phase: str,
    lm_role: str,
    **extra,
) -> RunContext:
    resolved_adapter, _notes = resolve_adapter(adapter or run.adapter)
    return run.fork(lm=lm, adapter=resolved_adapter, **extra)


@contextmanager
def optimizer_lm_context(run: RunContext, *, lm, adapter=None, phase: str, lm_role: str, **extra):
    module = extra.pop("module", "optimizer")
    call_site = CallSite(module=module, phase=phase, lm_role=lm_role)
    child = optimizer_run_context(run, lm=lm, adapter=adapter, phase=phase, lm_role=lm_role, **extra).fork(
        call_site=call_site
    )
    yield child
