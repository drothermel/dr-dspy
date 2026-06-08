from contextlib import contextmanager

from dspy.compile.resolve import resolve_adapter
from dspy.dsp.utils.settings import settings
from dspy.utils.transparency import reset_active_call_metadata, set_active_call_metadata


@contextmanager
def optimizer_lm_context(*, lm, adapter=None, phase: str, lm_role: str, **extra):
    transparency = settings.get("transparency", "strict")
    resolved_adapter, _notes = resolve_adapter(adapter or settings.adapter, transparency=transparency)
    metadata_token = set_active_call_metadata(phase=phase, lm_role=lm_role, module=extra.pop("module", "optimizer"))
    with settings.context(lm=lm, adapter=resolved_adapter, **extra):
        try:
            yield
        finally:
            reset_active_call_metadata(metadata_token)
