"""Call-site, adapter, and LM config resolution."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from dspy.clients.lm_strict import lm_kwargs_max_tokens
from dspy.core.types import LMConfig, coerce_lm_config, lm_defaults_config, merge_lm_config
from dspy.runtime.config import CallSite
from dspy.runtime.transparency.types import CompiledCall, TransparencyViolation
from dspy.task_spec import TaskSpec  # noqa: TC001

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.runtime.run_context import RunContext

LM_CONFIG_PROVENANCE_FIELDS: tuple[str, ...] = tuple(name for name in LMConfig.model_fields if name != "extensions")


def resolve_call_site(
    *,
    run: RunContext,
    call_site: CallSite | None = None,
    default_module: str,
    default_phase: str = "predict",
    default_lm_role: str = "default",
) -> CallSite:
    if call_site is not None:
        return call_site
    if run.call_site is not None:
        return run.call_site
    return CallSite(module=default_module, phase=default_phase, lm_role=default_lm_role)


def require_adapter(adapter: Adapter | None) -> Adapter:
    if adapter is not None:
        return adapter
    raise TransparencyViolation(
        "adapter not configured.",
        fixes=[
            "RunContext.create(lm=LM(...), adapter=JSONAdapter())",
            "or pass adapter=... when creating RunContext",
        ],
    )


def merge_call_config(
    lm: BaseLM,
    call_config: LMConfig | dict[str, Any] | None,
    *,
    override: dict[str, Any] | None = None,
) -> LMConfig:
    base = coerce_lm_config(call_config)
    if override:
        merged = merge_lm_config(base, coerce_lm_config(override))
        config = merged if merged is not None else coerce_lm_config(override)
    else:
        config = base
    return merge_lm_config(lm_defaults_config(lm), config) or config


def trace_config_provenance(lm: BaseLM, merged_config: LMConfig) -> dict[str, str]:
    provenance: dict[str, str] = {}
    lm_kwargs = getattr(lm, "kwargs", {})
    for field in LM_CONFIG_PROVENANCE_FIELDS:
        lm_value = lm_kwargs.get(field)
        if lm_value is None and field == "max_tokens":
            lm_value = lm_kwargs_max_tokens(lm_kwargs)
        call_value = getattr(merged_config, field, None)
        if call_value is not None and call_value != lm_value:
            provenance[field] = "predict.config"
        elif call_value is not None or lm_value is not None:
            provenance[field] = "lm.kwargs"
        else:
            provenance[field] = "unset"
    return provenance


def resolve_call(
    *,
    lm: BaseLM,
    adapter: Adapter,
    adapter_notes: list[str] | None = None,
    task_spec: TaskSpec,
    processed_task_spec: TaskSpec | None = None,
    config: LMConfig | None = None,
    config_provenance: dict[str, str] | None = None,
    messages: list[dict[str, Any]] | None = None,
    task_spec_mutations: list[str] | None = None,
    module: str = "Predict",
    phase: str = "predict",
    lm_role: str = "default",
    violations: list[str] | None = None,
) -> CompiledCall:
    adapter_class = type(adapter).__name__
    if adapter_notes:
        adapter_class = f"{adapter_class}({'; '.join(adapter_notes)})"
    cache = lm.provider_options.cache
    return CompiledCall(
        call_id=str(uuid.uuid4()),
        module=module,
        phase=phase,
        lm_role=lm_role,
        adapter_class=adapter_class,
        adapter_notes=adapter_notes or [],
        original_task_spec=task_spec,
        processed_task_spec=processed_task_spec or task_spec,
        task_spec_mutations=task_spec_mutations or [],
        messages=messages or [],
        config=config or LMConfig(),
        config_provenance=config_provenance or {},
        lm_model=getattr(lm, "model", ""),
        lm_kwargs=dict(getattr(lm, "kwargs", {})),
        cache=cache,
        violations=violations or [],
    )
