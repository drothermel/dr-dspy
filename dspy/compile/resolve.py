from __future__ import annotations

import uuid
from typing import Any

from dspy.adapters.base import Adapter
from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMConfig, _merge_lm_config, coerce_lm_config, lm_defaults_config
from dspy.task_spec import TaskSpec
from dspy.utils.transparency import CompiledCall, TransparencyMode


def resolve_adapter(
    settings_adapter: Adapter | None,
    *,
    transparency: TransparencyMode = "strict",
    fallback_adapter_factory: Any | None = None,
) -> tuple[Adapter, list[str]]:
    notes: list[str] = []
    if settings_adapter is not None:
        return (settings_adapter, notes)
    if transparency == "off":
        if fallback_adapter_factory is None:
            from dspy.adapters.chat_adapter import ChatAdapter

            fallback_adapter_factory = ChatAdapter
        notes.append("defaulted to ChatAdapter because transparency=off and adapter was not configured")
        return (fallback_adapter_factory(), notes)
    from dspy.utils.transparency import TransparencyViolation

    raise TransparencyViolation(
        "adapter not configured.",
        fixes=["settings.configure(adapter=JSONAdapter())", "or pass adapter=... in settings.context(...)"],
    )


def resolve_lm_config(
    lm: BaseLM, predict_config: LMConfig | dict[str, Any] | None, *, override: dict[str, Any] | None = None
) -> tuple[LMConfig, dict[str, str]]:
    base = coerce_lm_config(predict_config)
    if override:
        merged = _merge_lm_config(base, coerce_lm_config(override))
        config = merged if merged is not None else coerce_lm_config(override)
    else:
        config = base
    merged_request = _merge_lm_config(lm_defaults_config(lm), config) or config
    provenance: dict[str, str] = {}
    for field in ("temperature", "max_tokens", "n", "top_p"):
        lm_value = getattr(lm, "kwargs", {}).get(field)
        if lm_value is None and field == "max_tokens":
            lm_value = getattr(lm, "kwargs", {}).get("max_completion_tokens")
        call_value = getattr(merged_request, field, None)
        if call_value is not None and call_value != lm_value:
            provenance[field] = "predict.config"
        elif call_value is not None or lm_value is not None:
            provenance[field] = "lm.kwargs"
        else:
            provenance[field] = "unset"
    return (merged_request, provenance)


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
    cache = getattr(lm, "cache", lm.kwargs.get("cache") if hasattr(lm, "kwargs") else None)
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
        cache=cache,
        violations=violations or [],
    )
