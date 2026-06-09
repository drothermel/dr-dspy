from __future__ import annotations

import uuid
from typing import Any

from dspy.adapters.base import Adapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm_strict import lm_kwargs_max_tokens
from dspy.core.types import LMConfig, coerce_lm_config, lm_defaults_config, merge_lm_config
from dspy.task_spec import TaskSpec
from dspy.utils.transparency import CompiledCall, TransparencyViolation


def resolve_adapter(adapter: Adapter | None) -> tuple[Adapter, list[str]]:
    if adapter is not None:
        return (adapter, [])
    raise TransparencyViolation(
        "adapter not configured.",
        fixes=[
            "RunContext.create(lm=LM(...), adapter=JSONAdapter())",
            "or pass adapter=... when creating RunContext",
        ],
    )


def resolve_lm_config(
    lm: BaseLM, predict_config: LMConfig | dict[str, Any] | None, *, override: dict[str, Any] | None = None
) -> tuple[LMConfig, dict[str, str]]:
    base = coerce_lm_config(predict_config)
    if override:
        merged = merge_lm_config(base, coerce_lm_config(override))
        config = merged if merged is not None else coerce_lm_config(override)
    else:
        config = base
    merged_request = merge_lm_config(lm_defaults_config(lm), config) or config
    provenance: dict[str, str] = {}
    for field in ("temperature", "max_tokens", "n", "top_p"):
        lm_value = getattr(lm, "kwargs", {}).get(field)
        if lm_value is None and field == "max_tokens":
            lm_value = lm_kwargs_max_tokens(getattr(lm, "kwargs", {}))
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
