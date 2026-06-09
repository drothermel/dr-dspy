"""Transparency audit types, validation, and call resolution."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.clients.lm_strict import lm_kwargs_max_tokens
from dspy.core.types import LMConfig, coerce_lm_config, lm_defaults_config, merge_lm_config
from dspy.runtime.config import CallSite, TransparencyMode
from dspy.task_spec import TaskSpec  # noqa: TC001

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.runtime.run_context import RunContext

logger = logging.getLogger(__name__)


class TransparencyViolation(Exception):  # noqa: N818
    def __init__(self, message: str, *, fixes: list[str] | None = None) -> None:
        self.fixes = fixes or []
        full_message = message
        if self.fixes:
            full_message += "\nFixes:\n" + "\n".join(f"  - {fix}" for fix in self.fixes)
        super().__init__(full_message)


class CompiledCall(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    call_id: str
    module: str = "unknown"
    phase: str = "predict"
    lm_role: str = "default"
    adapter_class: str = ""
    adapter_notes: list[str] = Field(default_factory=list)
    original_task_spec: TaskSpec | None = None
    processed_task_spec: TaskSpec | None = None
    task_spec_mutations: list[str] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    config: LMConfig = Field(default_factory=LMConfig)
    config_provenance: dict[str, str] = Field(default_factory=dict)
    lm_model: str = ""
    lm_kwargs: dict[str, Any] = Field(default_factory=dict)
    cache: bool | None = None
    violations: list[str] = Field(default_factory=list)


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


def collect_config_violations(*, config: LMConfig, lm_kwargs: dict[str, Any], cache: bool | None) -> list[str]:
    violations: list[str] = []
    temperature = config.temperature if config.temperature is not None else lm_kwargs.get("temperature")
    max_tokens = config.max_tokens if config.max_tokens is not None else lm_kwargs_max_tokens(lm_kwargs)
    if temperature is None:
        violations.append(
            "temperature is None (provider default). Fix: LM(..., temperature=0.0) or pass config={'temperature': ...}."
        )
    if max_tokens is None:
        violations.append(
            "max_tokens is None (provider default). Fix: LM(..., max_tokens=4000) or pass config={'max_tokens': ...}."
        )
    if cache is None:
        violations.append(
            "cache is not explicit on the LM. Fix: LM(..., provider_options=LMProviderOptions(cache=False)) or cache=True."
        )
    return violations


def validate_compiled_call(call: CompiledCall, mode: TransparencyMode) -> list[str]:
    violations = list(call.violations)
    if not call.adapter_class:
        violations.append("adapter not configured. Fix: RunContext.create(lm=LM(...), adapter=JSONAdapter()).")
    if call.lm_model:
        violations.extend(collect_config_violations(config=call.config, lm_kwargs=call.lm_kwargs, cache=call.cache))
    if mode == "strict" and violations:
        raise TransparencyViolation(
            f"Transparency strict mode violation(s) in phase={call.phase!r}, lm_role={call.lm_role!r}:",
            fixes=violations,
        )
    if mode in ("warn", "verbose") and violations:
        for violation in violations:
            logger.warning("Transparency: %s", violation)
    if mode == "verbose":
        logger.info(
            "CompiledCall module=%s phase=%s adapter=%s lm=%s mutations=%s",
            call.module,
            call.phase,
            call.adapter_class,
            call.lm_model,
            call.task_spec_mutations,
        )
    return violations


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
