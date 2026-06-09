from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext

from dspy.clients.lm_normalize import lm_kwargs_max_tokens
from dspy.core.types import LMConfig
from dspy.task_spec import TaskSpec

logger = logging.getLogger(__name__)
TransparencyMode = Literal["strict", "warn", "verbose", "off"]
PLACEHOLDER_DESC_PREFIX = "${"


class TransparencyViolation(Exception):
    def __init__(self, message: str, *, fixes: list[str] | None = None) -> None:
        self.fixes = fixes or []
        full_message = message
        if self.fixes:
            full_message += "\nFixes:\n" + "\n".join(f"  - {fix}" for fix in self.fixes)
        super().__init__(full_message)


class CallSite(BaseModel):
    model_config = ConfigDict(frozen=True)

    module: str
    phase: str = "predict"
    lm_role: str = "default"


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


def is_placeholder_desc(desc: str, field_name: str) -> bool:
    return desc == f"${{{field_name}}}"


def collect_task_spec_violations(task_spec: TaskSpec | None) -> list[str]:
    if task_spec is None:
        return []
    return [
        f"Field {field.name!r} uses placeholder desc {field.desc!r}. Fix: set an explicit desc= on input_field/output_field."
        for field in (*task_spec.inputs, *task_spec.outputs)
        if is_placeholder_desc(field.desc, field.name)
    ]


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
    for task_spec in (call.original_task_spec, call.processed_task_spec):
        violations.extend(collect_task_spec_violations(task_spec))
    if call.lm_model:
        violations.extend(collect_config_violations(config=call.config, lm_kwargs={}, cache=call.cache))
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
