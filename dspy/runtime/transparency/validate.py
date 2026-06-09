"""Pure transparency violation collection."""

from __future__ import annotations

from typing import Any

from dspy.clients.lm_strict import lm_kwargs_max_tokens
from dspy.core.types import LMConfig  # noqa: TC001 — runtime field access on config
from dspy.runtime.transparency.types import CompiledCall  # noqa: TC001 — runtime field access on call


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


def collect_compiled_call_violations(call: CompiledCall) -> list[str]:
    violations = list(call.violations)
    if not call.adapter_class:
        violations.append("adapter not configured. Fix: RunContext.create(lm=LM(...), adapter=JSONAdapter()).")
    if call.lm_model:
        violations.extend(collect_config_violations(config=call.config, lm_kwargs=call.lm_kwargs, cache=call.cache))
    return violations
