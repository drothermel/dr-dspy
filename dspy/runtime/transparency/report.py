"""Transparency warning, verbose logging, and strict enforcement."""

from __future__ import annotations

import logging

from dspy.runtime.config import TransparencyMode
from dspy.runtime.transparency.types import CompiledCall, TransparencyViolation
from dspy.runtime.transparency.validate import collect_compiled_call_violations

logger = logging.getLogger(__name__)


def enforce_compiled_call_transparency(call: CompiledCall, mode: TransparencyMode) -> list[str]:
    violations = collect_compiled_call_violations(call)
    if mode == TransparencyMode.strict and violations:
        raise TransparencyViolation(
            f"Transparency strict mode violation(s) in phase={call.phase!r}, lm_role={call.lm_role!r}:",
            fixes=violations,
        )
    if mode in (TransparencyMode.warn, TransparencyMode.verbose) and violations:
        for violation in violations:
            logger.warning("Transparency: %s", violation)
    if mode == TransparencyMode.verbose:
        logger.info(
            "CompiledCall module=%s phase=%s adapter=%s lm=%s mutations=%s",
            call.module,
            call.phase,
            call.adapter_class,
            call.lm_model,
            call.task_spec_mutations,
        )
    return violations
