"""Transparency audit DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types import LMConfig
from dspy.task_spec import TaskSpec  # noqa: TC001


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
