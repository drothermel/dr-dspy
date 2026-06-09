from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProgramCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    score: float | None
    program: Any
    label: str | None = None
    subscores: list[float] | None = None
    full_eval: bool | None = None
    seed: Any | None = None


class CompileStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_calls: int = 0
    prompt_model_calls: int = 0
    error_occurred: bool = False
    best_score: float | None = None
    trial_logs: dict[Any, Any] = Field(default_factory=dict)
    copro_depth_stats: dict[str, Any] | None = None


class CompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    program: Any
    candidates: list[ProgramCandidate] = Field(default_factory=list)
    stats: CompileStats = Field(default_factory=CompileStats)

    @classmethod
    def with_compiled_program(cls, program: Any, **kwargs: Any) -> CompileResult:
        program._compiled = True
        return cls(program=program, **kwargs)
