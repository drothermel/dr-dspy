from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy.teleprompt.compilation import ProgramCandidate


@dataclass
class MIPROSearchSession:
    """Per-compile mutable state for a single MIPRO ``optimize_prompt_parameters`` run."""

    best_program: Any
    best_score: float
    trial_logs: dict[int, dict[str, Any]]
    total_eval_calls: int
    score_data: list[ProgramCandidate] = field(default_factory=list)
    param_score_dict: dict[str, list[tuple[float, Any, dict[str, int]]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    fully_evaled_param_combos: dict[str, dict[str, Any]] = field(default_factory=dict)
