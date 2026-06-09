"""Proposer protocol for instruction-candidate generation during optimization.

Import ``Proposer`` from ``dspy.propose.protocol``. Implementations return
candidate instructions keyed by predictor index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from dspy.primitives import Example, Module
    from dspy.runtime.run_context import RunContext

__all__ = ["Proposer", "TrialLogEntry", "TrialLogs"]


class TrialLogEntry(TypedDict, total=False):
    program_path: str
    score: float


TrialLogs = dict[int, TrialLogEntry]


class Proposer(Protocol):
    async def propose_instructions_for_program(
        self,
        trainset: list[Example],
        program: Module,
        demo_candidates: list | None,
        trial_logs: TrialLogs,
        num_candidates: int,
        *,
        run: RunContext,
    ) -> dict[int, list[str]]: ...

    async def propose_instruction_for_predictor(
        self,
        program: Module,
        predictor: object,
        pred_i: int,
        demo_candidates: list | None,
        demo_set_i: int,
        trial_logs: TrialLogs,
        tip: str | None = None,
        *,
        run: RunContext,
    ) -> str: ...
