from __future__ import annotations

from typing import Protocol

__all__ = ["Proposer"]


class Proposer(Protocol):
    async def propose_instructions_for_program(
        self,
        trainset,
        program,
        demo_candidates,
        trial_logs,
        N,  # noqa: N803
        *,
        run,
    ) -> dict[int, list[str]]: ...

    async def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i,
        demo_candidates,
        demo_set_i,
        trial_logs,
        tip=None,
        *,
        run,
    ) -> str: ...
