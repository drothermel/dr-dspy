from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic

from dspy.history.protocol import H
from dspy.predict.agent_termination import AgentTerminationReason

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def format_tool_exception(err: BaseException, *, limit: int = 5) -> str:
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


class AgentLoopControl(StrEnum):
    CONTINUE = "continue"
    BREAK = "break"
    RETURN = "return"


@dataclass(frozen=True)
class AgentStepResult(Generic[H]):
    history: H
    control: AgentLoopControl = AgentLoopControl.CONTINUE
    termination_reason: AgentTerminationReason | None = None
    return_value: Any = None


@dataclass(frozen=True)
class AgentLoopResult(Generic[H]):
    history: H
    termination_reason: AgentTerminationReason
    return_value: Any | None = None


class AgentLoopRunner(Generic[H]):
    async def run(
        self,
        *,
        max_iters: int,
        initial_history: H,
        step: Callable[[int, H], Awaitable[AgentStepResult[H]]],
        default_termination: AgentTerminationReason = AgentTerminationReason.MAX_ITERS,
    ) -> AgentLoopResult[H]:
        history = initial_history
        termination_reason = default_termination
        for turn_index in range(max_iters):
            step_result = await step(turn_index, history)
            history = step_result.history
            if step_result.control is AgentLoopControl.RETURN:
                return AgentLoopResult(
                    history=history,
                    termination_reason=step_result.termination_reason or default_termination,
                    return_value=step_result.return_value,
                )
            if step_result.control is AgentLoopControl.BREAK:
                return AgentLoopResult(
                    history=history,
                    termination_reason=step_result.termination_reason or default_termination,
                )
        return AgentLoopResult(history=history, termination_reason=termination_reason)
