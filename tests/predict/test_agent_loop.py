import pytest

from dspy.history import TurnEvent, TurnLog
from dspy.predict.agent_loop import AgentLoopControl, AgentLoopRunner, AgentStepResult
from dspy.predict.agent_termination import AgentTerminationReason
from dspy.primitives import Prediction


@pytest.mark.asyncio
async def test_agent_loop_runner_continues_until_max_iters():
    seen_indices: list[int] = []

    async def step(turn_index: int, history: TurnLog) -> AgentStepResult[TurnLog]:
        seen_indices.append(turn_index)
        return AgentStepResult(history=history.append_turn(TurnEvent(thought=f"t{turn_index}")))

    result = await AgentLoopRunner[TurnLog]().run(
        max_iters=3,
        initial_history=TurnLog.empty(),
        step=step,
    )
    assert result.termination_reason == AgentTerminationReason.MAX_ITERS
    assert result.return_value is None
    assert len(result.history.turns) == 3
    assert seen_indices == [0, 1, 2]


@pytest.mark.asyncio
async def test_agent_loop_runner_breaks_early_with_reason():
    async def step(turn_index: int, history: TurnLog) -> AgentStepResult[TurnLog]:
        if turn_index == 1:
            return AgentStepResult(
                history=history,
                control=AgentLoopControl.BREAK,
                termination_reason=AgentTerminationReason.SUBMIT,
            )
        return AgentStepResult(history=history.append_turn(TurnEvent(thought=f"t{turn_index}")))

    result = await AgentLoopRunner[TurnLog]().run(
        max_iters=5,
        initial_history=TurnLog.empty(),
        step=step,
    )
    assert result.termination_reason == AgentTerminationReason.SUBMIT
    assert len(result.history.turns) == 1


@pytest.mark.asyncio
async def test_agent_loop_runner_returns_early():
    prediction = Prediction(answer="done")

    async def step(turn_index: int, history: TurnLog) -> AgentStepResult[TurnLog]:
        if turn_index == 0:
            return AgentStepResult(
                history=history.append_turn(TurnEvent(thought="done")),
                control=AgentLoopControl.RETURN,
                termination_reason=AgentTerminationReason.SUBMIT,
                return_value=prediction,
            )
        return AgentStepResult(history=history)

    result = await AgentLoopRunner[TurnLog]().run(
        max_iters=5,
        initial_history=TurnLog.empty(),
        step=step,
    )
    assert result.termination_reason == AgentTerminationReason.SUBMIT
    assert result.return_value is prediction


@pytest.mark.asyncio
async def test_agent_loop_runner_threads_history_through_steps():
    async def step(turn_index: int, history: TurnLog) -> AgentStepResult[TurnLog]:
        assert len(history.turns) == turn_index
        return AgentStepResult(history=history.append_turn(TurnEvent(thought=f"t{turn_index}")))

    result = await AgentLoopRunner[TurnLog]().run(
        max_iters=2,
        initial_history=TurnLog.empty(),
        step=step,
    )
    assert [turn.thought for turn in result.history.turns] == ["t0", "t1"]
