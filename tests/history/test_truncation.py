import pytest

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.errors import ContextWindowExceededError
from dspy.history import TurnEvent, TurnLog, call_with_turn_log_truncation
from dspy.primitives.prediction import Prediction
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, TransparencyMode
from dspy.testing import DummyLM


def make_run():
    return RunContext.create(
        lm=DummyLM([{"answer": "ok"}]),
        adapter=ChatAdapter(),
        telemetry=TelemetryConfig(transparency=TransparencyMode.off, call_log=CallLogMode.off),
        init_run_log=False,
    )


@pytest.mark.asyncio
async def test_call_with_turn_log_truncation_returns_truncated_turn_log():
    received_turn_logs: list[TurnLog] = []
    turn_log = TurnLog(
        turns=(
            TurnEvent(thought="t1"),
            TurnEvent(thought="t2"),
            TurnEvent(thought="t3"),
        )
    )
    attempts = {"count": 0}

    async def module(*, turn_log: TurnLog, run, options=None, **kwargs):
        received_turn_logs.append(turn_log)
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ContextWindowExceededError
        return Prediction(answer="ok")

    run = make_run()
    extracted = await call_with_turn_log_truncation(module, turn_log=turn_log, run=run, question="Q")
    assert len(extracted.turn_log.turns) == 2
    first_turn = extracted.turn_log.turns[0]
    assert isinstance(first_turn, TurnEvent)
    assert first_turn.thought == "t2"
    assert len(received_turn_logs) == 2
    assert len(received_turn_logs[1].turns) == 2


@pytest.mark.asyncio
async def test_call_with_turn_log_truncation_raises_after_max_attempts():
    turn_log = TurnLog(
        turns=(
            TurnEvent(thought="t1"),
            TurnEvent(thought="t2"),
            TurnEvent(thought="t3"),
        )
    )

    async def module(*, turn_log: TurnLog, run, options=None, **kwargs):
        raise ContextWindowExceededError

    run = make_run()
    with pytest.raises(ValueError, match="even after 2 attempts"):
        await call_with_turn_log_truncation(module, turn_log=turn_log, run=run, max_attempts=2)
