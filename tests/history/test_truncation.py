import pytest

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.core.types import LMRequest
from dspy.errors import ContextWindowExceededError
from dspy.history import REPLEntry, TurnEvent, TurnLog, call_with_repl_history_truncation, call_with_turn_log_truncation
from dspy.primitives.prediction import Prediction
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, TransparencyMode
from dspy.testing import DummyLM

_ = LMRequest


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


@pytest.mark.asyncio
async def test_repl_history_truncate_oldest():
    from dspy.history import REPLHistory

    history = REPLHistory(
        entries=[
            REPLEntry(reasoning="r1", code="a=1", output="1"),
            REPLEntry(reasoning="r2", code="a=2", output="2"),
            REPLEntry(reasoning="r3", code="a=3", output="3"),
        ]
    )
    truncated = history.truncate_oldest()
    assert len(truncated.entries) == 2
    assert truncated.entries[0].code == "a=2"


@pytest.mark.asyncio
async def test_call_with_repl_history_truncation_returns_truncated_history():
    from dspy.history import REPLHistory

    history = REPLHistory(
        entries=[
            REPLEntry(reasoning="r1", code="a=1", output="1"),
            REPLEntry(reasoning="r2", code="a=2", output="2"),
            REPLEntry(reasoning="r3", code="a=3", output="3"),
        ]
    )
    attempts = {"count": 0}

    async def module(*, turn_log: REPLHistory, run, options=None, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ContextWindowExceededError
        return Prediction(code="done")

    run = make_run()
    extracted = await call_with_repl_history_truncation(module, turn_log=history, run=run)
    assert len(extracted.turn_log.entries) == 2
    assert extracted.turn_log.entries[0].code == "a=2"
