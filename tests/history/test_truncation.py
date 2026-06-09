import pytest

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.core.types import LMRequest
from dspy.errors import ContextWindowExceededError
from dspy.history import (
    REPLEntry,
    REPLHistory,
    TurnEvent,
    TurnLog,
    call_with_history_truncation,
)
from dspy.primitives import Prediction
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, TransparencyMode
from dspy.testing import DummyLM

_ = LMRequest


def make_run():
    return RunContext.create(
        lm=DummyLM([{"answer": "ok"}]),
        adapter=ChatAdapter(),
        telemetry=TelemetryConfig(transparency=TransparencyMode.off, call_log=CallLogMode.off),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history_factory", "expected_remaining_key", "expected_remaining_value"),
    [
        (
            lambda: TurnLog(
                turns=(
                    TurnEvent(thought="t1"),
                    TurnEvent(thought="t2"),
                    TurnEvent(thought="t3"),
                )
            ),
            "thought",
            "t2",
        ),
        (
            lambda: REPLHistory(
                entries=[
                    REPLEntry(reasoning="r1", code="a=1", output="1"),
                    REPLEntry(reasoning="r2", code="a=2", output="2"),
                    REPLEntry(reasoning="r3", code="a=3", output="3"),
                ]
            ),
            "code",
            "a=2",
        ),
    ],
)
async def test_call_with_history_truncation_returns_truncated_history(
    history_factory, expected_remaining_key, expected_remaining_value
):
    history = history_factory()
    received_histories = []
    attempts = {"count": 0}

    async def module(*, turn_log, run, options=None, **kwargs):
        received_histories.append(turn_log)
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ContextWindowExceededError
        return Prediction(answer="ok")

    run = make_run()
    extracted = await call_with_history_truncation(module, turn_log=history, run=run, question="Q")
    if isinstance(extracted.turn_log, TurnLog):
        assert len(extracted.turn_log.turns) == 2
        first_turn = extracted.turn_log.turns[0]
        assert isinstance(first_turn, TurnEvent)
        assert getattr(first_turn, expected_remaining_key) == expected_remaining_value
    else:
        assert len(extracted.turn_log.entries) == 2
        assert getattr(extracted.turn_log.entries[0], expected_remaining_key) == expected_remaining_value
    assert len(received_histories) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "history_factory",
    [
        lambda: TurnLog(
            turns=(
                TurnEvent(thought="t1"),
                TurnEvent(thought="t2"),
                TurnEvent(thought="t3"),
            )
        ),
        lambda: REPLHistory(
            entries=[
                REPLEntry(reasoning="r1", code="a=1", output="1"),
                REPLEntry(reasoning="r2", code="a=2", output="2"),
                REPLEntry(reasoning="r3", code="a=3", output="3"),
            ]
        ),
    ],
)
async def test_call_with_history_truncation_raises_after_max_attempts(history_factory):
    history = history_factory()

    async def module(*, turn_log, run, options=None, **kwargs):
        raise ContextWindowExceededError

    run = make_run()
    with pytest.raises(ValueError, match="even after 2 attempts"):
        await call_with_history_truncation(module, turn_log=history, run=run, max_attempts=2)


@pytest.mark.asyncio
async def test_repl_history_truncate_oldest():
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
