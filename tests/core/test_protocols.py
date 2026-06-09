import pytest
from pydantic import ValidationError

from dspy.core.types.lm import LMForward
from dspy.history import (
    AgentHistory,
    ConversationTurnLog,
    ReActTurnEvent,
    ReActV2TurnEvent,
    REPLHistory,
    TurnLog,
)
from dspy.predict.predict import Predict
from dspy.predict.protocol import Predictor
from tests.task_spec.helpers import ts
from tests.test_utils import DummyLM


def test_lm_satisfies_lm_forward_protocol():
    lm = DummyLM([{"answer": "ok"}])
    assert isinstance(lm, LMForward)


def test_predict_satisfies_predictor_protocol(make_run):
    run = make_run(lm=DummyLM([{"a": "ok"}]))
    predict = Predict(ts("q -> a", instructions="Answer."))
    predict.lm = run.lm
    predict.run = run
    assert isinstance(predict, Predictor)


def test_predict_with_run_satisfies_predictor_protocol(make_run):
    run = make_run(lm=DummyLM([{"a": "ok"}]))
    predict = Predict(ts("q -> a", instructions="Answer."), run=run)
    predict.lm = run.lm
    assert isinstance(predict, Predictor)
    assert predict.demos == []


def test_turn_log_satisfies_conversation_turn_log_protocol():
    assert isinstance(TurnLog.empty(), ConversationTurnLog)


def test_repl_history_satisfies_agent_history_protocol():
    assert isinstance(REPLHistory.empty(), AgentHistory)


def test_turn_log_append_turn_round_trip():
    log = TurnLog.empty().append_turn(
        ReActTurnEvent(thought="think", tool_name="search", tool_args={"q": "test"}, observation="found")
    )
    assert len(log.turns) == 1
    turn = log.turns[0]
    assert isinstance(turn, ReActTurnEvent)
    assert turn.thought == "think"
    assert turn.tool_name == "search"
    assert turn.tool_args == {"q": "test"}
    assert turn.observation == "found"


def test_turn_log_rejects_empty_event():
    with pytest.raises(ValueError, match="ReActV2TurnEvent requires"):
        TurnLog.empty().append_turn(ReActV2TurnEvent())


def test_turn_log_immutable_after_append():
    event = ReActTurnEvent(thought="original", tool_name="t", tool_args={}, observation="o")
    log = TurnLog.empty().append_turn(event)
    with pytest.raises(ValidationError):
        event.thought = "mutated"
    turn = log.turns[0]
    assert isinstance(turn, ReActTurnEvent)
    assert turn.thought == "original"


def test_turn_log_coerces_dict_turns_on_load():
    log = TurnLog.model_validate(
        {
            "turns": [
                {
                    "agent": "react",
                    "thought": "legacy",
                    "tool_name": "search",
                    "tool_args": {},
                    "observation": "done",
                }
            ]
        }
    )
    assert isinstance(log.turns[0], ReActTurnEvent)
    assert log.turns[0].thought == "legacy"
