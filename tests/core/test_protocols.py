import pytest
from pydantic import ValidationError

from dspy.core.types.lm import LMForward
from dspy.history import AgentHistory, ConversationTurnLog, REPLHistory, TurnEvent, TurnLog
from dspy.predict.predict import Predict
from dspy.predict.protocol import Predictor
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def test_lm_satisfies_lm_forward_protocol():
    lm = DummyLM([{"answer": "ok"}])
    assert isinstance(lm, LMForward)


def test_predict_satisfies_predictor_protocol(make_run):
    run = make_run(lm=DummyLM([{"a": "ok"}]))
    predict = Predict(ts("q -> a", instructions="Answer."))
    predict.lm = run.lm
    predict.run = run
    assert isinstance(predict, Predictor)


def test_turn_log_satisfies_conversation_turn_log_protocol():
    assert isinstance(TurnLog.empty(), ConversationTurnLog)


def test_repl_history_satisfies_agent_history_protocol():
    assert isinstance(REPLHistory.empty(), AgentHistory)


def test_turn_log_append_turn_round_trip():
    log = TurnLog.empty().append_turn(
        TurnEvent(thought="think", tool_name="search", tool_args={"q": "test"}, observation="found")
    )
    assert len(log.turns) == 1
    turn = log.turns[0]
    assert isinstance(turn, TurnEvent)
    assert turn.thought == "think"
    assert turn.tool_name == "search"
    assert turn.tool_args == {"q": "test"}
    assert turn.observation == "found"


def test_turn_log_rejects_empty_event():
    with pytest.raises(ValueError, match="Cannot append an empty TurnEvent"):
        TurnLog.empty().append_turn(TurnEvent())


def test_turn_log_immutable_after_append():
    event = TurnEvent(thought="original")
    log = TurnLog.empty().append_turn(event)
    with pytest.raises(ValidationError):
        event.thought = "mutated"
    turn = log.turns[0]
    assert isinstance(turn, TurnEvent)
    assert turn.thought == "original"


def test_turn_log_coerces_dict_turns_on_load():
    log = TurnLog.model_validate({"turns": [{"thought": "legacy"}]})
    assert isinstance(log.turns[0], TurnEvent)
    assert log.turns[0].thought == "legacy"
