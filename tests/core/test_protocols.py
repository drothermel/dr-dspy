from dspy.core.types.lm import LMForward
from dspy.history import ConversationTurnLog, TurnEvent, TurnLog
from dspy.predict.predict import Predict
from dspy.predict.protocol import Predictor
from dspy.utils.dummies import DummyLM
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


def test_turn_log_coerces_dict_turns_on_load():
    log = TurnLog(turns=({"thought": "legacy"},))
    assert isinstance(log.turns[0], TurnEvent)
    assert log.turns[0].thought == "legacy"
