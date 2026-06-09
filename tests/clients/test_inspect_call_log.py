import asyncio
from io import StringIO

from dspy.core.types import (
    CallRecord,
    LMMessage,
    LMMessageRole,
    LMOutput,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMToolCallPart,
    User,
)
from dspy.predict.predict import Predict
from dspy.runtime.inspect_call_log import pretty_print_call_log
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def test_inspect_call_log_basic(capsys, make_run):
    lm = DummyLM([{"response": "Hello"}, {"response": "How are you?"}])
    run = make_run(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor(query="Hi", run=run))
    asyncio.run(predictor(query="What's up?", run=run))
    call_log = run.call_log
    assert len(call_log) > 0
    assert isinstance(call_log, list)
    assert all(isinstance(entry, CallRecord) for entry in call_log)
    assert all(entry.messages for entry in call_log)


def test_inspect_call_log_renders_message_tool_calls(make_run):
    out = StringIO()
    history = [
        CallRecord(
            request=LMRequest(
                model="test",
                messages=[
                    LMMessage(
                        role=LMMessageRole.USER,
                        parts=[LMTextPart(text="Use a tool")],
                    )
                ],
            ),
            response=LMResponse(
                model="test",
                outputs=[
                    LMOutput(
                        parts=[
                            LMToolCallPart(name="search", args={"query": "cats"}, id="call_1"),
                        ]
                    )
                ],
            ),
            timestamp="2024-01-01T00:00:00",
            uuid="uuid-1",
        )
    ]
    pretty_print_call_log(history, n=1, file=out)
    rendered = out.getvalue()
    assert "Tool calls:" in rendered
    assert "search" in rendered
    assert "cats" in rendered


def test_inspect_call_log_renders_output_tool_calls_without_text(make_run):
    out = StringIO()
    history = [
        CallRecord(
            request=LMRequest(model="test", messages=[User("Use a tool")]),
            response=LMResponse(
                model="test",
                outputs=[
                    LMOutput(
                        parts=[
                            LMToolCallPart(name="search", args={"query": "cats"}, id="call_1"),
                        ]
                    )
                ],
            ),
            timestamp="2024-01-01T00:00:00",
            uuid="uuid-1",
        )
    ]
    pretty_print_call_log(history, n=1, file=out)
    rendered = out.getvalue()
    assert "Tool calls:" in rendered
    assert "search" in rendered


def test_inspect_call_log_with_n(capsys, make_run):
    lm = DummyLM([{"response": "Hello"}, {"response": "How are you?"}, {"response": "Goodbye"}])
    run = make_run(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    for query in ("Hi", "What's up?", "Bye"):
        asyncio.run(predictor(query=query, run=run))
    run.inspect_call_log(n=2)
    captured = capsys.readouterr()
    assert captured.out.count("[") >= 2


def test_run_inspect_call_log(capsys, make_run):
    lm = DummyLM([{"response": "Hello"}])
    run = make_run(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor(query="Hi", run=run))
    run.inspect_call_log()
    call_log = run.call_log
    assert len(call_log) == 1


def test_inspect_call_log_preserves_empty_messages():
    out = StringIO()
    history = [
        CallRecord(
            request=LMRequest(model="test", messages=[]),
            response=LMResponse.from_text("ok", model="test"),
            timestamp="2024-01-01T00:00:00",
            uuid="uuid-empty-messages",
        )
    ]
    pretty_print_call_log(history, n=1, file=out)
    assert "User message:" not in out.getvalue()


def test_inspect_call_log_handles_empty_outputs():
    out = StringIO()
    history = [
        CallRecord(
            request=LMRequest(model="test", messages=[User("hi")]),
            response=LMResponse.model_construct(model="test", outputs=[]),
            timestamp="2024-01-01T00:00:00",
            uuid="uuid-empty-outputs",
        )
    ]
    pretty_print_call_log(history, n=1, file=out)
    assert "Response:" not in out.getvalue()


def test_inspect_call_log_n_larger_than_history(capsys, make_run):
    lm = DummyLM([{"response": "Hello"}])
    run = make_run(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor(query="Hi", run=run))
    run.inspect_call_log(n=5)
    assert len(run.call_log) == 1
