import asyncio
from io import StringIO

import pytest

from dspy.clients.base_lm import GLOBAL_HISTORY, inspect_history
from dspy.core.types import (
    Assistant,
    LMHistoryEntry,
    LMMessage,
    LMOutput,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMToolCallPart,
    LMToolResultPart,
    ToolCall,
    User,
)
from dspy.predict.predict import Predict
from dspy.utils.dummies import DummyLM
from dspy.utils.inspect_history import pretty_print_history
from tests.task_spec.helpers import ts


@pytest.fixture(autouse=True)
def clear_history():
    GLOBAL_HISTORY.clear()
    return


def test_inspect_history_basic(capsys, make_run):
    lm = DummyLM([{"response": "Hello"}, {"response": "How are you?"}])
    run = make_run(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor.acall(query="Hi", run=run))
    asyncio.run(predictor.acall(query="What's up?", run=run))
    history = GLOBAL_HISTORY
    assert len(history) > 0
    assert isinstance(history, list)
    assert all(isinstance(entry, LMHistoryEntry) for entry in history)
    assert all(entry.messages for entry in history)


def test_inspect_history_renders_message_tool_calls(make_run):
    out = StringIO()
    history = [
        LMHistoryEntry(
            request=LMRequest(
                model="test",
                messages=[
                    Assistant(ToolCall(id="call_1", name="search", args={"query": "cats"})),
                    LMMessage(
                        role="tool",
                        parts=[
                            LMToolResultPart(call_id="call_1", name="search", content=[LMTextPart(text="cat result")])
                        ],
                    ),
                ],
            ),
            response=LMResponse.from_text("done"),
            timestamp="now",
            uuid="1",
        )
    ]
    pretty_print_history(history, n=1, file=out)
    text = out.getvalue()
    assert "Assistant message:" in text
    assert "Tool calls:" in text
    assert 'search: {"query": "cats"}' in text
    assert "Tool message:" in text
    assert "cat result" in text


def test_inspect_history_renders_output_tool_calls_without_text(make_run):
    out = StringIO()
    history = [
        LMHistoryEntry(
            request=LMRequest(model="test", messages=[User("Find cats")]),
            response=LMResponse(
                model="test",
                outputs=[
                    LMOutput(
                        parts=[
                            LMToolCallPart(id="call_1", name="lookup", args={"query": "cats"}),
                            LMToolCallPart(id="call_2", name="search", args={"query": "dogs"}),
                        ],
                        provider_output={
                            "tool_calls": [
                                {"name": "lookup", "arguments": {"query": "cats"}},
                                {"name": "search", "args": {"query": "dogs"}},
                            ]
                        },
                    )
                ],
            ),
            timestamp="now",
            uuid="1",
        )
    ]
    pretty_print_history(history, n=1, file=out)
    text = out.getvalue()
    assert "Response:" not in text
    assert "Tool calls:" in text
    assert 'lookup: {"query": "cats"}' in text
    assert 'search: {"query": "dogs"}' in text


def test_inspect_history_with_n(capsys, make_run):
    lm = DummyLM([{"response": "One"}, {"response": "Two"}, {"response": "Three"}])
    run = make_run(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor.acall(query="First", run=run))
    asyncio.run(predictor.acall(query="Second", run=run))
    asyncio.run(predictor.acall(query="Third", run=run))
    inspect_history(n=2)
    out, _err = capsys.readouterr()
    assert "First" not in out
    assert "Second" in out
    assert "Third" in out


def test_inspect_empty_history(capsys, make_run):
    lm = DummyLM([])
    run = make_run(lm=lm)
    inspect_history()
    history = GLOBAL_HISTORY
    assert len(history) == 0
    assert isinstance(history, list)


def test_inspect_history_n_larger_than_history(capsys, make_run):
    lm = DummyLM([{"response": "First"}, {"response": "Second"}])
    run = make_run(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor.acall(query="Query 1", run=run))
    asyncio.run(predictor.acall(query="Query 2", run=run))
    inspect_history(n=5)
    history = GLOBAL_HISTORY
    assert len(history) == 2
