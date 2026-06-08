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
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.utils.dummies import DummyLM
from dspy.utils.inspect_history import pretty_print_history
from tests.task_spec.helpers import ts


@pytest.fixture(autouse=True)
def clear_history():
    GLOBAL_HISTORY.clear()
    return


def test_inspect_history_basic(capsys):
    lm = DummyLM([{"response": "Hello"}, {"response": "How are you?"}])
    settings.configure(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor.acall(query="Hi"))
    asyncio.run(predictor.acall(query="What's up?"))
    history = GLOBAL_HISTORY
    assert len(history) > 0
    assert isinstance(history, list)
    assert all(isinstance(entry, LMHistoryEntry) for entry in history)
    assert all(entry.messages for entry in history)


def test_inspect_history_renders_message_tool_calls():
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


def test_inspect_history_renders_output_tool_calls_without_text():
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


def test_inspect_history_with_n(capsys):
    lm = DummyLM([{"response": "One"}, {"response": "Two"}, {"response": "Three"}])
    settings.configure(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor.acall(query="First"))
    asyncio.run(predictor.acall(query="Second"))
    asyncio.run(predictor.acall(query="Third"))
    inspect_history(n=2)
    out, _err = capsys.readouterr()
    assert "First" not in out
    assert "Second" in out
    assert "Third" in out


def test_inspect_empty_history(capsys):
    lm = DummyLM([])
    settings.configure(lm=lm)
    inspect_history()
    history = GLOBAL_HISTORY
    assert len(history) == 0
    assert isinstance(history, list)


def test_inspect_history_n_larger_than_history(capsys):
    lm = DummyLM([{"response": "First"}, {"response": "Second"}])
    settings.configure(lm=lm)
    predictor = Predict(ts("query: str -> response: str"))
    asyncio.run(predictor.acall(query="Query 1"))
    asyncio.run(predictor.acall(query="Query 2"))
    inspect_history(n=5)
    history = GLOBAL_HISTORY
    assert len(history) == 2
