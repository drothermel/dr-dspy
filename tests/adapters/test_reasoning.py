import asyncio
from typing import Any, cast

import pytest

from dspy.adapters.types.reasoning import Reasoning
from dspy.predict.chain_of_thought import ChainOfThought
from tests.task_spec.helpers import ts


def test_reasoning_basic_operations(make_run):
    reasoning = Reasoning(content="Hello World")
    assert str(reasoning) == "Hello World"
    assert repr(reasoning) == "'Hello World'"
    assert reasoning == "Hello World"
    assert reasoning == Reasoning(content="Hello World")
    assert reasoning != "hello world"
    assert reasoning != Reasoning(content="hello world")
    assert len(reasoning) == 11
    assert reasoning[0] == "H"
    assert reasoning[-1] == "d"
    assert reasoning[0:5] == "Hello"
    assert "World" in reasoning
    assert "xyz" not in reasoning
    chars = list(reasoning)
    assert len(chars) == 11
    assert chars[0] == "H"


def test_reasoning_concatenation(make_run):
    reasoning = Reasoning(content="Hello")
    result1 = reasoning + " World"
    assert result1 == "Hello World"
    assert isinstance(result1, str)
    result2 = "Prefix: " + reasoning
    assert result2 == "Prefix: Hello"
    assert isinstance(result2, str)
    reasoning2 = Reasoning(content=" World")
    result3 = reasoning + reasoning2
    assert isinstance(result3, Reasoning)
    assert result3.content == "Hello World"


def test_reasoning_string_methods(make_run):
    reasoning = Reasoning(content="  Hello World  ")
    text_ops = cast("Any", reasoning)
    assert text_ops.strip() == "Hello World"
    assert text_ops.lower() == "  hello world  "
    assert text_ops.upper() == "  HELLO WORLD  "
    assert text_ops.strip().split() == ["Hello", "World"]
    assert text_ops.strip().split(" ") == ["Hello", "World"]
    assert text_ops.replace("World", "Python") == "  Hello Python  "
    assert text_ops.strip().startswith("Hello")
    assert text_ops.strip().endswith("World")
    assert not text_ops.strip().startswith("World")
    assert text_ops.find("World") == 8
    assert text_ops.find("xyz") == -1
    assert text_ops.count("l") == 3
    assert text_ops.strip().join(["a", "b", "c"]) == "aHello WorldbHello Worldc"


def test_reasoning_with_chain_of_thought(make_run):
    from dspy.testing import DummyLM

    lm = DummyLM([{"reasoning": "Let me think step by step", "answer": "42"}])
    run = make_run(lm=lm)
    cot = ChainOfThought(ts("question -> answer"))
    result = asyncio.run(cot(question="What is the answer?", run=run))
    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.strip() == "Let me think step by step"
    assert result.reasoning.lower() == "let me think step by step"
    assert "step by step" in result.reasoning
    assert len(result.reasoning) == 25


def test_reasoning_error_message():
    reasoning = Reasoning(content="Hello")

    def access_missing_method() -> None:
        reasoning.nonexistent_method  # noqa: B018

    with pytest.raises(AttributeError, match="`Reasoning` object has no attribute 'nonexistent_method'"):
        access_missing_method()
