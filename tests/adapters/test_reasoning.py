import asyncio

import pytest

from dspy.adapters.types.reasoning import Reasoning
from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought


def test_reasoning_basic_operations():
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


def test_reasoning_concatenation():
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


def test_reasoning_string_methods():
    reasoning = Reasoning(content="  Hello World  ")

    assert reasoning.strip() == "Hello World"  # ty:ignore[call-non-callable]

    assert reasoning.lower() == "  hello world  "  # ty:ignore[call-non-callable]
    assert reasoning.upper() == "  HELLO WORLD  "  # ty:ignore[call-non-callable]

    assert reasoning.strip().split() == ["Hello", "World"]  # ty:ignore[call-non-callable]
    assert reasoning.strip().split(" ") == ["Hello", "World"]  # ty:ignore[call-non-callable]

    assert reasoning.replace("World", "Python") == "  Hello Python  "  # ty:ignore[call-non-callable]

    assert reasoning.strip().startswith("Hello")  # ty:ignore[call-non-callable]
    assert reasoning.strip().endswith("World")  # ty:ignore[call-non-callable]
    assert not reasoning.strip().startswith("World")  # ty:ignore[call-non-callable]

    assert reasoning.find("World") == 8  # ty:ignore[call-non-callable]
    assert reasoning.find("xyz") == -1  # ty:ignore[call-non-callable]

    assert reasoning.count("l") == 3  # ty:ignore[call-non-callable]

    assert reasoning.strip().join(["a", "b", "c"]) == "aHello WorldbHello Worldc"  # ty:ignore[call-non-callable]


def test_reasoning_with_chain_of_thought():
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([{"reasoning": "Let me think step by step", "answer": "42"}])
    settings.configure(lm=lm)

    cot = ChainOfThought("question -> answer")
    result = asyncio.run(cot.acall(question="What is the answer?"))

    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.strip() == "Let me think step by step"
    assert result.reasoning.lower() == "let me think step by step"
    assert "step by step" in result.reasoning
    assert len(result.reasoning) == 25


def test_reasoning_error_message():
    reasoning = Reasoning(content="Hello")

    with pytest.raises(AttributeError, match="`Reasoning` object has no attribute 'nonexistent_method'"):
        reasoning.nonexistent_method  # noqa: B018
