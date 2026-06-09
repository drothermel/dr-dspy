import asyncio
from unittest import mock

import pytest

try:
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.task_spec import default_task_instructions
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


def test_initialization_with_string_signature(make_run):
    lm = DummyLM([{"reasoning": "find the number after 1", "answer": "2"}])
    run = make_run(lm=lm)
    predict = ChainOfThought(
        ts("question -> answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))
    )
    assert list(predict.predict.task_spec.output_fields.keys()) == ["reasoning", "answer"]
    assert asyncio.run(predict(question="What is 1+1?", run=run)).answer == "2"


@pytest.mark.asyncio
async def test_async_chain_of_thought(make_run):
    lm = DummyLM([{"reasoning": "find the number after 1", "answer": "2"}])
    run = make_run(lm=lm)
    program = ChainOfThought(
        ts("question -> answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))
    )
    result = await program.acall(question="What is 1+1?", run=run)
    assert result.answer == "2"


def test_chain_of_thought_with_native_reasoning(make_run):
    lm = LM(
        model="anthropic/claude-3-7-sonnet-20250219",
        temperature=0.0,
        max_tokens=4000,
        cache=False,
        reasoning_effort="low",
    )
    run = make_run(lm=lm, adapter=ChatAdapter())
    with mock.patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="[[ ## answer ## ]]\nParis\n[[ ## completion ## ]]",
                        reasoning_content="Step-by-step thinking about the capital of France",
                    )
                )
            ],
            model="anthropic/claude-3-7-sonnet-20250219",
        )
        cot = ChainOfThought(
            ts("question -> answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))
        )
        result = asyncio.run(cot(question="What is the capital of France?", run=run))
        assert result.answer == "Paris"
        assert result.reasoning == "Step-by-step thinking about the capital of France"
        _args, _kwargs = mock_completion.call_args


def test_chain_of_thought_with_manual_reasoning(make_run):
    lm = LM(model="openai/gpt-4o-mini")
    run = make_run(lm=lm)
    with mock.patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    reasoning="Step-by-step thinking about the capital of France",
                    message=Message(
                        content="[[ ## reasoning ## ]]\nStep-by-step thinking about the capital of France\n[[ ## answer ## ]]\nParis\n[[ ## completion ## ]]"
                    ),
                )
            ],
            model="openai/gpt-4o-mini",
        )
        cot = ChainOfThought(
            ts("question -> answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))
        )
        result = asyncio.run(cot(question="What is the capital of France?", run=run))
        assert result.answer == "Paris"
        assert result.reasoning == "Step-by-step thinking about the capital of France"
