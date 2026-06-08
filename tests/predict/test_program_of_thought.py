import asyncio
from unittest.mock import patch

import pytest

from dspy.predict.program_of_thought import ProgramOfThought
from dspy.task_spec import FieldSpec, make_task_spec
from dspy.utils.dummies import DummyLM

BasicQA = make_task_spec(
    {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="often between 1 and 5 words")},
    instructions="Answer the question.",
    name="BasicQA",
)
ExtremumFinder = make_task_spec(
    {
        "input_list": FieldSpec.input("input_list"),
        "maximum": FieldSpec.output("maximum", desc="The maximum of the given numbers"),
        "minimum": FieldSpec.output("minimum", desc="The minimum of the given numbers"),
    },
    instructions="Find the maximum and minimum values.",
    name="ExtremumFinder",
)


@pytest.mark.deno
def test_pot_code_generation(make_run):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    run = make_run(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = asyncio.run(pot(question="What is 1+1?", run=run))
    assert res.answer == "2"
    assert pot.interpreter.deno_process is None


@pytest.mark.deno
def test_old_style_pot(make_run):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    run = make_run(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = asyncio.run(pot(question="What is 1+1?", run=run))
    assert res.answer == "2"
    assert pot.interpreter.deno_process is None


@pytest.mark.deno
def test_pot_support_multiple_fields(make_run):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nmaximum = 6\nminimum = 2\nSUBMIT({'maximum': maximum, 'minimum': minimum})\n```",
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    run = make_run(lm=lm)
    pot = ProgramOfThought(ExtremumFinder)
    res = asyncio.run(pot(input_list="2, 3, 5, 6", run=run))
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert pot.interpreter.deno_process is None


@pytest.mark.deno
def test_pot_code_generation_with_one_error(make_run):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```"},
            {"reasoning": "Reason_B", "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```"},
            {"reasoning": "Reason_C", "answer": "2"},
        ]
    )
    run = make_run(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = asyncio.run(pot(question="What is 1+1?", run=run))
    assert res.answer == "2"
    assert pot.interpreter.deno_process is None


@pytest.mark.deno
def test_pot_code_generation_persistent_errors(make_run):
    max_iters = 3
    lm = DummyLM(
        [{"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```"}]
        * max_iters
    )
    run = make_run(lm=lm)
    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with pytest.raises(RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: ZeroDivisionError:"):
        asyncio.run(pot(question="What is 1+1?", run=run))


def test_pot_code_parse_error(make_run):
    max_iters = 3
    lm = DummyLM([{"reasoning": "Reason_A", "generated_code": "```python\ninvalid=python=code\n```"}] * max_iters)
    run = make_run(lm=lm)
    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with (
        patch("dspy.predict.program_of_thought.ProgramOfThought._execute_code") as mock_execute_code,
        pytest.raises(
            RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: Error: Code format is not correct."
        ),
    ):
        asyncio.run(pot(question="What is 1+1?", run=run))
    mock_execute_code.assert_not_called()
