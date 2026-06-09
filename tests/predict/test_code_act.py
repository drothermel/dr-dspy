import asyncio
from typing import Any, cast
from unittest.mock import patch

import pytest

from dspy.adapters.types.tool import Tool
from dspy.predict.agent_termination import AgentTerminationReason
from dspy.predict.code_act import CodeAct
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM

pytestmark = pytest.mark.deno
BasicQA = make_task_spec(
    {
        "question": input_field("question", desc="The question."),
        "answer": output_field("answer", desc="often between 1 and 5 words"),
    },
    instructions="Answer the question.",
    name="BasicQA",
)
ExtremumFinder = make_task_spec(
    {
        "input_list": input_field("input_list", desc="The input list."),
        "maximum": output_field("maximum", desc="The maximum of the given numbers"),
        "minimum": output_field("minimum", desc="The minimum of the given numbers"),
    },
    instructions="Find the maximum and minimum values.",
    name="ExtremumFinder",
)


def add(a: float, b: float) -> float:
    return a + b


ADD_TOOL = Tool(add, description="Add two numbers.")


def test_codeact_code_generation(make_run):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    run = make_run(lm=lm)
    program = CodeAct(BasicQA, tools=[ADD_TOOL])
    res = asyncio.run(program(question="What is 1+1?", run=run))
    assert res.answer == "2"
    assert res.termination_reason == AgentTerminationReason.SUBMIT
    assert res.turn_log.turns == ({"generated_code": "result = add(1,1)\nprint(result)", "code_output": '"2\\n"'},)
    assert program.interpreter.deno_process is None


def extract_maximum_minimum(input_list: str) -> dict[str, float]:
    numbers = list(map(float, input_list.split(",")))
    return {"maximum": max(numbers), "minimum": min(numbers)}


EXTRACT_TOOL = Tool(extract_maximum_minimum, description="Extract maximum and minimum from a comma-separated list.")


def test_codeact_support_multiple_fields(make_run):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    run = make_run(lm=lm)
    program = CodeAct(ExtremumFinder, tools=[EXTRACT_TOOL])
    res = asyncio.run(program(input_list="2, 3, 5, 6", run=run))
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert res.turn_log.turns == (
        {
            "generated_code": "result = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)",
            "code_output": "\"{'maximum': 6.0, 'minimum': 2.0}\\n\"",
        },
    )
    assert program.interpreter.deno_process is None


def test_codeact_code_parse_failure(make_run):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nparse(error\n```", "finished": False},
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    run = make_run(lm=lm)
    program = CodeAct(BasicQA, tools=[ADD_TOOL])
    res = asyncio.run(program(question="What is 1+1?", run=run))
    assert res.answer == "2"
    assert res.turn_log.turns == (
        {
            "generated_code": "parse(error",
            "observation": "Failed to execute the generated code: Invalid Python syntax. message: ",
        },
        {"generated_code": "result = add(1,1)\nprint(result)", "code_output": '"2\\n"'},
    )
    assert program.interpreter.deno_process is None


def test_codeact_code_execution_failure(make_run):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nunknown+1\n```", "finished": False},
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    run = make_run(lm=lm)
    program = CodeAct(BasicQA, tools=[ADD_TOOL])
    res = asyncio.run(program(question="What is 1+1?", run=run))
    assert res.answer == "2"
    assert res.turn_log.turns == (
        {
            "generated_code": "unknown+1",
            "observation": "Failed to execute the generated code: NameError: [\"name 'unknown' is not defined\"]",
        },
        {"generated_code": "result = add(1,1)\nprint(result)", "code_output": '"2\\n"'},
    )
    assert program.interpreter.deno_process is None


class CustomTool:
    def __call__(self, a: float, b: float) -> float:
        return a + b


def test_codeact_tool_validation_requires_tool_instances():
    with pytest.raises(TypeError, match="tools must be Tool instances"):
        CodeAct(BasicQA, tools=cast("Any", [add]))


def test_codeact_tool_validation_rejects_callable_objects():
    with pytest.raises(ValueError, match=r"CodeAct only accepts functions and not callable objects\."):
        CodeAct(BasicQA, tools=[Tool(CustomTool(), description="Add two numbers.")])


def test_codeact_shuts_down_interpreter_when_extractor_raises(make_run):
    from dspy.history import call_with_turn_log_truncation

    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
        ]
    )
    run = make_run(lm=lm)
    program = CodeAct(BasicQA, tools=[ADD_TOOL])

    original = call_with_turn_log_truncation

    async def failing_extractor(module, *args, **kwargs):
        if module is program.extractor:
            raise RuntimeError("extractor failed")
        return await original(module, *args, **kwargs)

    with (
        patch("dspy.predict.code_act.call_with_turn_log_truncation", side_effect=failing_extractor),
        pytest.raises(RuntimeError, match="extractor failed"),
    ):
        asyncio.run(program(question="What is 1+1?", run=run))
    assert program.interpreter.deno_process is None
