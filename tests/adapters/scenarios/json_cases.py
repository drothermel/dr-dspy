from __future__ import annotations

import enum
from typing import Literal

from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.adapters.scenarios.chat_cases import FunctionCallingLM, non_native_tool_history_case
from tests.adapters.scenarios.tools import search_tool


class Label(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


def described_bool_outputs_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "input1": input_field("input1", desc="The input 1."),
            "output1": output_field("output1", desc="String output field"),
            "output2": output_field("output2", type_=bool, desc="The output 2."),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`.",
    )
    return FormatScenarioCase(task_spec=task_spec, demos=(), inputs={"input1": "Test input"})


def int_mapping_outputs_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "count": output_field("count", type_=int, desc="The count."),
            "metadata": output_field("metadata", type_=dict[str, int], desc="The metadata."),
        },
        instructions="Given the fields `question`, produce the fields `count`, `metadata`.",
    )
    return FormatScenarioCase(task_spec=task_spec, demos=(), inputs={"question": "Count things"})


def literal_enum_outputs_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "text": input_field("text", desc="The text."),
            "decision": output_field("decision", type_=Literal["accept", "reject"], desc="The decision."),
            "label": output_field("label", type_=Label, desc="The label."),
        },
        instructions="Given the fields `text`, produce the fields `decision`, `label`.",
    )
    return FormatScenarioCase(task_spec=task_spec, demos=(), inputs={"text": "Looks good"})


def json_native_tool_calling_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Q?", "tools": [search_tool()]},
        lm=FunctionCallingLM([{}]),
    )


def tool_calls_output_demo_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, produce the fields `tool_calls`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(
            {
                "question": "Q1",
                "tool_calls": ToolCalls.from_dict_list([{"name": "search", "args": {"query": "cats"}}]),
            },
        ),
        inputs={"question": "Q2"},
    )


def json_non_native_tool_history_case() -> FormatScenarioCase:
    return non_native_tool_history_case()
