from __future__ import annotations

from typing import Any, Literal, cast

from typing_extensions import override

from dspy.adapters.types.audio import Audio
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.code import Code
from dspy.adapters.types.document import Document
from dspy.adapters.types.field_type import FieldTypeMixin
from dspy.adapters.types.file import File
from dspy.adapters.types.image import Image
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.core.types import NativeAdaptationMode
from dspy.history import TurnLog
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.adapters.scenarios.tools import search_tool
from tests.history.turn_fixtures import react_v2_turn, task_io_turn
from tests.task_spec.helpers import ts
from tests.test_utils import DummyLM


class Event(FieldTypeMixin):
    label: str

    @override
    def format(self):
        return [{"type": "event", "event": {"label": self.label}}]

    @classmethod
    @override
    def description(cls) -> str:
        return "An event block."


class AnthropicLM(DummyLM):
    def __init__(self):
        super().__init__([{}])
        self.model = "anthropic/claude-3-5-sonnet"

    @property
    def citations_adaptation_mode(self):
        return NativeAdaptationMode.SKIP


class ReasoningLM(DummyLM):
    def __init__(self, answers):
        super().__init__(answers)
        self.kwargs["reasoning"] = {"effort": "low"}

    @property
    @override
    def supports_reasoning(self):
        return True


class FunctionCallingLM(DummyLM):
    @property
    @override
    def supports_function_calling(self):
        return True


def history_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `turn_log`, `question`, produce the fields `answer`.",
    )
    history = TurnLog.model_validate(
        {
            "turns": [
                task_io_turn(question="What is 1+1?", answer="2"),
                task_io_turn(question="What is 2+2?", answer="4"),
            ],
        }
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"turn_log": history, "question": "What is 3+3?"},
    )


def list_value_string_case() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts("context -> answer", instructions="Given the fields `context`, produce the fields `answer`."),
        demos=(),
        inputs={"context": ["alpha", "beta"]},
    )


def literal_output_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "verdict": output_field("verdict", type_=Literal["yes", "no"], desc="The verdict."),
        },
        instructions="Given the fields `question`, produce the fields `verdict`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Is the sky blue?"},
    )


def multimodal_custom_type_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "image": input_field("image", type_=Image, desc="The image."),
            "audio": input_field("audio", type_=Audio, desc="The audio."),
            "file": input_field("file", type_=File, desc="The file."),
            "document": input_field("document", type_=Document, desc="The document."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `image`, `audio`, `file`, `document`, produce the fields `answer`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={
            "image": Image("https://example.com/cat.png"),
            "audio": Audio(data="QUJD", audio_format="wav"),
            "file": File.from_file_id("file-123", filename="notes.txt"),
            "document": Document(data="Alpha beta", title="Doc"),
        },
    )


def base_custom_type_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "event": input_field("event", type_=Event, desc="The event."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `event`, produce the fields `answer`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"event": Event(label="launch")},
    )


def citations_output_demo_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "citations": output_field("citations", type_=Citations, desc="The citations."),
        },
        instructions="Given the fields `question`, produce the fields `citations`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(
            {
                "question": "Q1",
                "citations": Citations.from_dict_list(
                    [{"cited_text": "alpha", "document_index": 0, "start_char_index": 0, "end_char_index": 5}]
                ),
            },
        ),
        inputs={"question": "Q2"},
    )


def native_citations_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
            "citations": output_field("citations", type_=Citations, desc="The citations."),
        },
        instructions="Given the fields `question`, produce the fields `answer`, `citations`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Q?"},
        lm=AnthropicLM(),
    )


def passthrough_lm_kwargs_case() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`."),
        demos=(),
        inputs={"question": "Q?"},
        config={"temperature": 0.7, "max_tokens": 42, "extensions": {"stream": True}},
    )


def native_reasoning_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `answer`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Q?"},
        lm=ReasoningLM([{}]),
    )


def reasoning_code_outputs_case() -> FormatScenarioCase:
    python_code = cast("Any", Code)["python"]
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "code": output_field("code", type_=python_code, desc="The code."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `code`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"question": "Q1", "reasoning": Reasoning(content="Think"), "code": python_code(code="print('hi')")},),
        inputs={"question": "Q2"},
    )


def native_tool_calling_case() -> FormatScenarioCase:
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


def non_native_tool_history_case() -> FormatScenarioCase:
    def search(query: str) -> str:
        return query

    tool = Tool(search, description="Search for documents.")
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "next_thought": output_field("next_thought", desc="The next thought."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `turn_log`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = ToolCallResults.from_tool_calls_and_values([tool_call], ["cat"])
    history = TurnLog.model_validate(
        {
            "turns": [
                react_v2_turn(
                    pending_inputs={"question": "Q1"},
                    next_thought="I should search.",
                    tool_calls=ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                )
            ],
        }
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Q2", "turn_log": history, "tools": [tool]},
    )


def tool_input_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Q?", "tools": [search_tool()]},
    )
