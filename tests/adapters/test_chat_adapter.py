import asyncio
import json
from typing import Literal
from unittest import mock

import pydantic
import pytest
from typing_extensions import override

from tests.test_utils import DummyLM

try:
    from litellm.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.code import Code
from dspy.adapters.types.field_type import FieldTypeMixin
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.clients.lm import LM
from dspy.clients.openai_format.parse import provider_tool_call_to_part
from dspy.core.types import LMOutput, LMPart, LMResponse, LMTextPart, LMThinkingPart
from dspy.errors import AdapterParseError
from dspy.history import TurnLog
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.conftest import (
    adapter_format_as_openai,
    default_model_response,
    format_messages_and_lm_kwargs,
    make_adapter_run,
)
from tests.history.turn_fixtures import react_v2_turn, task_io_turn
from tests.task_spec.helpers import ts


def outputs_to_lm_response(outputs: list[dict]) -> LMResponse:
    lm_outputs = []
    for output in outputs:
        parts: list[LMPart] = []
        text = output.get("text")
        if isinstance(text, str):
            parts.append(LMTextPart(text=text))
        reasoning = output.get("reasoning_content")
        if isinstance(reasoning, str):
            parts.append(LMThinkingPart(text=reasoning))
        parts.extend(provider_tool_call_to_part(tool_call) for tool_call in output.get("tool_calls") or [])
        lm_outputs.append(LMOutput(parts=parts, provider_output=output))
    return LMResponse(model="test", outputs=lm_outputs)


@pytest.mark.parametrize(
    ("input_literal", "output_literal", "input_value", "expected_input_str", "expected_output_str"),
    [
        (
            Literal["one", "two", 'three"'],
            Literal["four", "five", 'six"'],
            "two",
            "Literal['one', 'two', 'three\"']",
            "Literal['four', 'five', 'six\"']",
        ),
        (
            Literal["she's here", "okay", "test"],
            Literal["done", "maybe'soon", "later"],
            "she's here",
            "Literal[\"she's here\", 'okay', 'test']",
            "Literal['done', \"maybe'soon\", 'later']",
        ),
        (
            Literal["both\"and'", "another"],
            Literal["yet\"another'", "plain"],
            "another",
            "Literal['both\"and\\'', 'another']",
            "Literal['yet\"another\\'', 'plain']",
        ),
        (Literal["foo", "bar"], Literal["baz", "qux"], "foo", "Literal['foo', 'bar']", "Literal['baz', 'qux']"),
        (Literal[1, "bar"], Literal[True, 3, "foo"], "bar", "Literal[1, 'bar']", "Literal[True, 3, 'foo']"),
    ],
)
def test_chat_adapter_quotes_literals_as_expected(
    input_literal, output_literal, input_value, expected_input_str, expected_output_str
):
    TestSignature = make_task_spec(
        {
            "input_text": input_field("input_text", type_=input_literal, desc="The input text."),
            "output_text": output_field("output_text", type_=output_literal, desc="The output text."),
        },
        instructions="Given the fields `input_text`, produce the fields `output_text`.",
    )
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=TestSignature, demos=[], inputs={"input_text": input_value}
    )
    content = messages[0]["content"]
    assert expected_input_str in content
    assert expected_output_str in content


def test_chat_adapter_sync_call():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = ChatAdapter()
    lm = DummyLM([{"answer": "Paris"}])
    result = asyncio.run(
        adapter(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "What is the capital of France?"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    )
    assert result == [{"answer": "Paris"}]


@pytest.mark.asyncio
async def test_chat_adapter_async_call():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = ChatAdapter()
    lm = DummyLM([{"answer": "Paris"}])
    result = await adapter(
        lm=lm,
        config={},
        task_spec=signature,
        demos=[],
        inputs={"question": "What is the capital of France?"},
        run=make_adapter_run(lm=lm, adapter=adapter),
    )
    assert result == [{"answer": "Paris"}]


def test_chat_adapter_native_tool_calling_still_enables_native_reasoning():

    class NativeToolReasoningLM(DummyLM):
        def __init__(self, answers):
            super().__init__(answers)
            self.kwargs["reasoning"] = {"effort": "low"}

        @property
        @override
        def supports_function_calling(self):
            return True

        @property
        @override
        def supports_reasoning(self):
            return True

    def search(query: str) -> str:
        return query

    NativeToolReasoningSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "next_thought": output_field("next_thought", type_=Reasoning, desc="The next thought."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    _, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=True),
        task_spec=NativeToolReasoningSignature,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
        lm=NativeToolReasoningLM([{}]),
    )
    assert "tools" in lm_kwargs
    assert lm_kwargs["reasoning_effort"] == "low"


def test_chat_adapter_nonnative_strips_native_tool_kwargs():

    def search(query: str) -> str:
        return query

    NonNativeToolSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    _, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=False),
        task_spec=NonNativeToolSignature,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
        config={"tool_choice": {"mode": "required", "allowed": ["submit"], "parallel": True}},
    )
    assert "tools" not in lm_kwargs
    assert "tool_choice" not in lm_kwargs
    assert "parallel_tool_calls" not in lm_kwargs


@pytest.mark.parametrize(
    "adapter",
    [
        ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True),
        JSONAdapter(use_native_function_calling=True, parallel_tool_calls=True),
        XMLAdapter(use_native_function_calling=True, parallel_tool_calls=True),
    ],
)
def test_adapter_native_tool_calling_can_request_parallel_tool_calls(adapter):

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str) -> str:
        return query

    NativeToolSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    _messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter,
        task_spec=NativeToolSignature,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
        lm=FunctionCallingLM([{}]),
    )
    assert lm_kwargs["tool_choice"] == "auto"
    assert lm_kwargs["parallel_tool_calls"] is True


def test_adapter_native_tool_calling_respects_lm_kwargs_parallel_tool_call_override():

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str) -> str:
        return query

    NativeToolSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    _messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True),
        task_spec=NativeToolSignature,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
        config={"tool_choice": {"mode": "auto", "parallel": False}},
        lm=FunctionCallingLM([{}]),
    )
    assert lm_kwargs["tool_choice"] == "auto"
    assert lm_kwargs["parallel_tool_calls"] is False


def test_chat_adapter_native_tool_history_replay():

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str) -> str:
        return query

    NativeToolHistorySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "next_thought": output_field("next_thought", type_=Reasoning, desc="The next thought."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `turn_log`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = ToolCallResults.from_tool_calls_and_values([tool_call], [{"items": ["cat"]}])
    history = TurnLog.model_validate(
        {
            "turns": [
                react_v2_turn(
                    pending_inputs={"question": "Q1"},
                    next_thought=Reasoning(content="I should search."),
                    tool_calls=ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                )
            ],
        }
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=True),
        task_spec=NativeToolHistorySignature,
        demos=[],
        inputs={"question": "Q2", "turn_log": history, "tools": [Tool(search, description="Search for documents.")]},
        lm=FunctionCallingLM([{}]),
    )
    assert messages[1]["role"] == "user"
    assert "Q1" in messages[1]["content"]
    assert messages[2] == {
        "role": "assistant",
        "content": "[[ ## next_thought ## ]]\nI should search.\n\n[[ ## completed ## ]]\n",
        "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": '{"query": "cats"}'}, "id": "call_1"}
        ],
    }
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"query": "cats"}
    assert messages[3] == {"role": "tool", "content": '{"items": ["cat"]}', "tool_call_id": "call_1", "name": "search"}
    assert messages[4]["role"] == "user"
    assert "Q2" in messages[4]["content"]
    assert "history" not in messages[4]["content"]
    assert "tools" not in messages[4]["content"]
    assert "tool_call_results" not in messages[4]["content"]
    assert "None" not in messages[4]["content"]
    assert "tools" in lm_kwargs


def test_chat_adapter_native_tool_history_replays_parallel_tool_results():

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str) -> str:
        return query

    NativeToolHistorySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "next_thought": output_field("next_thought", type_=Reasoning, desc="The next thought."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `turn_log`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    tool_calls = ToolCalls(
        tool_calls=[
            ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"}),
            ToolCalls.ToolCall(id="call_2", name="search", args={"query": "dogs"}),
        ]
    )
    tool_call_results = ToolCallResults.from_tool_calls_and_values(tool_calls, [{"items": ["cat"]}, {"items": ["dog"]}])
    history = TurnLog.model_validate(
        {
            "turns": [
                react_v2_turn(
                    pending_inputs={"question": "Q1"},
                    next_thought=Reasoning(content="I should search twice."),
                    tool_calls=tool_calls.model_copy(update={"tool_call_results": tool_call_results}),
                )
            ],
        }
    )
    messages, _lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=True),
        task_spec=NativeToolHistorySignature,
        demos=[],
        inputs={"question": "Q2", "turn_log": history, "tools": [Tool(search, description="Search for documents.")]},
        lm=FunctionCallingLM([{}]),
    )
    assert [tool_call["id"] for tool_call in messages[2]["tool_calls"]] == ["call_1", "call_2"]
    assert [(message["role"], message["tool_call_id"], message["content"]) for message in messages[3:5]] == [
        ("tool", "call_1", '{"items": ["cat"]}'),
        ("tool", "call_2", '{"items": ["dog"]}'),
    ]


def test_chat_adapter_native_tool_history_skips_empty_user_message():

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str) -> str:
        return query

    NativeToolHistorySignature = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "next_thought": output_field("next_thought", type_=Reasoning, desc="The next thought."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `turn_log`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = ToolCallResults.from_tool_calls_and_values([tool_call], ["cat result"])
    history = TurnLog.model_validate(
        {
            "turns": [react_v2_turn(tool_calls=ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results))],
        }
    )
    messages, _ = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=True),
        task_spec=NativeToolHistorySignature,
        demos=[],
        inputs={"turn_log": history, "tools": [Tool(search, description="Search for documents.")]},
        lm=FunctionCallingLM([{}]),
    )
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] is None
    assert messages[2]["role"] == "tool"


@pytest.mark.parametrize(
    ("tool_call_id", "tool_call_results"),
    [
        ("call_1", None),
        (
            "call_1",
            ToolCallResults(
                tool_call_results=[
                    ToolCallResults.ToolCallResult(call_id="other_call", name="search", value="cat result")
                ]
            ),
        ),
        (
            None,
            ToolCallResults(
                tool_call_results=[ToolCallResults.ToolCallResult(call_id=None, name="search", value="cat result")]
            ),
        ),
    ],
)
def test_chat_adapter_native_tool_history_skips_unmatched_tool_calls(tool_call_id, tool_call_results):

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str) -> str:
        return query

    NativeToolHistorySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "next_thought": output_field("next_thought", type_=Reasoning, desc="The next thought."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `turn_log`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    tool_call = ToolCalls.ToolCall(id=tool_call_id, name="search", args={"query": "cats"})
    tool_calls = ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results)
    history = TurnLog.model_validate(
        {
            "turns": [
                react_v2_turn(
                    pending_inputs={"question": "Q1"},
                    next_thought=Reasoning(content="I should search."),
                    tool_calls=tool_calls,
                )
            ],
        }
    )
    messages, _ = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=True),
        task_spec=NativeToolHistorySignature,
        demos=[],
        inputs={"question": "Q2", "turn_log": history, "tools": [Tool(search, description="Search for documents.")]},
        lm=FunctionCallingLM([{}]),
    )
    assert all("tool_calls" not in message for message in messages)
    assert all(message["role"] != "tool" for message in messages)
    assert messages[2]["role"] == "assistant"
    assert "I should search." in messages[2]["content"]


@pytest.mark.parametrize(
    "adapter", [ChatAdapter(use_native_function_calling=False), JSONAdapter(use_native_function_calling=False)]
)
def test_non_native_tool_history_remains_text_based(adapter):

    def search(query: str) -> str:
        return query

    ToolHistorySignature = make_task_spec(
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
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=ToolHistorySignature,
        demos=[],
        inputs={
            "question": "Q2",
            "turn_log": TurnLog.model_validate(
                {
                    "turns": [
                        react_v2_turn(
                            pending_inputs={"question": "Q1"},
                            next_thought="I should search.",
                            tool_calls=ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                        )
                    ],
                }
            ),
            "tools": [Tool(search, description="Search for documents.")],
        },
    )
    assert all(message["role"] != "tool" for message in messages)
    assert [message["role"] for message in messages[1:]] == ["user", "assistant", "user", "user"]
    assert "Q1" in messages[1]["content"]
    assert "tool_call_results" not in messages[1]["content"]
    assert "tool_calls" in messages[2]["content"]
    assert "[[ ## tool_call_results ## ]]" in messages[3]["content"]
    assert "cat" in messages[3]["content"]
    assert "Q2" not in messages[3]["content"]
    assert "Q2" in messages[4]["content"]
    assert "None" not in messages[3]["content"]


def test_chat_adapter_format_accepts_custom_history_formatter_returning_messages_only():
    from dspy.adapters.utils import build_lm_message

    class CustomHistoryAdapter(ChatAdapter):
        @override
        def format_conversation_history(self, task_spec, turn_log_field_name, inputs):
            del inputs[turn_log_field_name]
            return [build_lm_message("user", "custom history")]

    HistorySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, `turn_log`, produce the fields `answer`.",
    )
    messages = adapter_format_as_openai(
        adapter=CustomHistoryAdapter(),
        task_spec=HistorySignature,
        demos=[],
        inputs={
            "question": "Q2",
            "turn_log": TurnLog.model_validate(
                {
                    "turns": [react_v2_turn(pending_inputs={"question": "Q1"})],
                }
            ),
        },
    )
    assert messages[1] == {"role": "user", "content": "custom history"}
    assert messages[2]["role"] == "user"
    assert "Q2" in messages[2]["content"]


def test_chat_adapter_with_pydantic_models(make_run):

    class DogClass(pydantic.BaseModel):
        dog_breeds: list[str] = pydantic.Field(description="List of the breeds of dogs")
        num_dogs: int = pydantic.Field(description="Number of dogs the owner has", ge=0, le=10)

    class PetOwner(pydantic.BaseModel):
        name: str = pydantic.Field(description="Name of the owner")
        num_pets: int = pydantic.Field(description="Amount of pets the owner has", ge=0, le=100)
        dogs: DogClass = pydantic.Field(description="Nested Pydantic class with dog specific information ")

    class Answer(pydantic.BaseModel):
        result: str
        analysis: str

    TestSignature = make_task_spec(
        {
            "owner": input_field("owner", type_=PetOwner, desc="The owner."),
            "question": input_field("question", desc="The question."),
            "output": output_field("output", type_=Answer, desc="The output."),
        },
        instructions="Given the fields `owner`, `question`, produce the fields `output`.",
    )
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=TestSignature,
        demos=[],
        inputs={
            "owner": PetOwner(name="John", num_pets=5, dogs=DogClass(dog_breeds=["labrador", "chihuahua"], num_dogs=2)),
            "question": "How many non-dog pets does John have?",
        },
    )
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    assert "1. `owner` (PetOwner)" in system_content
    assert "2. `question` (str)" in system_content
    assert "1. `output` (Answer)" in system_content
    assert "name" in user_content
    assert "num_pets" in user_content
    assert "dogs" in user_content
    assert "dog_breeds" in user_content
    assert "num_dogs" in user_content
    assert "How many non-dog pets does John have?" in user_content


def test_chat_adapter_signature_information(make_run):
    TestSignature = make_task_spec(
        {
            "input1": input_field("input1", desc="String Input"),
            "input2": input_field("input2", type_=int, desc="Integer Input"),
            "output": output_field("output", desc="String Output"),
        },
        instructions="Given the fields `input1`, `input2`, produce the fields `output`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=ChatAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = default_model_response("[[ ## output ## ]]\nok\n\n[[ ## completed ## ]]")
        asyncio.run(program(input1="Test", input2=11, run=run))
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
        system_content = call_kwargs["messages"][0]["content"]
        user_content = call_kwargs["messages"][1]["content"]
        assert "1. `input1` (str)" in system_content
        assert "2. `input2` (int)" in system_content
        assert "1. `output` (str)" in system_content
        assert "[[ ## input1 ## ]]\n{input1}" in system_content
        assert "[[ ## input2 ## ]]\n{input2}" in system_content
        assert "[[ ## output ## ]]\n{output}" in system_content
        assert "[[ ## completed ## ]]" in system_content
        assert "[[ ## input1 ## ]]" in user_content
        assert "[[ ## input2 ## ]]" in user_content
        assert "[[ ## output ## ]]" in user_content
        assert "[[ ## completed ## ]]" in user_content


def test_chat_adapter_exception_raised_on_failure():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = ChatAdapter()
    invalid_completion = "{'output':'mismatched value'}"
    with pytest.raises(AdapterParseError, match=r"Adapter ChatAdapter failed to parse.*"):
        adapter.parse(task_spec=signature, completion=invalid_completion)


def test_chat_adapter_with_tool():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Answer question with the help of the tools",
    )

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    def get_population(country: str, year: int) -> str:
        return f"The population of {country} in {year} is 1000000"

    tools = [
        Tool(get_weather, description="Get the weather for a city"),
        Tool(get_population, description="Get the population for a country"),
    ]
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=MySignature,
        demos=[],
        inputs={"question": "What is the weather in Tokyo?", "tools": tools},
    )
    assert len(messages) == 2
    assert ToolCalls.description() in messages[0]["content"]
    assert "What is the weather in Tokyo?" in messages[1]["content"]
    assert "get_weather" in messages[1]["content"]
    assert "get_population" in messages[1]["content"]
    assert "{'city': {'type': 'string'}}" in messages[1]["content"]
    assert "{'country': {'type': 'string'}, 'year': {'type': 'integer'}}" in messages[1]["content"]


def test_chat_adapter_with_code():
    CodeAnalysis = make_task_spec(
        {
            "code": input_field("code", type_=Code, desc="The code."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Analyze the time complexity of the code",
    )
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=CodeAnalysis, demos=[], inputs={"code": "print('Hello, world!')"}
    )
    assert len(messages) == 2
    assert Code.description() in messages[0]["content"]
    assert "print('Hello, world!')" in messages[1]["content"]
    CodeGeneration = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "code": output_field("code", type_=Code, desc="The code."),
        },
        instructions="Generate code to answer the question",
    )
    adapter = ChatAdapter()
    lm = LM(model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='[[ ## code ## ]]\nprint("Hello, world!")'))],
            model="openai/gpt-4o-mini",
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=CodeGeneration,
                demos=[],
                inputs={"question": "Write a python program to print 'Hello, world!'"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["code"].code == 'print("Hello, world!")'


def test_code_output_field_omits_json_schema_in_prompt():
    CodeGeneration = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "code": output_field("code", type_=Code, desc="The code."),
        },
        instructions="Generate code to answer the question",
    )
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=CodeGeneration, demos=[], inputs={"question": "Hello"}
    )
    system_content = messages[0]["content"]
    assert Code.description() in system_content
    assert "JSON schema" not in system_content
    assert '"properties"' not in system_content
    assert "Code type in DSPy" not in system_content


def test_citations_output_field_keeps_json_schema_in_prompt():
    CitationGeneration = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "citations": output_field("citations", type_=Citations, desc="The citations."),
        },
        instructions="Given the fields `question`, produce the fields `citations`.",
    )
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=CitationGeneration, demos=[], inputs={"question": "Hello"}
    )
    system_content = messages[0]["content"]
    assert "must adhere to the JSON schema" in system_content
    assert "Type description of Citations" in system_content


def test_chat_adapter_formats_conversation_history():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, `turn_log`, produce the fields `answer`.",
    )
    history = TurnLog.model_validate(
        {
            "turns": [
                task_io_turn(question="What is the capital of France?", answer="Paris"),
                task_io_turn(question="What is the capital of Germany?", answer="Berlin"),
            ],
        }
    )
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=MySignature,
        demos=[],
        inputs={"question": "What is the capital of France?", "turn_log": history},
    )
    assert len(messages) == 6
    assert messages[1]["content"] == "[[ ## question ## ]]\nWhat is the capital of France?"
    assert messages[2]["content"] == "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]\n"
    assert messages[3]["content"] == "[[ ## question ## ]]\nWhat is the capital of Germany?"
    assert messages[4]["content"] == "[[ ## answer ## ]]\nBerlin\n\n[[ ## completed ## ]]\n"


def test_chat_adapter_toolcalls_native_function_calling():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`, `tool_calls`.",
    )

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [Tool(get_weather, description="Get the weather for a city")]
    adapter = JSONAdapter(use_native_function_calling=True)
    lm = LM(model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                function=Function(arguments='{"city":"Paris"}', name="get_weather"),
                                id="call_pQm8ajtSMxgA0nrzK2ivFmxG",
                                type="function",
                            )
                        ],
                    ),
                )
            ],
            model="openai/gpt-4o-mini",
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Paris?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["tool_calls"] == ToolCalls(
            tool_calls=[
                ToolCalls.ToolCall(id="call_pQm8ajtSMxgA0nrzK2ivFmxG", name="get_weather", args={"city": "Paris"})
            ]
        )
        assert result[0]["answer"] is None
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))], model="openai/gpt-4o-mini"
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Paris?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["answer"] == "Paris"
        assert result[0]["tool_calls"] is None


def test_chat_adapter_toolcalls_vague_match():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [Tool(get_weather, description="Get the weather for a city")]
    adapter = ChatAdapter()
    lm = LM(model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content='[[ ## tool_calls ## ]]\n{"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]}'
                    )
                )
            ],
            model="openai/gpt-4o-mini",
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Paris?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["tool_calls"] == ToolCalls(
            tool_calls=[ToolCalls.ToolCall(name="get_weather", args={"city": "Paris"})]
        )
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content='[[ ## tool_calls ## ]]\n{"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]}'
                    )
                )
            ],
            model="openai/gpt-4o-mini",
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Paris?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["tool_calls"] == ToolCalls(
            tool_calls=[ToolCalls.ToolCall(name="get_weather", args={"city": "Paris"})]
        )


def test_chat_adapter_native_reasoning():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `answer`.",
    )
    adapter = ChatAdapter()
    from dspy.core.types import LMProviderOptions

    lm = LM(
        model="anthropic/claude-3-7-sonnet-20250219",
        provider_options=LMProviderOptions(extensions={"reasoning": {"effort": "low"}}),
    )
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
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
        modified_signature, _, _ = adapter._call_preprocess(
            lm,
            {},
            MySignature,
            {"question": "What is the capital of France?"},
        )
        assert "reasoning" not in modified_signature.output_fields
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the capital of France?"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["reasoning"] == Reasoning(content="Step-by-step thinking about the capital of France")


def test_chat_adapter_parses_float_with_underscores(make_run):

    class Score(pydantic.BaseModel):
        score: float

    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "score": output_field("score", type_=Score, desc="The score."),
        },
        instructions="Given the fields `question`, produce the fields `score`.",
    )
    adapter = ChatAdapter()
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content='[[ ## score ## ]]\n{"score": 123456.789}\n[[ ## completed ## ]]'))
            ],
            model="openai/gpt-4o-mini",
        )
        lm = LM("openai/gpt-4o-mini")
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the score?"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["score"].score == 123456.789


def test_format_system_message(make_run):
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answers": output_field("answers", type_=list[str], desc="The answers."),
            "scores": output_field("scores", type_=list[float], desc="The scores."),
        },
        instructions="Answer the question with multiple answers and scores",
    )
    adapter = ChatAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answers` (list[str]): The answers.\n2. `scores` (list[float]): The scores.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answers ## ]]\n{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}\n\n[[ ## scores ## ]]\n{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer the question with multiple answers and scores'
    assert system_message == expected_system_message


def test_null_content_raises_adapter_parse_error(make_run):
    from dspy.errors import AdapterParseError

    lm = LM("openai/gpt-4o-mini")
    response = ModelResponse(choices=[Choices(message=Message(content=None))], model="openai/gpt-4o-mini")
    run = make_run(lm=lm)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock, return_value=response):
        cot = ChainOfThought(ts("question -> answer"))
        with pytest.raises(AdapterParseError):
            asyncio.run(cot(question="test", run=run))


def test_empty_string_content_raises_adapter_parse_error(make_run):
    from dspy.errors import AdapterParseError

    lm = LM("openai/gpt-4o-mini")
    response = ModelResponse(choices=[Choices(message=Message(content=""))], model="openai/gpt-4o-mini")
    run = make_run(lm=lm)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock, return_value=response):
        cot = ChainOfThought(ts("question -> answer"))
        with pytest.raises(AdapterParseError):
            asyncio.run(cot(question="test", run=run))


def test_tool_call_with_null_content_does_not_raise():
    adapter = ChatAdapter(use_native_function_calling=True)
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`, `tool_calls`.",
    )
    outputs = [
        {
            "text": None,
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}, "id": "call_1", "type": "function"}
            ],
        }
    ]
    result = adapter._call_postprocess(
        processed_task_spec=task_spec,
        original_task_spec=task_spec,
        response=outputs_to_lm_response(outputs),
    )
    assert result is not None
    assert len(result) == 1
    assert result[0]["tool_calls"].tool_calls[0].id == "call_1"


def test_tool_call_with_unstructured_content_does_not_raise():
    adapter = ChatAdapter(use_native_function_calling=True)
    original_task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "next_thought": output_field("next_thought", type_=Reasoning, desc="The next thought."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    processed_task_spec = original_task_spec.delete("tools").delete("tool_calls").delete("next_thought")
    outputs = [
        {
            "text": "I'll search for that now.",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}, "id": "call_1", "type": "function"}
            ],
            "reasoning_content": "I need a search result.",
        }
    ]
    result = adapter._call_postprocess(
        processed_task_spec=processed_task_spec,
        original_task_spec=original_task_spec,
        response=outputs_to_lm_response(outputs),
    )
    assert result[0]["tool_calls"].tool_calls[0].id == "call_1"
    assert result[0]["next_thought"] == Reasoning(content="I need a search result.")


def test_tool_call_with_structured_content_preserves_other_outputs():
    adapter = ChatAdapter(use_native_function_calling=True)
    original_task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`, `tool_calls`.",
    )
    processed_task_spec = original_task_spec.delete("tools").delete("tool_calls")
    outputs = [
        {
            "text": "[[ ## answer ## ]]\nI should use a tool.\n\n[[ ## completed ## ]]",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}, "id": "call_1", "type": "function"}
            ],
        }
    ]
    result = adapter._call_postprocess(
        processed_task_spec=processed_task_spec,
        original_task_spec=original_task_spec,
        response=outputs_to_lm_response(outputs),
    )
    assert result[0]["answer"] == "I should use a tool."
    assert result[0]["tool_calls"].tool_calls[0].id == "call_1"


def test_native_fc_raises_when_lm_does_not_support_function_calling():
    class NoFunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return False

    def search(query: str) -> str:
        return query

    adapter = ChatAdapter(use_native_function_calling=True)
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    with pytest.raises(ValueError, match="does not support function calling"):
        adapter._call_preprocess(
            lm=NoFunctionCallingLM([{}]),
            config={},
            task_spec=task_spec,
            inputs={"question": "test", "tools": [Tool(search, description="Search.")]},
        )


def test_tool_calls_with_malformed_text_raises_parse_error():
    adapter = ChatAdapter(use_native_function_calling=True)
    original_task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`, `tool_calls`.",
    )
    processed_task_spec = original_task_spec.delete("tools").delete("tool_calls")
    outputs = [
        {
            "text": "this is not valid structured output",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}, "id": "call_1", "type": "function"}
            ],
        }
    ]
    with pytest.raises(AdapterParseError):
        adapter._call_postprocess(
            processed_task_spec=processed_task_spec,
            original_task_spec=original_task_spec,
            response=outputs_to_lm_response(outputs),
        )


def test_tool_calls_without_text_output_fields_skips_text_parse():
    adapter = ChatAdapter(use_native_function_calling=True)
    original_task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    processed_task_spec = original_task_spec.delete("tools").delete("tool_calls")
    outputs = [
        {
            "text": "unstructured completion text",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}, "id": "call_1", "type": "function"}
            ],
        }
    ]
    result = adapter._call_postprocess(
        processed_task_spec=processed_task_spec,
        original_task_spec=original_task_spec,
        response=outputs_to_lm_response(outputs),
    )
    assert result[0]["tool_calls"].tool_calls[0].id == "call_1"
    assert "answer" not in result[0] or result[0].get("answer") is None


def test_provider_tool_calls_preserve_id_and_repair_arguments():
    adapter = ChatAdapter(use_native_function_calling=True)
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    outputs = [
        {
            "text": None,
            "tool_calls": [
                {
                    "function": {"name": "search", "arguments": '{"query": "cats",}'},
                    "call_id": "call_from_responses",
                    "type": "function",
                }
            ],
        }
    ]
    result = adapter._call_postprocess(
        processed_task_spec=task_spec,
        original_task_spec=task_spec,
        response=outputs_to_lm_response(outputs),
    )
    assert result[0]["tool_calls"] == ToolCalls(
        tool_calls=[ToolCalls.ToolCall(id="call_from_responses", name="search", args={"query": "cats"})]
    )


def test_native_response_type_without_parse_lm_output_raises():

    class OpaqueType(FieldTypeMixin):
        label: str

        @override
        def format(self) -> str:
            return self.label

    OpaqueSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", type_=OpaqueType, desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    adapter = ChatAdapter(native_response_types=[OpaqueType])
    lm = DummyLM([{}])
    with pytest.raises(TypeError, match="parse_lm_output"):
        asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=OpaqueSignature,
                demos=[],
                inputs={"question": "test"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )


def test_chat_adapter_parse_rejects_nonempty_preamble():
    from dspy.errors import AdapterParseError
    from dspy.task_spec import input_field, make_task_spec, output_field

    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="q"),
            "answer": output_field("answer", desc="a"),
        },
        instructions="answer",
    )
    adapter = ChatAdapter()
    completion = "intro text\n[[ ## answer ## ]]\nParis"
    with pytest.raises(AdapterParseError, match="preamble"):
        adapter.parse(task_spec=task_spec, completion=completion)


def test_chat_adapter_parse_hyphenated_field_name():
    from dspy.task_spec import input_field, make_task_spec, output_field

    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="q"),
            "my-answer": output_field("my-answer", desc="a"),
        },
        instructions="answer",
    )
    adapter = ChatAdapter()
    completion = "[[ ## my-answer ## ]]\nParis"
    parsed = adapter.parse(task_spec=task_spec, completion=completion)
    assert parsed == {"my-answer": "Paris"}
