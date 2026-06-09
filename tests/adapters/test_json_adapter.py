import asyncio
import enum
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest import mock

import pydantic
import pytest
from typing_extensions import override

from dspy.adapters.types.document import Document
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from litellm.utils import ModelResponse
try:
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
    from litellm.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from openai.types.responses import ResponseOutputMessage

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.code import Code
from dspy.adapters.types.image import Image
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.clients.lm import LM
from dspy.history import TurnLog
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.task_spec import FieldSpec, make_task_spec
from dspy.utils.exceptions import LMUnexpectedError
from tests.adapters.conftest import adapter_format_as_openai, format_messages_and_lm_kwargs, make_adapter_run
from tests.task_spec.helpers import ts


def _structured_output_model_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                message=Message(
                    content='{"output1":"x","output2":true,"output3":{"subfield1":1,"subfield2":1.0},"output4_unannotated":"y"}'
                )
            )
        ],
        model="openai/gpt-4o",
    )


def test_json_adapter_format_exact_messages_for_simple_signature():
    StringSignature = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(),
        task_spec=StringSignature,
        demos=[],
        inputs={"question": "What is the capital of France?"},
    )
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs
    assert messages == [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.',
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nWhat is the capital of France?\n\nRespond with a JSON object in the following order of fields: `answer`.",
        },
    ]


def test_json_adapter_format_exact_messages_with_demo_and_typed_output():
    MultiAnswer = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "answer": FieldSpec.output("answer"),
            "confidence": FieldSpec.output("confidence", type_=float),
        },
        instructions="Given the fields `question`, produce the fields `answer`, `confidence`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(),
        task_spec=MultiAnswer,
        demos=[{"question": "Q1", "answer": "A1", "confidence": 0.9}],
        inputs={"question": "Q2"},
    )
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs
    assert messages == [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str): \n2. `confidence` (float):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}",\n  "confidence": "{confidence}        # note: the value you produce must be a single float value"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`, `confidence`.',
        },
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {"role": "assistant", "content": '{\n  "answer": "A1",\n  "confidence": 0.9\n}'},
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ2\n\nRespond with a JSON object in the following order of fields: `answer`, then `confidence` (must be formatted as a valid Python float).",
        },
    ]


def test_json_adapter_format_exact_messages_with_described_and_bool_outputs():
    TestSignature = make_task_spec(
        {
            "input1": FieldSpec.input("input1"),
            "output1": FieldSpec.output("output1", desc="String output field"),
            "output2": FieldSpec.output("output2", type_=bool),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(), task_spec=TestSignature, demos=[], inputs={"input1": "Test input"}
    )
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs
    assert messages == [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `input1` (str):\nYour output fields are:\n1. `output1` (str): String output field\n2. `output2` (bool):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## input1 ## ]]\n{input1}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "output1": "{output1}",\n  "output2": "{output2}        # note: the value you produce must be True or False"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `input1`, produce the fields `output1`, `output2`.',
        },
        {
            "role": "user",
            "content": "[[ ## input1 ## ]]\nTest input\n\nRespond with a JSON object in the following order of fields: `output1`, then `output2` (must be formatted as a valid Python bool).",
        },
    ]


def test_json_adapter_format_exact_messages_with_history_demo_pydantic_tools_and_image():

    def search(query: str, k: int = 3) -> str:
        return query

    class Location(pydantic.BaseModel):
        city: str
        country: str

    class Profile(pydantic.BaseModel):
        name: str
        location: Location
        interests: list[str]

    class AnswerCard(pydantic.BaseModel):
        answer: str
        sources: list[str]

    RichRenderingSignature = make_task_spec(
        {
            "history": FieldSpec.input("history", type_=TurnLog),
            "image": FieldSpec.input("image", type_=Image),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "profile": FieldSpec.input("profile", type_=Profile),
            "question": FieldSpec.input("question"),
            "answer": FieldSpec.output("answer", type_=AnswerCard),
        },
        instructions="Answer using all supplied context.",
    )
    tool = Tool(search, description="Search for documents.")
    demo_profile = Profile(name="Ada", location=Location(city="London", country="UK"), interests=["math", "machines"])
    current_profile = Profile(
        name="Grace", location=Location(city="Arlington", country="USA"), interests=["compilers", "navy"]
    )
    history = TurnLog(
        turns=(
            {
                "profile": demo_profile,
                "question": "Who is Ada?",
                "answer": AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
            },
        )
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(),
        task_spec=RichRenderingSignature,
        demos=[
            {
                "image": Image("https://example.com/demo.png"),
                "tools": [tool],
                "profile": demo_profile,
                "question": "What should we mention?",
                "answer": AnswerCard(answer="Mention analytical engines.", sources=["demo"]),
            }
        ],
        inputs={
            "history": history,
            "image": Image("https://example.com/current.png"),
            "tools": [tool],
            "profile": current_profile,
            "question": "What should the answer include?",
        },
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `history` (TurnLog): \n2. `image` (Image): \n3. `tools` (list[Tool]): \n4. `profile` (Profile): \n5. `question` (str):\nYour output fields are:\n1. `answer` (AnswerCard):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## history ## ]]\n{history}\n\n[[ ## image ## ]]\n{image}\n\n[[ ## tools ## ]]\n{tools}\n\n[[ ## profile ## ]]\n{profile}\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"answer\\": {\\"type\\": \\"string\\", \\"title\\": \\"Answer\\"}, \\"sources\\": {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"string\\"}, \\"title\\": \\"Sources\\"}}, \\"required\\": [\\"answer\\", \\"sources\\"], \\"title\\": \\"AnswerCard\\"}"\n}\nIn adhering to this structure, your objective is: \n        Answer using all supplied context.',
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is an example of the task, though some input or output fields are not supplied.",
                },
                {"type": "text", "text": "[[ ## image ## ]]\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                {
                    "type": "text",
                    "text": "\n\n[[ ## tools ## ]]\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]",
                },
                {
                    "type": "text",
                    "text": '\n\n[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}',
                },
                {"type": "text", "text": "\n\n[[ ## question ## ]]\nWhat should we mention?"},
            ],
        },
        {
            "role": "assistant",
            "content": '{\n  "answer": {\n    "answer": "Mention analytical engines.",\n    "sources": [\n      "demo"\n    ]\n  }\n}',
        },
        {
            "role": "user",
            "content": '[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}\n\n[[ ## question ## ]]\nWho is Ada?',
        },
        {
            "role": "assistant",
            "content": '{\n  "answer": {\n    "answer": "Ada is a mathematician.",\n    "sources": [\n      "memory"\n    ]\n  }\n}',
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[[ ## image ## ]]\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                {
                    "type": "text",
                    "text": "\n\n[[ ## tools ## ]]\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]",
                },
                {
                    "type": "text",
                    "text": '\n\n[[ ## profile ## ]]\n{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, "interests": ["compilers", "navy"]}',
                },
                {"type": "text", "text": "\n\n[[ ## question ## ]]\nWhat should the answer include?"},
                {
                    "type": "text",
                    "text": "\n\nRespond with a JSON object in the following order of fields: `answer` (must be formatted as a valid Python AnswerCard).",
                },
            ],
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_int_and_mapping_outputs():
    IntDictSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "count": FieldSpec.output("count", type_=int),
            "metadata": FieldSpec.output("metadata", type_=dict[str, int]),
        },
        instructions="Given the fields `question`, produce the fields `count`, `metadata`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(), task_spec=IntDictSignature, demos=[], inputs={"question": "Count things"}
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `count` (int): \n2. `metadata` (dict[str, int]):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "count": "{count}        # note: the value you produce must be a single int value",\n  "metadata": "{metadata}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"additionalProperties\\": {\\"type\\": \\"integer\\"}}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `count`, `metadata`.',
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nCount things\n\nRespond with a JSON object in the following order of fields: `count` (must be formatted as a valid Python int), then `metadata` (must be formatted as a valid Python dict[str, int]).",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_literal_and_enum_outputs():

    class Label(enum.Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"

    LiteralEnumSignature = make_task_spec(
        {
            "text": FieldSpec.input("text"),
            "decision": FieldSpec.output("decision", type_=Literal["accept", "reject"]),
            "label": FieldSpec.output("label", type_=Label),
        },
        instructions="Given the fields `text`, produce the fields `decision`, `label`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(), task_spec=LiteralEnumSignature, demos=[], inputs={"text": "Looks good"}
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `text` (str):\nYour output fields are:\n1. `decision` (Literal[\'accept\', \'reject\']): \n2. `label` (Label):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## text ## ]]\n{text}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "decision": "{decision}        # note: the value you produce must exactly match (no extra characters) one of: accept; reject",\n  "label": "{label}        # note: the value you produce must be one of: positive; negative"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `text`, produce the fields `decision`, `label`.',
        },
        {
            "role": "user",
            "content": "[[ ## text ## ]]\nLooks good\n\nRespond with a JSON object in the following order of fields: `decision` (must be formatted as a valid Python Literal['accept', 'reject']), then `label` (must be formatted as a valid Python Label).",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_nested_pydantic_output():

    class JsonNestedAddress(pydantic.BaseModel):
        city: str
        country: str

    class JsonNestedSummary(pydantic.BaseModel):
        title: str
        address: JsonNestedAddress
        scores: list[float]

    PydanticSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "summary": FieldSpec.output("summary", type_=JsonNestedSummary)},
        instructions="Given the fields `question`, produce the fields `summary`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(), task_spec=PydanticSignature, demos=[], inputs={"question": "Summarize"}
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `summary` (JsonNestedSummary):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "summary": "{summary}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"JsonNestedAddress\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"city\\": {\\"type\\": \\"string\\", \\"title\\": \\"City\\"}, \\"country\\": {\\"type\\": \\"string\\", \\"title\\": \\"Country\\"}}, \\"required\\": [\\"city\\", \\"country\\"], \\"title\\": \\"JsonNestedAddress\\"}}, \\"properties\\": {\\"address\\": {\\"$ref\\": \\"#/$defs/JsonNestedAddress\\"}, \\"scores\\": {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"number\\"}, \\"title\\": \\"Scores\\"}, \\"title\\": {\\"type\\": \\"string\\", \\"title\\": \\"Title\\"}}, \\"required\\": [\\"title\\", \\"address\\", \\"scores\\"], \\"title\\": \\"JsonNestedSummary\\"}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `summary`.',
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nSummarize\n\nRespond with a JSON object in the following order of fields: `summary` (must be formatted as a valid Python JsonNestedSummary).",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_incomplete_demo():
    IncompleteDemoSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "context": FieldSpec.input("context"),
            "answer": FieldSpec.output("answer"),
            "score": FieldSpec.output("score", type_=float),
        },
        instructions="Given the fields `question`, `context`, produce the fields `answer`, `score`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(),
        task_spec=IncompleteDemoSignature,
        demos=[{"question": "Q1", "answer": "A1"}],
        inputs={"question": "Q2", "context": "C2"},
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str): \n2. `context` (str):\nYour output fields are:\n1. `answer` (str): \n2. `score` (float):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\n[[ ## context ## ]]\n{context}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}",\n  "score": "{score}        # note: the value you produce must be a single float value"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `context`, produce the fields `answer`, `score`.',
        },
        {
            "role": "user",
            "content": "This is an example of the task, though some input or output fields are not supplied.\n\n[[ ## question ## ]]\nQ1",
        },
        {
            "role": "assistant",
            "content": '{\n  "answer": "A1",\n  "score": "Not supplied for this particular example. "\n}',
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ2\n\n[[ ## context ## ]]\nC2\n\nRespond with a JSON object in the following order of fields: `answer`, then `score` (must be formatted as a valid Python float).",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_and_lm_kwargs_with_native_tool_calling():

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str, k: int = 3) -> str:
        return query

    NativeToolSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "tool_calls": FieldSpec.output("tool_calls", type_=ToolCalls),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(use_native_function_calling=True),
        task_spec=NativeToolSignature,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
        lm=FunctionCallingLM([{}]),
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str):\nYour output fields are:\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `tools`, produce the fields `tool_calls`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ?\n\nRespond with a JSON object in the following order of fields: .",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 3}},
                        "required": ["query", "k"],
                    },
                },
            }
        ]
    }
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_tool_calls_output_demo():
    ToolCallsSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "tool_calls": FieldSpec.output("tool_calls", type_=ToolCalls)},
        instructions="Given the fields `question`, produce the fields `tool_calls`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(use_native_function_calling=False),
        task_spec=ToolCallsSignature,
        demos=[
            {"question": "Q1", "tool_calls": ToolCalls.from_dict_list([{"name": "search", "args": {"query": "cats"}}])}
        ],
        inputs={"question": "Q2"},
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `tool_calls` (ToolCalls): \n    Type description of ToolCalls: Tool calls must be a JSON object with `tool_calls`, a list of calls. Each call must include `name` and `args`. Example: {"tool_calls": [{"name": "search", "args": {"query": "cats"}}]}\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "tool_calls": "{tool_calls}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"ToolCall\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"args\\": {\\"type\\": \\"object\\", \\"additionalProperties\\": true, \\"title\\": \\"Args\\"}, \\"name\\": {\\"type\\": \\"string\\", \\"title\\": \\"Name\\"}}, \\"required\\": [\\"name\\", \\"args\\"], \\"title\\": \\"ToolCall\\"}}, \\"properties\\": {\\"tool_calls\\": {\\"type\\": \\"array\\", \\"items\\": {\\"$ref\\": \\"#/$defs/ToolCall\\"}, \\"title\\": \\"Tool Calls\\"}}, \\"required\\": [\\"tool_calls\\"], \\"title\\": \\"ToolCalls\\"}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `tool_calls`.',
        },
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {
            "role": "assistant",
            "content": '{\n  "tool_calls": {\n    "tool_calls": [\n      {\n        "name": "search",\n        "args": {\n          "query": "cats"\n        }\n      }\n    ]\n  }\n}',
        },
        {
            "role": "user",
            "content": '[[ ## question ## ]]\nQ2\n\nRespond with a JSON object in the following order of fields: `tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).',
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_non_native_tool_result_history_field():

    def search(query: str) -> str:
        return query

    ToolHistorySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "history": FieldSpec.input("history", type_=TurnLog),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "next_thought": FieldSpec.output("next_thought"),
            "tool_calls": FieldSpec.output("tool_calls", type_=ToolCalls),
        },
        instructions="Given the fields `question`, `history`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = ToolCallResults.from_tool_calls_and_values([tool_call], ["cat"])
    messages, _lm_kwargs = format_messages_and_lm_kwargs(
        adapter=JSONAdapter(use_native_function_calling=False),
        task_spec=ToolHistorySignature,
        demos=[],
        inputs={
            "question": "Q2",
            "history": TurnLog(
                turns=(
                    {
                        "question": "Q1",
                        "next_thought": "I should search.",
                        "tool_calls": ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                    },
                )
            ),
            "tools": [Tool(search, description="Search for documents.")],
        },
    )
    assert (
        messages[3]["content"]
        == '[[ ## tool_call_results ## ]]\n{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}'
    )
    assert (
        messages[4]["content"]
        == '[[ ## question ## ]]\nQ2\n\n[[ ## tools ## ]]\n["search, whose description is <desc>Search for documents.</desc>. It takes arguments {\'query\': {\'type\': \'string\'}}."]\n\nRespond with a JSON object in the following order of fields: `next_thought`, then `tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).'
    )


def test_json_adapter_passes_structured_output_when_supported_by_model(make_run):

    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1", ge=0, le=10)
        subfield2: float = pydantic.Field(description="Float subfield 2")

    TestSignature = make_task_spec(
        {
            "input1": FieldSpec.input("input1"),
            "output1": FieldSpec.output("output1"),
            "output2": FieldSpec.output("output2", type_=bool, desc="Boolean output field"),
            "output3": FieldSpec.output("output3", type_=OutputField3, desc="Nested output field"),
            "output4_unannotated": FieldSpec.output("output4_unannotated", desc="Unannotated output field"),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`, `output3`, `output4_unannotated`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = _structured_output_model_response()
        asyncio.run(program(input1="Test input", run=run))

    def clean_schema_extra(field_name, field_info):
        attrs = dict(field_info.__repr_args__())
        if "json_schema_extra" in attrs:
            attrs["json_schema_extra"] = {
                k: v
                for k, v in attrs["json_schema_extra"].items()
                if k != "__dspy_field_type" and (not (k == "desc" and v == f"${{{field_name}}}"))
            }
        return attrs

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    response_format = call_kwargs.get("response_format")
    assert response_format is not None
    assert issubclass(response_format, pydantic.BaseModel)
    assert response_format.model_fields.keys() == {"output1", "output2", "output3", "output4_unannotated"}


def test_json_adapter_not_using_structured_outputs_when_not_supported_by_model(make_run):
    TestSignature = make_task_spec(
        {
            "input1": FieldSpec.input("input1"),
            "output1": FieldSpec.output("output1"),
            "output2": FieldSpec.output("output2", type_=bool),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="fakeprovider/fakemodel"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'output1': 'Test output', 'output2': True}"))],
            model="openai/gpt-4o",
        )
        asyncio.run(program(input1="Test input", run=run))
    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert "response_format" not in call_kwargs


def test_json_adapter_with_structured_outputs_does_not_mutate_original_signature(make_run):

    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1")
        subfield2: float = pydantic.Field(description="Float subfield 2")

    TestSignature = make_task_spec(
        {
            "input1": FieldSpec.input("input1"),
            "output1": FieldSpec.output("output1"),
            "output2": FieldSpec.output("output2", type_=bool, desc="Boolean output field"),
            "output3": FieldSpec.output("output3", type_=OutputField3, desc="Nested output field"),
            "output4_unannotated": FieldSpec.output("output4_unannotated", desc="Unannotated output field"),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`, `output3`, `output4_unannotated`.",
    )
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = _structured_output_model_response()
        asyncio.run(program(input1="Test input", run=run))
    assert program.task_spec.equals(TestSignature)


def test_json_adapter_sync_call():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=JSONAdapter())
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
async def test_json_adapter_async_call():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=JSONAdapter())
    result = await adapter(
        lm=lm,
        config={},
        task_spec=signature,
        demos=[],
        inputs={"question": "What is the capital of France?"},
        run=make_adapter_run(lm=lm, adapter=adapter),
    )
    assert result == [{"answer": "Paris"}]


def test_json_adapter_on_pydantic_model(make_run):
    from litellm.utils import Choices, Message, ModelResponse

    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    TestSignature = make_task_spec(
        {
            "user": FieldSpec.input("user", type_=User, desc="The user who asks the question"),
            "question": FieldSpec.input("question", desc="Question the user asks"),
            "answer": FieldSpec.output("answer", type_=Answer, desc="Answer to this question"),
        },
        instructions="Given the fields `user`, `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}"
                    )
                )
            ],
            model="openai/gpt-4o",
        )
        result = asyncio.run(
            program(
                user=User(id=5, name="name_test", email="email_test"),
                question="What is the capital of France?",
                run=run,
            )
        )
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content
        expected_output_structure = 'Outputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": \\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": [\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        assert expected_output_structure in content
        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None
        expected_input_data = '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\nWhat is the capital of France?\n\n'
        assert expected_input_data in user_message_content
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_parse_raise_error_on_mismatch_fields():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = JSONAdapter()
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer1': 'Paris'}"))], model="openai/gpt-4o"
        )
        lm = LM(model="openai/gpt-4o-mini")
        with pytest.raises(AdapterParseError) as e:
            asyncio.run(
                adapter(
                    lm=lm,
                    config={},
                    task_spec=signature,
                    demos=[],
                    inputs={"question": "What is the capital of France?"},
                    run=make_adapter_run(lm=lm, adapter=adapter),
                )
            )
    assert e.value.adapter_name == "JSONAdapter"
    assert e.value.task_spec == signature
    assert e.value.lm_response == "{'answer1': 'Paris'}"
    assert e.value.parsed_result == {}
    assert (
        str(e.value)
        == "Adapter JSONAdapter failed to parse the LM response. \n\nLM Response: {'answer1': 'Paris'} \n\nExpected to find output fields in the LM response: [answer] \n\nActual output fields parsed from the LM response: [] \n\n"
    )


def test_json_adapter_formats_image():
    image = Image(url="https://example.com/image.jpg")
    MySignature = make_task_spec(
        {"image": FieldSpec.input("image", type_=Image), "text": FieldSpec.output("text")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(adapter=adapter, task_spec=MySignature, demos=[], inputs={"image": image})
    assert len(messages) == 2
    user_message_content = messages[1]["content"]
    assert user_message_content is not None
    assert len(user_message_content) == 3
    assert user_message_content[0]["type"] == "text"
    assert user_message_content[2]["type"] == "text"
    expected_image_content = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    assert expected_image_content in user_message_content


def test_json_adapter_formats_image_with_few_shot_examples():
    MySignature = make_task_spec(
        {"image": FieldSpec.input("image", type_=Image), "text": FieldSpec.output("text")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    adapter = JSONAdapter()
    demos = [
        Example.from_record({"image": Image(url="https://example.com/image1.jpg"), "text": "This is a test image"}),
        Example.from_record(
            {"image": Image(url="https://example.com/image2.jpg"), "text": "This is another test image"}
        ),
    ]
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=MySignature,
        demos=demos,
        inputs={"image": Image(url="https://example.com/image3.jpg")},
    )
    assert len(messages) == 6
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in messages[1]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in messages[3]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}} in messages[5]["content"]


def test_json_adapter_formats_image_with_nested_images():

    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    MySignature = make_task_spec(
        {"image": FieldSpec.input("image", type_=ImageWrapper), "text": FieldSpec.output("text")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    image1 = Image(url="https://example.com/image1.jpg")
    image2 = Image(url="https://example.com/image2.jpg")
    image3 = Image(url="https://example.com/image3.jpg")
    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=MySignature, demos=[], inputs={"image": image_wrapper}
    )
    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]


def test_json_adapter_formats_with_nested_documents():

    class DocumentWrapper(pydantic.BaseModel):
        documents: list[Document]

    MySignature = make_task_spec(
        {"document": FieldSpec.input("document", type_=DocumentWrapper), "text": FieldSpec.output("text")},
        instructions="Given the fields `document`, produce the fields `text`.",
    )
    doc1 = Document(data="Hello, world!")
    doc2 = Document(data="Hello, world 2!")
    document_wrapper = DocumentWrapper(documents=[doc1, doc2])
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=MySignature, demos=[], inputs={"document": document_wrapper}
    )
    expected_doc1_content = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "Hello, world!"},
        "citations": {"enabled": True},
    }
    expected_doc2_content = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "Hello, world 2!"},
        "citations": {"enabled": True},
    }
    assert expected_doc1_content in messages[1]["content"]
    assert expected_doc2_content in messages[1]["content"]


def test_json_adapter_formats_image_with_few_shot_examples_with_nested_images():

    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    MySignature = make_task_spec(
        {"image": FieldSpec.input("image", type_=ImageWrapper), "text": FieldSpec.output("text")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    image1 = Image(url="https://example.com/image1.jpg")
    image2 = Image(url="https://example.com/image2.jpg")
    image3 = Image(url="https://example.com/image3.jpg")
    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    demos = [Example.from_record({"image": image_wrapper, "text": "This is a test image"})]
    image_wrapper_2 = ImageWrapper(images=[Image(url="https://example.com/image4.jpg")], tag=["test", "example"])
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=MySignature, demos=demos, inputs={"image": image_wrapper_2}
    )
    assert len(messages) == 4
    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image4.jpg"}} in messages[-1]["content"]


def test_json_adapter_with_tool():
    MySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "answer": FieldSpec.output("answer"),
            "tool_calls": FieldSpec.output("tool_calls", type_=ToolCalls),
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
    adapter = JSONAdapter()
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
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"answer":"sunny","tool_calls":{"tool_calls":[]}}'))],
            model="openai/gpt-4o-mini",
        )
        lm = LM(model="openai/gpt-4o-mini")
        asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Tokyo?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert len(call_kwargs["tools"]) > 0
    assert call_kwargs["tools"][0] == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
    assert call_kwargs["tools"][1] == {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population for a country",
            "parameters": {
                "type": "object",
                "properties": {"country": {"type": "string"}, "year": {"type": "integer"}},
                "required": ["country", "year"],
            },
        },
    }


def test_json_adapter_with_code():
    CodeAnalysis = make_task_spec(
        {"code": FieldSpec.input("code", type_=Code), "result": FieldSpec.output("result")},
        instructions="Analyze the time complexity of the code",
    )
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=CodeAnalysis, demos=[], inputs={"code": "print('Hello, world!')"}
    )
    assert len(messages) == 2
    assert Code.description() in messages[0]["content"]
    assert "print('Hello, world!')" in messages[1]["content"]
    CodeGeneration = make_task_spec(
        {"question": FieldSpec.input("question"), "code": FieldSpec.output("code", type_=Code)},
        instructions="Generate code to answer the question",
    )
    adapter = JSONAdapter()
    lm = LM(model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'code': 'print(\"Hello, world!\")'}"))],
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


def test_json_adapter_formats_conversation_history():
    MySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "history": FieldSpec.input("history", type_=TurnLog),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `question`, `history`, produce the fields `answer`.",
    )
    history = TurnLog(
        turns=(
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        )
    )
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=MySignature,
        demos=[],
        inputs={"question": "What is the capital of France?", "history": history},
    )
    assert len(messages) == 6
    assert messages[1]["content"] == "[[ ## question ## ]]\nWhat is the capital of France?"
    assert messages[2]["content"] == '{\n  "answer": "Paris"\n}'
    assert messages[3]["content"] == "[[ ## question ## ]]\nWhat is the capital of Germany?"
    assert messages[4]["content"] == '{\n  "answer": "Berlin"\n}'


@pytest.mark.asyncio
async def test_json_adapter_on_pydantic_model_async(make_run):
    from litellm.utils import Choices, Message, ModelResponse

    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    TestSignature = make_task_spec(
        {
            "user": FieldSpec.input("user", type_=User, desc="The user who asks the question"),
            "question": FieldSpec.input("question", desc="Question the user asks"),
            "answer": FieldSpec.output("answer", type_=Answer, desc="Answer to this question"),
        },
        instructions="Given the fields `user`, `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}"
                    )
                )
            ],
            model="openai/gpt-4o",
        )
        run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
        result = await program(
            user=User(id=5, name="name_test", email="email_test"),
            question="What is the capital of France?",
            run=run,
        )
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content
        expected_output_structure = 'Outputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": \\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": [\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        assert expected_output_structure in content
        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None
        expected_input_data = '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\nWhat is the capital of France?\n\n'
        assert expected_input_data in user_message_content
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error(make_run):
    TestSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="String output field")},
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.side_effect = RuntimeError("Structured output failed!")
        with pytest.raises(LMUnexpectedError, match="Structured output failed"):
            asyncio.run(program(question="Dummy question!", run=run))
        assert mock_completion.call_count == 1
        _, first_call_kwargs = mock_completion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)


def test_json_adapter_json_mode_no_structured_outputs(make_run):
    TestSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="String output field")},
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    program = Predict(TestSignature)
    with (
        mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion,
        mock.patch("litellm.get_supported_openai_params") as mock_get_supported_openai_params,
        mock.patch("litellm.supports_response_schema") as mock_supports_response_schema,
    ):
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]
        )
        mock_get_supported_openai_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False
        result = asyncio.run(program(question="Dummy question!", run=run))
        assert mock_completion.call_count == 1
        assert result.answer == "Test output"
        _, call_kwargs = mock_completion.call_args_list[0]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_json_mode_no_structured_outputs_async(make_run):
    TestSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="String output field")},
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with (
        mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_acompletion,
        mock.patch("litellm.get_supported_openai_params") as mock_get_supported_openai_params,
        mock.patch("litellm.supports_response_schema") as mock_supports_response_schema,
    ):
        mock_acompletion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]
        )
        mock_get_supported_openai_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False
        run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
        result = await program(question="Dummy question!", run=run)
        assert mock_acompletion.call_count == 1
        assert result.answer == "Test output"
        _, call_kwargs = mock_acompletion.call_args_list[0]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error_async(make_run):
    TestSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="String output field")},
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = RuntimeError("Structured output failed!")
        run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
        with pytest.raises(LMUnexpectedError, match="Structured output failed"):
            await program(question="Dummy question!", run=run)
        assert mock_acompletion.call_count == 1
        _, first_call_kwargs = mock_acompletion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)


def test_error_message_on_json_adapter_failure(make_run):
    TestSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="String output field")},
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.side_effect = RuntimeError("RuntimeError!")
        with pytest.raises(LMUnexpectedError) as error:
            asyncio.run(program(question="Dummy question!", run=run))
        assert "RuntimeError!" in str(error.value)
        mock_completion.side_effect = ValueError("ValueError!")
        with pytest.raises(LMUnexpectedError) as error:
            asyncio.run(program(question="Dummy question!", run=run))
        assert "ValueError!" in str(error.value)


@pytest.mark.asyncio
async def test_error_message_on_json_adapter_failure_async(make_run):
    TestSignature = make_task_spec(
        {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="String output field")},
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_acompletion:
        run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
        mock_acompletion.side_effect = RuntimeError("RuntimeError!")
        with pytest.raises(LMUnexpectedError) as error:
            await program(question="Dummy question!", run=run)
        assert "RuntimeError!" in str(error.value)
        mock_acompletion.side_effect = ValueError("ValueError!")
        with pytest.raises(LMUnexpectedError) as error:
            await program(question="Dummy question!", run=run)
        assert "ValueError!" in str(error.value)


def test_json_adapter_toolcalls_native_function_calling():
    MySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "answer": FieldSpec.output("answer"),
            "tool_calls": FieldSpec.output("tool_calls", type_=ToolCalls),
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


def test_json_adapter_toolcalls_no_native_function_calling():
    MySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "answer": FieldSpec.output("answer"),
            "tool_calls": FieldSpec.output("tool_calls", type_=ToolCalls),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`, `tool_calls`.",
    )

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [Tool(get_weather, description="Get the weather for a city")]
    with mock.patch("dspy.adapters.json_adapter._get_structured_outputs_response_format") as mock_structured:
        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="{'answer': 'sunny', 'tool_calls': {'tool_calls': []}}"))],
                model="openai/gpt-4o-mini",
            )
            adapter = JSONAdapter(use_native_function_calling=False)
            lm = LM(model="openai/gpt-4o-mini")
            asyncio.run(
                adapter(
                    lm=lm,
                    config={},
                    task_spec=MySignature,
                    demos=[],
                    inputs={"question": "What is the weather in Tokyo?", "tools": tools},
                    run=make_adapter_run(lm=lm, adapter=adapter),
                )
            )
        mock_structured.assert_not_called()
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_json_adapter_native_reasoning():
    MySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "reasoning": FieldSpec.output("reasoning", type_=Reasoning),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `answer`.",
    )
    adapter = JSONAdapter()
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
                        content="{'answer': 'Paris'}",
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


def test_json_adapter_with_responses_api(make_run):
    TestSignature = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="openai/gpt-4o",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=cast(
                    "Any",
                    [{"type": "output_text", "text": '{"answer": "Washington, D.C."}', "annotations": []}],
                ),
            )
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=ResponseAPIUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        user=None,
    )
    lm = LM(model="openai/gpt-4o", model_type="responses")
    run = make_run(lm=lm, adapter=JSONAdapter())
    program = Predict(TestSignature)
    with mock.patch("litellm.aresponses", new_callable=mock.AsyncMock, return_value=api_response) as mock_responses:
        result = asyncio.run(program(question="What is the capital of the USA?", run=run))
    assert result.answer == "Washington, D.C."
    mock_responses.assert_called_once()
    call_kwargs = mock_responses.call_args.kwargs
    assert "response_format" not in call_kwargs
    assert "text" in call_kwargs
    assert isinstance(call_kwargs["text"]["format"], dict)
    assert isinstance(call_kwargs["text"]["format"]["name"], str)
    assert call_kwargs["text"]["format"]["type"] == "json_schema"
    assert isinstance(call_kwargs["text"]["format"]["schema"], dict)


def test_format_system_message():
    MySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "answers": FieldSpec.output("answers", type_=list[str]),
            "scores": FieldSpec.output("scores", type_=list[float]),
        },
        instructions="Answer the question with multiple answers and scores",
    )
    adapter = JSONAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answers` (list[str]): \n2. `scores` (list[float]):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answers": "{answers}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"string\\"}}",\n  "scores": "{scores}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"number\\"}}"\n}\nIn adhering to this structure, your objective is: \n        Answer the question with multiple answers and scores'
    assert system_message == expected_system_message
