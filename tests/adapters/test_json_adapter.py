import enum
from typing import Literal
from unittest import mock

import pydantic
import pytest

from dspy.adapters.types.document import Document
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

try:
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
    from litellm.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]
from openai.types.responses import ResponseOutputMessage

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.code import Code
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature, make_signature
from dspy.utils.exceptions import LMUnexpectedError
from tests.adapters.conftest import format_messages_and_lm_kwargs


def test_json_adapter_format_exact_messages_for_simple_signature():
    class StringSignature(Signature):
        question: str = InputField()
        answer: str = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(),
        StringSignature,
        demos=[],
        inputs={"question": "What is the capital of France?"},
    )

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.

{
  "answer": "{answer}"
}
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`.""",
        },
        {
            "role": "user",
            "content": """[[ ## question ## ]]
What is the capital of France?

Respond with a JSON object in the following order of fields: `answer`.""",
        },
    ]


def test_json_adapter_format_exact_messages_with_demo_and_typed_output():
    class MultiAnswer(Signature):
        question: str = InputField()
        answer: str = OutputField()
        confidence: float = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(),
        MultiAnswer,
        demos=[{"question": "Q1", "answer": "A1", "confidence": 0.9}],
        inputs={"question": "Q2"},
    )

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str):\x20
2. `confidence` (float):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.

{
  "answer": "{answer}",
  "confidence": "{confidence}        # note: the value you produce must be a single float value"
}
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`, `confidence`.""",
        },
        {"role": "user", "content": """[[ ## question ## ]]
Q1"""},
        {
            "role": "assistant",
            "content": """{
  "answer": "A1",
  "confidence": 0.9
}""",
        },
        {
            "role": "user",
            "content": """[[ ## question ## ]]
Q2

Respond with a JSON object in the following order of fields: `answer`, then `confidence` (must be formatted as a valid Python float).""",
        },
    ]


def test_json_adapter_format_exact_messages_with_described_and_bool_outputs():
    class TestSignature(Signature):
        input1: str = InputField()
        output1: str = OutputField(desc="String output field")
        output2: bool = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(), TestSignature, [], {"input1": "Test input"})

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `input1` (str):
Your output fields are:
1. `output1` (str): String output field
2. `output2` (bool):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## input1 ## ]]
{input1}

Outputs will be a JSON object with the following fields.

{
  "output1": "{output1}",
  "output2": "{output2}        # note: the value you produce must be True or False"
}
In adhering to this structure, your objective is:\x20
        Given the fields `input1`, produce the fields `output1`, `output2`.""",
        },
        {
            "role": "user",
            "content": """[[ ## input1 ## ]]
Test input

Respond with a JSON object in the following order of fields: `output1`, then `output2` (must be formatted as a valid Python bool).""",
        },
    ]


def test_json_adapter_format_exact_messages_with_history_demo_pydantic_tools_and_image():
    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
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

    class RichRenderingSignature(Signature):
        """Answer using all supplied context."""

        history: History = InputField()
        image: Image = InputField()
        tools: list[Tool] = InputField()
        profile: Profile = InputField()
        question: str = InputField()
        answer: AnswerCard = OutputField()

    tool = Tool(search)
    demo_profile = Profile(
        name="Ada",
        location=Location(city="London", country="UK"),
        interests=["math", "machines"],
    )
    current_profile = Profile(
        name="Grace",
        location=Location(city="Arlington", country="USA"),
        interests=["compilers", "navy"],
    )
    history = History(
        messages=[
            {
                "profile": demo_profile,
                "question": "Who is Ada?",
                "answer": AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
            }
        ]
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(),
        RichRenderingSignature,
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

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `history` (History): \n'
                 '2. `image` (Image): \n'
                 '3. `tools` (list[Tool]): \n'
                 '4. `profile` (Profile): \n'
                 '5. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `answer` (AnswerCard):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## history ## ]]\n'
                 '{history}\n'
                 '\n'
                 '[[ ## image ## ]]\n'
                 '{image}\n'
                 '\n'
                 '[[ ## tools ## ]]\n'
                 '{tools}\n'
                 '\n'
                 '[[ ## profile ## ]]\n'
                 '{profile}\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "answer": "{answer}        # note: the value you produce must adhere to the JSON '
                 'schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"answer\\": {\\"type\\": '
                 '\\"string\\", \\"title\\": \\"Answer\\"}, \\"sources\\": {\\"type\\": \\"array\\", '
                 '\\"items\\": {\\"type\\": \\"string\\"}, \\"title\\": \\"Sources\\"}}, '
                 '\\"required\\": [\\"answer\\", \\"sources\\"], \\"title\\": \\"AnswerCard\\"}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Answer using all supplied context.'},
     {"role": "user",
      "content": [{"type": "text",
                   "text": "This is an example of the task, though some input or output fields are not "
                           "supplied.\n"
                           "\n"
                           "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Ada", "location": {"city": "London", "country": "UK"}, '
                           '"interests": ["math", "machines"]}\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should we mention?'}]},
     {"role": "assistant",
      "content": '{\n'
                 '  "answer": {\n'
                 '    "answer": "Mention analytical engines.",\n'
                 '    "sources": [\n'
                 '      "demo"\n'
                 '    ]\n'
                 '  }\n'
                 '}'},
     {"role": "user",
      "content": '[[ ## profile ## ]]\n'
                 '{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": '
                 '["math", "machines"]}\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 'Who is Ada?'},
     {"role": "assistant",
      "content": '{\n'
                 '  "answer": {\n'
                 '    "answer": "Ada is a mathematician.",\n'
                 '    "sources": [\n'
                 '      "memory"\n'
                 '    ]\n'
                 '  }\n'
                 '}'},
     {"role": "user",
      "content": [{"type": "text", "text": "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, '
                           '"interests": ["compilers", "navy"]}\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should the answer include?\n'
                           '\n'
                           'Respond with a JSON object in the following order of fields: `answer` '
                           '(must be formatted as a valid Python AnswerCard).'}]}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

def test_json_adapter_format_exact_messages_with_int_and_mapping_outputs():
    class IntDictSignature(Signature):
        question: str = InputField()
        count: int = OutputField()
        metadata: dict[str, int] = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(), IntDictSignature, [], {"question": "Count things"})

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `count` (int): \n'
                 '2. `metadata` (dict[str, int]):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "count": "{count}        # note: the value you produce must be a single int '
                 'value",\n'
                 '  "metadata": "{metadata}        # note: the value you produce must adhere to the '
                 'JSON schema: {\\"type\\": \\"object\\", \\"additionalProperties\\": {\\"type\\": '
                 '\\"integer\\"}}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `count`, `metadata`.'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Count things\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `count` (must be "
                 "formatted as a valid Python int), then `metadata` (must be formatted as a valid "
                 "Python dict[str, int])."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_literal_and_enum_outputs():
    class Label(enum.Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"

    class LiteralEnumSignature(Signature):
        text: str = InputField()
        decision: Literal["accept", "reject"] = OutputField()
        label: Label = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(), LiteralEnumSignature, [], {"text": "Looks good"})

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `text` (str):\n'
                 'Your output fields are:\n'
                 "1. `decision` (Literal['accept', 'reject']): \n"
                 '2. `label` (Label):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## text ## ]]\n'
                 '{text}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "decision": "{decision}        # note: the value you produce must exactly match '
                 '(no extra characters) one of: accept; reject",\n'
                 '  "label": "{label}        # note: the value you produce must be one of: positive; '
                 'negative"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `text`, produce the fields `decision`, `label`.'},
     {"role": "user",
      "content": "[[ ## text ## ]]\n"
                 "Looks good\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `decision` (must be "
                 "formatted as a valid Python Literal['accept', 'reject']), then `label` (must be "
                 "formatted as a valid Python Label)."}]
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

    class PydanticSignature(Signature):
        question: str = InputField()
        summary: JsonNestedSummary = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(), PydanticSignature, [], {"question": "Summarize"})

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `summary` (JsonNestedSummary):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "summary": "{summary}        # note: the value you produce must adhere to the JSON '
                 'schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"JsonNestedAddress\\": '
                 '{\\"type\\": \\"object\\", \\"properties\\": {\\"city\\": {\\"type\\": \\"string\\", '
                 '\\"title\\": \\"City\\"}, \\"country\\": {\\"type\\": \\"string\\", \\"title\\": '
                 '\\"Country\\"}}, \\"required\\": [\\"city\\", \\"country\\"], \\"title\\": '
                 '\\"JsonNestedAddress\\"}}, \\"properties\\": {\\"address\\": {\\"$ref\\": '
                 '\\"#/$defs/JsonNestedAddress\\"}, \\"scores\\": {\\"type\\": \\"array\\", '
                 '\\"items\\": {\\"type\\": \\"number\\"}, \\"title\\": \\"Scores\\"}, \\"title\\": '
                 '{\\"type\\": \\"string\\", \\"title\\": \\"Title\\"}}, \\"required\\": [\\"title\\", '
                 '\\"address\\", \\"scores\\"], \\"title\\": \\"JsonNestedSummary\\"}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `summary`.'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Summarize\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `summary` (must be "
                 "formatted as a valid Python JsonNestedSummary)."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_incomplete_demo():
    class IncompleteDemoSignature(Signature):
        question: str = InputField()
        context: str = InputField()
        answer: str = OutputField()
        score: float = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(),
        IncompleteDemoSignature,
        [{"question": "Q1", "answer": "A1"}],
        {"question": "Q2", "context": "C2"},
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str): \n'
                 '2. `context` (str):\n'
                 'Your output fields are:\n'
                 '1. `answer` (str): \n'
                 '2. `score` (float):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 '[[ ## context ## ]]\n'
                 '{context}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "answer": "{answer}",\n'
                 '  "score": "{score}        # note: the value you produce must be a single float '
                 'value"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, `context`, produce the fields `answer`, '
                 '`score`.'},
     {"role": "user",
      "content": "This is an example of the task, though some input or output fields are not "
                 "supplied.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "Q1"},
     {"role": "assistant",
      "content": '{\n  "answer": "A1",\n  "score": "Not supplied for this particular example. "\n}'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 "[[ ## context ## ]]\n"
                 "C2\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `answer`, then `score` "
                 "(must be formatted as a valid Python float)."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_and_lm_kwargs_with_native_tool_calling():
    class FunctionCallingLM(DummyLM):
        @property
        def supports_function_calling(self):
            return True

    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
        return query

    class NativeToolSignature(Signature):
        question: str = InputField()
        tools: list[Tool] = InputField()
        tool_calls: ToolCalls = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        JSONAdapter(use_native_function_calling=True),
        NativeToolSignature,
        [],
        {"question": "Q?", "tools": [Tool(search)]},
        lm=FunctionCallingLM([{}]),
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "Inputs will have the following structure:\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "Outputs will be a JSON object with the following fields.\n"
                 "\n"
                 "{}\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, `tools`, produce the fields `tool_calls`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q?\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: ."}]
    assert messages == expected_messages
    expected_lm_kwargs = {"tools": [{"type": "function",
                "function": {"name": "search",
                             "description": "Search for documents.",
                             "parameters": {"type": "object",
                                            "properties": {"query": {"type": "string"},
                                                           "k": {"type": "integer", "default": 3}},
                                            "required": ["query", "k"]}}}]}
    assert lm_kwargs == expected_lm_kwargs

def test_json_adapter_format_exact_messages_with_tool_calls_output_demo():
    class ToolCallsSignature(Signature):
        question: str = InputField()
        tool_calls: ToolCalls = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(JSONAdapter(use_native_function_calling=False),
        ToolCallsSignature,
        [{"question": "Q1", "tool_calls": ToolCalls.from_dict_list([{"name": "search", "args": {"query": "cats"}}])}],
        {"question": "Q2"},
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `tool_calls` (ToolCalls): \n'
                 '    Type description of ToolCalls: Tool calls must be a JSON object with `tool_calls`, '
                 'a list of calls. Each call must include `name` and `args`. Example: {"tool_calls": '
                 '[{"name": "search", "args": {"query": "cats"}}]}\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "tool_calls": "{tool_calls}        # note: the value you produce must adhere to '
                 'the JSON schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"ToolCall\\": '
                 '{\\"type\\": \\"object\\", \\"properties\\": {\\"args\\": {\\"type\\": \\"object\\", '
                 '\\"additionalProperties\\": true, \\"title\\": \\"Args\\"}, \\"name\\": {\\"type\\": '
                 '\\"string\\", \\"title\\": \\"Name\\"}}, \\"required\\": [\\"name\\", \\"args\\"], '
                 '\\"title\\": \\"ToolCall\\"}}, \\"properties\\": {\\"tool_calls\\": {\\"type\\": '
                 '\\"array\\", \\"items\\": {\\"$ref\\": \\"#/$defs/ToolCall\\"}, \\"title\\": \\"Tool '
                 'Calls\\"}}, \\"required\\": [\\"tool_calls\\"], \\"title\\": \\"ToolCalls\\"}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `tool_calls`.'},
     {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
     {"role": "assistant",
      "content": '{\n'
                 '  "tool_calls": {\n'
                 '    "tool_calls": [\n'
                 '      {\n'
                 '        "name": "search",\n'
                 '        "args": {\n'
                 '          "query": "cats"\n'
                 '        }\n'
                 '      }\n'
                 '    ]\n'
                 '  }\n'
                 '}'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 'Respond with a JSON object in the following order of fields: `tool_calls` (must be '
                 'a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).'}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_non_native_tool_result_history_field():
    def search(query: str) -> str:
        return query

    class ToolHistorySignature(Signature):
        question: str = InputField()
        history: History = InputField()
        tools: list[Tool] = InputField()
        next_thought: str = OutputField()
        tool_calls: ToolCalls = OutputField()

    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = ToolCallResults.from_tool_calls_and_values([tool_call], ["cat"])

    messages, _lm_kwargs = format_messages_and_lm_kwargs(
        JSONAdapter(use_native_function_calling=False),
        ToolHistorySignature,
        [],
        {
            "question": "Q2",
            "history": History(
                messages=[
                    {
                        "question": "Q1",
                        "next_thought": "I should search.",
                        "tool_calls": ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                    }
                ]
            ),
            "tools": [Tool(search)],
        },
    )

    assert messages[3]["content"] == (
        "[[ ## tool_call_results ## ]]\n"
        '{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}'
    )
    assert messages[4]["content"] == (
        "[[ ## question ## ]]\n"
        "Q2\n"
        "\n"
        "[[ ## tools ## ]]\n"
        '["search. It takes arguments {\'query\': {\'type\': \'string\'}}."]\n'
        "\n"
        "Respond with a JSON object in the following order of fields: `next_thought`, then "
        '`tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).'
    )


def test_json_adapter_passes_structured_output_when_supported_by_model():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1", ge=0, le=10)
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(Signature):
        input1: str = InputField()
        output1: str = OutputField()  # Description intentionally left blank
        output2: bool = OutputField(desc="Boolean output field")
        output3: OutputField3 = OutputField(desc="Nested output field")
        output4_unannotated = OutputField(desc="Unannotated output field")

    program = Predict(TestSignature)

    # Configure DSPy to use an OpenAI LM that supports structured outputs
    settings.configure(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        program(input1="Test input")

    def clean_schema_extra(field_name, field_info):
        attrs = dict(field_info.__repr_args__())
        if "json_schema_extra" in attrs:
            attrs["json_schema_extra"] = {
                k: v
                for k, v in attrs["json_schema_extra"].items()
                if k != "__dspy_field_type" and not (k == "desc" and v == f"${{{field_name}}}")
            }
        return attrs

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    response_format = call_kwargs.get("response_format")
    assert response_format is not None
    assert issubclass(response_format, pydantic.BaseModel)
    assert response_format.model_fields.keys() == {"output1", "output2", "output3", "output4_unannotated"}


def test_json_adapter_not_using_structured_outputs_when_not_supported_by_model():
    class TestSignature(Signature):
        input1: str = InputField()
        output1: str = OutputField()
        output2: bool = OutputField()

    program = Predict(TestSignature)

    # Configure DSPy to use a model from a fake provider that doesn't support structured outputs
    settings.configure(lm=LM(model="fakeprovider/fakemodel", cache=False), adapter=JSONAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content=("{'output1': 'Test output', 'output2': True}")))],
            model="openai/gpt-4o",
        )

        program(input1="Test input")

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert "response_format" not in call_kwargs



def test_json_adapter_with_structured_outputs_does_not_mutate_original_signature():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1")
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(Signature):
        input1: str = InputField()
        output1: str = OutputField()  # Description intentionally left blank
        output2: bool = OutputField(desc="Boolean output field")
        output3: OutputField3 = OutputField(desc="Nested output field")
        output4_unannotated = OutputField(desc="Unannotated output field")

    settings.configure(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    program = Predict(TestSignature)
    with mock.patch("litellm.completion"):
        program(input1="Test input")

    assert program.signature.output_fields == TestSignature.output_fields


def test_json_adapter_sync_call():
    signature = make_signature("question->answer")
    adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=adapter)
    with settings.context(adapter=adapter):
        result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


@pytest.mark.asyncio
async def test_json_adapter_async_call():
    signature = make_signature("question->answer")
    adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=adapter)
    with settings.context(adapter=adapter):
        result = await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


def test_json_adapter_on_pydantic_model():
    from litellm.utils import Choices, Message, ModelResponse

    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    class TestSignature(Signature):
        user: User = InputField(desc="The user who asks the question")
        question: str = InputField(desc="Question the user asks")
        answer: Answer = OutputField(desc="Answer to this question")

    program = Predict(TestSignature)

    settings.configure(lm=LM(model="openai/gpt-4o", cache=False), adapter=JSONAdapter())

    with mock.patch("litellm.completion") as mock_completion:
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
        result = program(
            user={"id": 5, "name": "name_test", "email": "email_test"}, question="What is the capital of France?"
        )

        # Check that litellm.completion was called exactly once
        mock_completion.assert_called_once()

        _, call_kwargs = mock_completion.call_args
        # Assert that there are exactly 2 messages (system + user)
        assert len(call_kwargs["messages"]) == 2

        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None

        # Assert that system prompt includes correct input field descriptions
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content

        # Assert that system prompt includes correct output field description
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content

        # Assert that system prompt includes input formatting structure
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content

        # Assert that system prompt includes output formatting structure
        expected_output_structure = (
            "Outputs will be a JSON object with the following fields.\n\n{\n  "
            '"answer": "{answer}        # note: the value you produce must adhere to the JSON schema: '
            '{\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": '
            '\\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": '
            '[\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        )
        assert expected_output_structure in content

        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None

        # Assert that the user input data is formatted correctly
        expected_input_data = (
            '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\n'
            "What is the capital of France?\n\n"
        )
        assert expected_input_data in user_message_content

        # Assert that the adapter output has expected fields and values
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_parse_raise_error_on_mismatch_fields():
    signature = make_signature("question->answer")
    adapter = JSONAdapter()
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'answer1': 'Paris'}")),
            ],
            model="openai/gpt-4o",
        )
        lm = LM(model="openai/gpt-4o-mini")
        with pytest.raises(AdapterParseError) as e:
            adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

    assert e.value.adapter_name == "JSONAdapter"
    assert e.value.signature == signature
    assert e.value.lm_response == "{'answer1': 'Paris'}"
    assert e.value.parsed_result == {}

    assert str(e.value) == (
        "Adapter JSONAdapter failed to parse the LM response. \n\n"
        "LM Response: {'answer1': 'Paris'} \n\n"
        "Expected to find output fields in the LM response: [answer] \n\n"
        "Actual output fields parsed from the LM response: [] \n\n"
    )


def test_json_adapter_formats_image():
    # Test basic image formatting
    image = Image(url="https://example.com/image.jpg")

    class MySignature(Signature):
        image: Image = InputField()
        text: str = OutputField()

    adapter = JSONAdapter()
    messages = adapter.format(MySignature, [], {"image": image})

    assert len(messages) == 2
    user_message_content = messages[1]["content"]
    assert user_message_content is not None

    # The message should have 3 chunks of types: text, image_url, text
    assert len(user_message_content) == 3
    assert user_message_content[0]["type"] == "text"
    assert user_message_content[2]["type"] == "text"

    # Assert that the image is formatted correctly
    expected_image_content = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    assert expected_image_content in user_message_content


def test_json_adapter_formats_image_with_few_shot_examples():
    class MySignature(Signature):
        image: Image = InputField()
        text: str = OutputField()

    adapter = JSONAdapter()

    demos = [
        Example(
            image=Image(url="https://example.com/image1.jpg"),
            text="This is a test image",
        ),
        Example(
            image=Image(url="https://example.com/image2.jpg"),
            text="This is another test image",
        ),
    ]
    messages = adapter.format(MySignature, demos, {"image": Image(url="https://example.com/image3.jpg")})  # ty:ignore[invalid-argument-type]

    # 1 system message, 2 few shot examples (1 user and assistant message for each example), 1 user message
    assert len(messages) == 6

    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in messages[1]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in messages[3]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}} in messages[5]["content"]


def test_json_adapter_formats_image_with_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    class MySignature(Signature):
        image: ImageWrapper = InputField()
        text: str = OutputField()

    image1 = Image(url="https://example.com/image1.jpg")
    image2 = Image(url="https://example.com/image2.jpg")
    image3 = Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])

    adapter = JSONAdapter()
    messages = adapter.format(MySignature, [], {"image": image_wrapper})

    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}

    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]


def test_json_adapter_formats_with_nested_documents():
    class DocumentWrapper(pydantic.BaseModel):
        documents: list[Document]

    class MySignature(Signature):
        document: DocumentWrapper = InputField()
        text: str = OutputField()

    doc1 = Document(data="Hello, world!")
    doc2 = Document(data="Hello, world 2!")

    document_wrapper = DocumentWrapper(documents=[doc1, doc2])

    adapter = JSONAdapter()
    messages = adapter.format(MySignature, [], {"document": document_wrapper})

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

    class MySignature(Signature):
        image: ImageWrapper = InputField()
        text: str = OutputField()

    image1 = Image(url="https://example.com/image1.jpg")
    image2 = Image(url="https://example.com/image2.jpg")
    image3 = Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    demos = [
        Example(
            image=image_wrapper,
            text="This is a test image",
        ),
    ]

    image_wrapper_2 = ImageWrapper(images=[Image(url="https://example.com/image4.jpg")], tag=["test", "example"])
    adapter = JSONAdapter()
    messages = adapter.format(MySignature, demos, {"image": image_wrapper_2})  # ty:ignore[invalid-argument-type]

    assert len(messages) == 4

    # Image information in the few-shot example's user message
    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]

    # The query image is formatted in the last user message
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image4.jpg"}} in messages[-1]["content"]


def test_json_adapter_with_tool():
    class MySignature(Signature):
        """Answer question with the help of the tools"""

        question: str = InputField()
        tools: list[Tool] = InputField()
        answer: str = OutputField()
        tool_calls: ToolCalls = OutputField()

    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny"

    def get_population(country: str, year: int) -> str:
        """Get the population for a country"""
        return f"The population of {country} in {year} is 1000000"

    tools = [Tool(get_weather), Tool(get_population)]

    adapter = JSONAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

    assert len(messages) == 2

    # The output field type description should be included in the system message even if the output field is nested
    assert ToolCalls.description() in messages[0]["content"]

    # The user message should include the question and the tools
    assert "What is the weather in Tokyo?" in messages[1]["content"]
    assert "get_weather" in messages[1]["content"]
    assert "get_population" in messages[1]["content"]

    # Tool arguments format should be included in the user message
    assert "{'city': {'type': 'string'}}" in messages[1]["content"]
    assert "{'country': {'type': 'string'}, 'year': {'type': 'integer'}}" in messages[1]["content"]

    with mock.patch("litellm.completion") as mock_completion:
        lm = LM(model="openai/gpt-4o-mini")
        adapter(lm, {}, MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args

    # Assert tool calls are included in the `tools` arg
    assert len(call_kwargs["tools"]) > 0
    assert call_kwargs["tools"][0] == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                    },
                },
                "required": ["city"],
            },
        },
    }
    assert call_kwargs["tools"][1] == {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population for a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                    },
                    "year": {
                        "type": "integer",
                    },
                },
                "required": ["country", "year"],
            },
        },
    }


def test_json_adapter_with_code():
    # Test with code as input field
    class CodeAnalysis(Signature):
        """Analyze the time complexity of the code"""

        code: Code = InputField()
        result: str = OutputField()

    adapter = JSONAdapter()
    messages = adapter.format(CodeAnalysis, [], {"code": "print('Hello, world!')"})

    assert len(messages) == 2

    # The output field type description should be included in the system message even if the output field is nested
    assert Code.description() in messages[0]["content"]

    # The user message should include the question and the tools
    assert "print('Hello, world!')" in messages[1]["content"]

    # Test with code as output field
    class CodeGeneration(Signature):
        """Generate code to answer the question"""

        question: str = InputField()
        code: Code = OutputField()

    adapter = JSONAdapter()
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'code': 'print(\"Hello, world!\")'}"))],
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            LM(model="openai/gpt-4o-mini", cache=False),
            {},
            CodeGeneration,
            [],
            {"question": "Write a python program to print 'Hello, world!'"},
        )
        assert result[0]["code"].code == 'print("Hello, world!")'


def test_json_adapter_formats_conversation_history():
    class MySignature(Signature):
        question: str = InputField()
        history: History = InputField()
        answer: str = OutputField()

    history = History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )

    adapter = JSONAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the capital of France?", "history": history})

    assert len(messages) == 6
    assert messages[1]["content"] == "[[ ## question ## ]]\nWhat is the capital of France?"
    assert messages[2]["content"] == '{\n  "answer": "Paris"\n}'
    assert messages[3]["content"] == "[[ ## question ## ]]\nWhat is the capital of Germany?"
    assert messages[4]["content"] == '{\n  "answer": "Berlin"\n}'


@pytest.mark.asyncio
async def test_json_adapter_on_pydantic_model_async():
    from litellm.utils import Choices, Message, ModelResponse

    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    class TestSignature(Signature):
        user: User = InputField(desc="The user who asks the question")
        question: str = InputField(desc="Question the user asks")
        answer: Answer = OutputField(desc="Answer to this question")

    program = Predict(TestSignature)

    with mock.patch("litellm.acompletion") as mock_completion:
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

        with settings.context(lm=LM(model="openai/gpt-4o", cache=False), adapter=JSONAdapter()):
            result = await program.acall(
                user={"id": 5, "name": "name_test", "email": "email_test"}, question="What is the capital of France?"
            )

        # Check that litellm.acompletion was called exactly once
        mock_completion.assert_called_once()

        _, call_kwargs = mock_completion.call_args
        # Assert that there are exactly 2 messages (system + user)
        assert len(call_kwargs["messages"]) == 2

        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None

        # Assert that system prompt includes correct input field descriptions
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content

        # Assert that system prompt includes correct output field description
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content

        # Assert that system prompt includes input formatting structure
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content

        # Assert that system prompt includes output formatting structure
        expected_output_structure = (
            "Outputs will be a JSON object with the following fields.\n\n{\n  "
            '"answer": "{answer}        # note: the value you produce must adhere to the JSON schema: '
            '{\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": '
            '\\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": '
            '[\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        )
        assert expected_output_structure in content

        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None

        # Assert that the user input data is formatted correctly
        expected_input_data = (
            '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\n'
            "What is the capital of France?\n\n"
        )
        assert expected_input_data in user_message_content

        # Assert that the adapter output has expected fields and values
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField(desc="String output field")

    settings.configure(lm=LM(model="openai/gpt-4o-mini", cache=False), adapter=JSONAdapter())
    program = Predict(TestSignature)

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = RuntimeError("Structured output failed!")

        with pytest.raises(LMUnexpectedError, match="Structured output failed"):
            program(question="Dummy question!")

        assert mock_completion.call_count == 1
        _, first_call_kwargs = mock_completion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)


def test_json_adapter_json_mode_no_structured_outputs():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField(desc="String output field")

    settings.configure(lm=LM(model="openai/gpt-4o", cache=False), adapter=JSONAdapter())
    program = Predict(TestSignature)

    with (
        mock.patch("litellm.completion") as mock_completion,
        mock.patch("litellm.get_supported_openai_params") as mock_get_supported_openai_params,
        mock.patch("litellm.supports_response_schema") as mock_supports_response_schema,
    ):
        # Call a model that allows json but not structured outputs
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]
        )
        mock_get_supported_openai_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False

        result = program(question="Dummy question!")

        assert mock_completion.call_count == 1
        assert result.answer == "Test output"

        _, call_kwargs = mock_completion.call_args_list[0]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_json_mode_no_structured_outputs_async():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField(desc="String output field")

    program = Predict(TestSignature)

    with (
        mock.patch("litellm.acompletion") as mock_acompletion,
        mock.patch("litellm.get_supported_openai_params") as mock_get_supported_openai_params,
        mock.patch("litellm.supports_response_schema") as mock_supports_response_schema,
    ):
        # Call a model that allows json but not structured outputs
        mock_acompletion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]
        )
        mock_get_supported_openai_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False

        with settings.context(lm=LM(model="openai/gpt-4o", cache=False), adapter=JSONAdapter()):
            result = await program.acall(question="Dummy question!")

        assert mock_acompletion.call_count == 1
        assert result.answer == "Test output"

        _, call_kwargs = mock_acompletion.call_args_list[0]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error_async():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField(desc="String output field")

    program = Predict(TestSignature)

    with mock.patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.side_effect = RuntimeError("Structured output failed!")

        with settings.context(lm=LM(model="openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):  # noqa: SIM117
            with pytest.raises(LMUnexpectedError, match="Structured output failed"):
                await program.acall(question="Dummy question!")

        assert mock_acompletion.call_count == 1
        _, first_call_kwargs = mock_acompletion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)


def test_error_message_on_json_adapter_failure():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField(desc="String output field")

    program = Predict(TestSignature)

    settings.configure(lm=LM(model="openai/gpt-4o-mini", cache=False), adapter=JSONAdapter())

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = RuntimeError("RuntimeError!")

        with pytest.raises(LMUnexpectedError) as error:
            program(question="Dummy question!")

        assert "RuntimeError!" in str(error.value)

        mock_completion.side_effect = ValueError("ValueError!")
        with pytest.raises(LMUnexpectedError) as error:
            program(question="Dummy question!")

        assert "ValueError!" in str(error.value)


@pytest.mark.asyncio
async def test_error_message_on_json_adapter_failure_async():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField(desc="String output field")

    program = Predict(TestSignature)

    with mock.patch("litellm.acompletion") as mock_acompletion:  # noqa: SIM117
        with settings.context(lm=LM(model="openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            mock_acompletion.side_effect = RuntimeError("RuntimeError!")
            with pytest.raises(LMUnexpectedError) as error:
                await program.acall(question="Dummy question!")

            assert "RuntimeError!" in str(error.value)

            mock_acompletion.side_effect = ValueError("ValueError!")
            with pytest.raises(LMUnexpectedError) as error:
                await program.acall(question="Dummy question!")

            assert "ValueError!" in str(error.value)


def test_json_adapter_toolcalls_native_function_calling():
    class MySignature(Signature):
        question: str = InputField()
        tools: list[Tool] = InputField()
        answer: str = OutputField()
        tool_calls: ToolCalls = OutputField()

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [Tool(get_weather)]

    adapter = JSONAdapter(use_native_function_calling=True)

    # Case 1: Tool calls are present in the response, while content is None.
    with mock.patch("litellm.completion") as mock_completion:
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
                ),
            ],
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            LM(model="openai/gpt-4o-mini", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the weather in Paris?", "tools": tools},
        )

        assert result[0]["tool_calls"] == ToolCalls(
            tool_calls=[
                ToolCalls.ToolCall(
                    id="call_pQm8ajtSMxgA0nrzK2ivFmxG",
                    name="get_weather",
                    args={"city": "Paris"},
                )
            ]
        )
        # `answer` is not present, so we set it to None
        assert result[0]["answer"] is None

    # Case 2: Tool calls are not present in the response, while content is present.
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            LM(model="openai/gpt-4o-mini", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the weather in Paris?", "tools": tools},
        )
        assert result[0]["answer"] == "Paris"
        assert result[0]["tool_calls"] is None


def test_json_adapter_toolcalls_no_native_function_calling():
    class MySignature(Signature):
        question: str = InputField()
        tools: list[Tool] = InputField()
        answer: str = OutputField()
        tool_calls: ToolCalls = OutputField()

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [Tool(get_weather)]

    # Patch _get_structured_outputs_response_format to track calls
    with mock.patch("dspy.adapters.json_adapter._get_structured_outputs_response_format") as mock_structured:
        # Patch litellm.completion to return a dummy response
        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="{'answer': 'sunny', 'tool_calls': {'tool_calls': []}}"))],
                model="openai/gpt-4o-mini",
            )
            adapter = JSONAdapter(use_native_function_calling=False)
            lm = LM(model="openai/gpt-4o-mini", cache=False)
            adapter(lm, {}, MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

        # _get_structured_outputs_response_format is not called because without using native function calling,
        # JSONAdapter falls back to json mode for stable quality.
        mock_structured.assert_not_called()
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_json_adapter_native_reasoning():
    class MySignature(Signature):
        question: str = InputField()
        reasoning: Reasoning = OutputField()
        answer: str = OutputField()

    adapter = JSONAdapter()

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': 'Paris'}",
                        reasoning_content="Step-by-step thinking about the capital of France",
                    ),
                )
            ],
            model="anthropic/claude-3-7-sonnet-20250219",
        )
        modified_signature = adapter._call_preprocess(
            LM(model="anthropic/claude-3-7-sonnet-20250219", reasoning_effort="low", cache=False),
            {},
            MySignature,
            {"question": "What is the capital of France?"},
        )
        assert "reasoning" not in modified_signature.output_fields

        result = adapter(
            LM(model="anthropic/claude-3-7-sonnet-20250219", reasoning_effort="low", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the capital of France?"},
        )
        assert result[0]["reasoning"] == Reasoning(content="Step-by-step thinking about the capital of France")


def test_json_adapter_with_responses_api():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField()

    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="openai/gpt-4o",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1", type="message", role="assistant", status="completed", content=[{"type": "output_text", "text": '{"answer": "Washington, D.C."}', "annotations": []}],  # ty:ignore[invalid-argument-type]
            ),
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

    lm = LM(model="openai/gpt-4o", model_type="responses", cache=False)
    settings.configure(lm=lm, adapter=JSONAdapter())

    program = Predict(TestSignature)
    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as mock_responses:
        result = program(question="What is the capital of the USA?")

    assert result.answer == "Washington, D.C."
    mock_responses.assert_called_once()
    # Verify that response_format was converted to text.format
    call_kwargs = mock_responses.call_args.kwargs
    assert "response_format" not in call_kwargs
    assert "text" in call_kwargs

    assert isinstance(call_kwargs["text"]["format"], dict)
    assert isinstance(call_kwargs["text"]["format"]["name"], str)
    assert call_kwargs["text"]["format"]["type"] == "json_schema"
    assert isinstance(call_kwargs["text"]["format"]["schema"], dict)


def test_format_system_message():
    class MySignature(Signature):
        """Answer the question with multiple answers and scores"""

        question: str = InputField()
        answers: list[str] = OutputField()
        scores: list[float] = OutputField()

    adapter = JSONAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answers` (list[str]):\x20
2. `scores` (list[float]):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.

{
  "answers": "{answers}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"string\\"}}",
  "scores": "{scores}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"number\\"}}"
}
In adhering to this structure, your objective is:\x20
        Answer the question with multiple answers and scores"""
    assert system_message == expected_system_message
