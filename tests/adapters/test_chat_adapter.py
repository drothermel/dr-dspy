import asyncio
import json
import re
from typing import Any, Literal, cast
from unittest import mock

import pydantic
import pytest
from typing_extensions import override

from dspy.testing import DummyLM

try:
    from litellm.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.audio import Audio
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.code import Code
from dspy.adapters.types.document import Document
from dspy.adapters.types.field_type import FieldTypeMixin
from dspy.adapters.types.file import File
from dspy.adapters.types.image import Image
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
from dspy.primitives import Example
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.conftest import (
    adapter_format_as_openai,
    default_model_response,
    format_messages_and_lm_kwargs,
    make_adapter_run,
)
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


def test_chat_adapter_format_exact_messages_for_simple_signature():
    QA = ts("question -> answer", instructions="Answer the question.")
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(), task_spec=QA, demos=[], inputs={"question": "What is the capital of France?"}
    )
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs
    assert messages == [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer the question.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nWhat is the capital of France?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]


def test_chat_adapter_format_exact_messages_with_demo_and_typed_outputs():
    MultiAnswer = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answers": output_field("answers", type_=list[str], desc="The answers."),
            "scores": output_field("scores", type_=list[float], desc="The scores."),
        },
        instructions="Answer the question with multiple answers and scores",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=MultiAnswer,
        demos=[{"question": "Q1", "answers": ["A1", "A2"], "scores": [0.1, 0.9]}],
        inputs={"question": "Q2"},
    )
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs
    assert messages == [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answers` (list[str]): The answers.\n2. `scores` (list[float]): The scores.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answers ## ]]\n{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}\n\n[[ ## scores ## ]]\n{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer the question with multiple answers and scores',
        },
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {
            "role": "assistant",
            "content": '[[ ## answers ## ]]\n["A1", "A2"]\n\n[[ ## scores ## ]]\n[0.1, 0.9]\n\n[[ ## completed ## ]]\n',
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ2\n\nRespond with the corresponding output fields, starting with the field `[[ ## answers ## ]]` (must be formatted as a valid Python list[str]), then `[[ ## scores ## ]]` (must be formatted as a valid Python list[float]), and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]


def test_chat_adapter_format_exact_messages_with_nested_pydantic_models():

    class Address(pydantic.BaseModel):
        city: str
        country: str

    class Person(pydantic.BaseModel):
        name: str
        address: Address
        tags: list[str]

    class Summary(pydantic.BaseModel):
        headline: str
        score: float

    PydanticSignature = make_task_spec(
        {
            "person": input_field("person", type_=Person, desc="The person."),
            "summary": output_field("summary", type_=Summary, desc="The summary."),
        },
        instructions="Given the fields `person`, produce the fields `summary`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=PydanticSignature,
        demos=[],
        inputs={"person": Person(name="Ada", address=Address(city="London", country="UK"), tags=["math", "code"])},
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `person` (Person): The person.\nYour output fields are:\n1. `summary` (Summary): The summary.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## person ## ]]\n{person}\n\n[[ ## summary ## ]]\n{summary}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "properties": {"headline": {"type": "string", "title": "Headline"}, "score": {"type": "number", "title": "Score"}}, "required": ["headline", "score"], "title": "Summary"}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `person`, produce the fields `summary`.',
        },
        {
            "role": "user",
            "content": '[[ ## person ## ]]\n{"name": "Ada", "address": {"city": "London", "country": "UK"}, "tags": ["math", "code"]}\n\nRespond with the corresponding output fields, starting with the field `[[ ## summary ## ]]` (must be formatted as a valid Python Summary), and then ending with the marker for `[[ ## completed ## ]]`.',
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_incomplete_demo():
    IncompleteDemoSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "context": input_field("context", desc="The context."),
            "answer": output_field("answer", desc="The answer."),
            "confidence": output_field("confidence", type_=float, desc="The confidence."),
        },
        instructions="Given the fields `question`, `context`, produce the fields `answer`, `confidence`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=IncompleteDemoSignature,
        demos=[{"question": "Q1", "answer": "A1"}],
        inputs={"question": "Q2", "context": "C2"},
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\n2. `context` (str): The context.\nYour output fields are:\n1. `answer` (str): The answer.\n2. `confidence` (float): The confidence.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## context ## ]]\n{context}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## confidence ## ]]\n{confidence}        # note: the value you produce must be a single float value\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `context`, produce the fields `answer`, `confidence`.",
        },
        {
            "role": "user",
            "content": "This is an example of the task, though some input or output fields are not supplied.\n\n[[ ## question ## ]]\nQ1",
        },
        {
            "role": "assistant",
            "content": "[[ ## answer ## ]]\nA1\n\n[[ ## confidence ## ]]\nNot supplied for this particular example.\n\n[[ ## completed ## ]]\n",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ2\n\n[[ ## context ## ]]\nC2\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, then `[[ ## confidence ## ]]` (must be formatted as a valid Python float), and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_history():
    HistorySignature = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `turn_log`, `question`, produce the fields `answer`.",
    )
    history = TurnLog.model_validate(
        {
            "turns": [{"question": "What is 1+1?", "answer": "2"}, {"question": "What is 2+2?", "answer": "4"}],
        }
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=HistorySignature,
        demos=[],
        inputs={"turn_log": history, "question": "What is 3+3?"},
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `turn_log` (TurnLog): The history.\n2. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## turn_log ## ]]\n{turn_log}\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `turn_log`, `question`, produce the fields `answer`.",
        },
        {"role": "user", "content": "[[ ## question ## ]]\nWhat is 1+1?"},
        {"role": "assistant", "content": "[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n"},
        {"role": "user", "content": "[[ ## question ## ]]\nWhat is 2+2?"},
        {"role": "assistant", "content": "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]\n"},
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nWhat is 3+3?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_list_value_for_string_input():
    ListAsStringSignature = ts(
        "context -> answer", instructions="Given the fields `context`, produce the fields `answer`."
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(), task_spec=ListAsStringSignature, demos=[], inputs={"context": ["alpha", "beta"]}
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `context` (str): The context.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## context ## ]]\n{context}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `context`, produce the fields `answer`.",
        },
        {
            "role": "user",
            "content": "[[ ## context ## ]]\n[1] «alpha»\n[2] «beta»\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_literal_output():
    LiteralSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "verdict": output_field("verdict", type_=Literal["yes", "no"], desc="The verdict."),
        },
        instructions="Given the fields `question`, produce the fields `verdict`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(), task_spec=LiteralSignature, demos=[], inputs={"question": "Is the sky blue?"}
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `verdict` (Literal['yes', 'no']): The verdict.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## verdict ## ]]\n{verdict}        # note: the value you produce must exactly match (no extra characters) one of: yes; no\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `verdict`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nIs the sky blue?\n\nRespond with the corresponding output fields, starting with the field `[[ ## verdict ## ]]` (must be formatted as a valid Python Literal['yes', 'no']), and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_multimodal_custom_type_inputs():
    CustomTypeSignature = make_task_spec(
        {
            "image": input_field("image", type_=Image, desc="The image."),
            "audio": input_field("audio", type_=Audio, desc="The audio."),
            "file": input_field("file", type_=File, desc="The file."),
            "document": input_field("document", type_=Document, desc="The document."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `image`, `audio`, `file`, `document`, produce the fields `answer`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=CustomTypeSignature,
        demos=[],
        inputs={
            "image": Image("https://example.com/cat.png"),
            "audio": Audio(data="QUJD", audio_format="wav"),
            "file": File.from_file_id("file-123", filename="notes.txt"),
            "document": Document(data="Alpha beta", title="Doc"),
        },
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `image` (Image): The image.\n2. `audio` (Audio): The audio.\n3. `file` (File): The file.\n4. `document` (Document): The document.\n    Type description of Document: A document containing text content that can be referenced and cited. Include the full text content and optionally a title for proper referencing.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## image ## ]]\n{image}\n\n[[ ## audio ## ]]\n{audio}\n\n[[ ## file ## ]]\n{file}\n\n[[ ## document ## ]]\n{document}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `image`, `audio`, `file`, `document`, produce the fields `answer`.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[[ ## image ## ]]\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                {"type": "text", "text": "\n\n[[ ## audio ## ]]\n"},
                {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                {"type": "text", "text": "\n\n[[ ## file ## ]]\n"},
                {"type": "file", "file": {"file_id": "file-123", "filename": "notes.txt"}},
                {"type": "text", "text": "\n\n[[ ## document ## ]]\n"},
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": "Alpha beta"},
                    "citations": {"enabled": True},
                    "title": "Doc",
                },
                {
                    "type": "text",
                    "text": "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
                },
            ],
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_history_demo_pydantic_tools_and_image():

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
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "image": input_field("image", type_=Image, desc="The image."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "profile": input_field("profile", type_=Profile, desc="The profile."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", type_=AnswerCard, desc="The answer."),
        },
        instructions="Answer using all supplied context.",
    )
    tool = Tool(search, description="Search for documents.")
    demo_profile = Profile(name="Ada", location=Location(city="London", country="UK"), interests=["math", "machines"])
    current_profile = Profile(
        name="Grace", location=Location(city="Arlington", country="USA"), interests=["compilers", "navy"]
    )
    history = TurnLog.model_validate(
        {
            "turns": [
                {
                    "profile": demo_profile,
                    "question": "Who is Ada?",
                    "answer": AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
                }
            ],
        }
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
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
            "turn_log": history,
            "image": Image("https://example.com/current.png"),
            "tools": [tool],
            "profile": current_profile,
            "question": "What should the answer include?",
        },
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `turn_log` (TurnLog): The history.\n2. `image` (Image): The image.\n3. `tools` (list[Tool]): The tools.\n4. `profile` (Profile): The profile.\n5. `question` (str): The question.\nYour output fields are:\n1. `answer` (AnswerCard): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## turn_log ## ]]\n{turn_log}\n\n[[ ## image ## ]]\n{image}\n\n[[ ## tools ## ]]\n{tools}\n\n[[ ## profile ## ]]\n{profile}\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, "sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, "required": ["answer", "sources"], "title": "AnswerCard"}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer using all supplied context.',
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
            "content": '[[ ## answer ## ]]\n{"answer": "Mention analytical engines.", "sources": ["demo"]}\n\n[[ ## completed ## ]]\n',
        },
        {
            "role": "user",
            "content": '[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}\n\n[[ ## question ## ]]\nWho is Ada?',
        },
        {
            "role": "assistant",
            "content": '[[ ## answer ## ]]\n{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n\n[[ ## completed ## ]]\n',
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
                    "text": "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]` (must be formatted as a valid Python AnswerCard), and then ending with the marker for `[[ ## completed ## ]]`.",
                },
            ],
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_base_custom_type_input():

    class Event(FieldTypeMixin):
        label: str

        @override
        def format(self):
            return [{"type": "event", "event": {"label": self.label}}]

        @classmethod
        @override
        def description(cls) -> str:
            return "An event block."

    EventSignature = make_task_spec(
        {
            "event": input_field("event", type_=Event, desc="The event."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `event`, produce the fields `answer`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(), task_spec=EventSignature, demos=[], inputs={"event": Event(label="launch")}
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `event` (Event): The event.\n    Type description of Event: An event block.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## event ## ]]\n{event}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `event`, produce the fields `answer`.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[[ ## event ## ]]\n"},
                {"type": "event", "event": {"label": "launch"}},
                {
                    "type": "text",
                    "text": "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
                },
            ],
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_with_citations_output_demo():
    CitationSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "citations": output_field("citations", type_=Citations, desc="The citations."),
        },
        instructions="Given the fields `question`, produce the fields `citations`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=CitationSignature,
        demos=[
            {
                "question": "Q1",
                "citations": Citations.from_dict_list(
                    [{"cited_text": "alpha", "document_index": 0, "start_char_index": 0, "end_char_index": 5}]
                ),
            }
        ],
        inputs={"question": "Q2"},
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `citations` (Citations): The citations.\n    Type description of Citations: Citations with quoted text and source references. Include the exact text being cited and information about its source.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## citations ## ]]\n{citations}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "$defs": {"Citation": {"type": "object", "description": "Individual citation with character location information.", "properties": {"type": {"type": "string", "default": "char_location", "title": "Type"}, "cited_text": {"type": "string", "title": "Cited Text"}, "document_index": {"type": "integer", "title": "Document Index"}, "document_title": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Document Title"}, "end_char_index": {"type": "integer", "title": "End Char Index"}, "start_char_index": {"type": "integer", "title": "Start Char Index"}, "supported_text": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Supported Text"}}, "required": ["cited_text", "document_index", "start_char_index", "end_char_index"], "title": "Citation"}}, "description": "Experimental: This class may change or be removed in a future release without warning (introduced in v3.0.4).\\n\\nCitations extracted from an LM response with source references.\\n\\n    This type represents citations returned by language models that support\\n    citation extraction, particularly Anthropic\'s Citations API through LiteLLM.\\n    Citations include the quoted text and source information.\\n\\n    Examples:\\n        ```python\\n        import os\\n        from dspy.adapters.types.citation import Citations\\n        from dspy.adapters.types.document import Document\\n        from dspy.clients.lm import LM\\n        from dspy.predict.predict import Predict\\n        from dspy.signatures.field import InputField, OutputField\\n        from dspy.signatures.signature import Signature\\n\\n        os.environ[\\"ANTHROPIC_API_KEY\\"] = \\"YOUR_ANTHROPIC_API_KEY\\"\\n\\n        class AnswerWithSources(Signature):\\n            \'\'\'Answer questions using provided documents with citations.\'\'\'\\n            documents: list[Document] = InputField()\\n            question: str = InputField()\\n            answer: str = OutputField()\\n            citations: Citations = OutputField()\\n\\n        # Create documents to provide as sources\\n        docs = [\\n            Document(\\n                data=\\"The Earth orbits the Sun in an elliptical path.\\",\\n                title=\\"Basic Astronomy Facts\\"\\n            ),\\n            Document(\\n                data=\\"Water boils at 100°C at standard atmospheric pressure.\\",\\n                title=\\"Physics Fundamentals\\",\\n                metadata={\\"author\\": \\"Dr. Smith\\", \\"year\\": 2023}\\n            )\\n        ]\\n\\n        # Use with a model that supports citations like Claude\\n        lm = LM(\\"anthropic/claude-opus-4-1-20250805\\")\\n        predictor = Predict(AnswerWithSources)\\n        result = predictor(documents=docs, question=\\"What temperature does water boil?\\", lm=lm)\\n\\n        for citation in result.citations.citations:\\n            print(citation.format())\\n        ```\\n    ", "properties": {"citations": {"type": "array", "items": {"$ref": "#/$defs/Citation"}, "title": "Citations"}}, "required": ["citations"], "title": "Citations"}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `citations`.',
        },
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {
            "role": "assistant",
            "content": '[[ ## citations ## ]]\n[{"type": "char_location", "cited_text": "alpha", "document_index": 0, "start_char_index": 0, "end_char_index": 5}]\n\n[[ ## completed ## ]]\n',
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ2\n\nRespond with the corresponding output fields, starting with the field `[[ ## citations ## ]]` (must be formatted as a valid Python Citations), and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]

    def normalize_citations_schema_description(content):
        start = content.find('{"type": "object", "$defs":')
        if start == -1:
            return content
        prefix = content[:start]
        schema_part = content[start:]
        schema_part = re.sub(r'"description": "(?:[^"\\]|\\.)*"(?:, )?', "", schema_part)
        return prefix + schema_part

    messages[0]["content"] = normalize_citations_schema_description(messages[0]["content"])
    expected_messages[0]["content"] = normalize_citations_schema_description(expected_messages[0]["content"])
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_and_lm_kwargs_with_native_citations():
    from dspy.core.types.config import NativeAdaptationMode

    class AnthropicLM(DummyLM):
        def __init__(self):
            super().__init__([{}])
            self.model = "anthropic/claude-3-5-sonnet"

        @property
        def citations_adaptation_mode(self):
            return NativeAdaptationMode.SKIP

    CitationSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
            "citations": output_field("citations", type_=Citations, desc="The citations."),
        },
        instructions="Given the fields `question`, produce the fields `answer`, `citations`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(), task_spec=CitationSignature, demos=[], inputs={"question": "Q?"}, lm=AnthropicLM()
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`, `citations`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_preserves_passthrough_lm_kwargs():
    PassthroughSignature = ts(
        "question -> answer", instructions="Given the fields `question`, produce the fields `answer`."
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=PassthroughSignature,
        demos=[],
        inputs={"question": "Q?"},
        config={"temperature": 0.7, "max_tokens": 42, "extensions": {"stream": True}},
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {"temperature": 0.7, "max_tokens": 42, "stream": True}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_and_lm_kwargs_with_native_reasoning():

    class ReasoningLM(DummyLM):
        def __init__(self, answers):
            super().__init__(answers)
            self.kwargs["reasoning"] = {"effort": "low"}

        @property
        @override
        def supports_reasoning(self):
            return True

    NativeReasoningSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `answer`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=NativeReasoningSignature,
        demos=[],
        inputs={"question": "Q?"},
        lm=ReasoningLM([{}]),
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `reasoning`, `answer`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {"reasoning_effort": "low"}
    assert lm_kwargs == expected_lm_kwargs


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


def test_chat_adapter_format_exact_messages_with_reasoning_and_code_outputs():
    python_code = cast("Any", Code)["python"]
    CodeSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "code": output_field("code", type_=python_code, desc="The code."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `code`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=CodeSignature,
        demos=[{"question": "Q1", "reasoning": Reasoning(content="Think"), "code": python_code(code="print('hi')")}],
        inputs={"question": "Q2"},
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `reasoning` (Reasoning): The reasoning.\n2. `code` (Code_python): The code.\n    Type description of Code_python: Code represented in a string, specified in the `code` field. If this is an output field, the code field should follow the markdown code block format, e.g. \n```python\n{code}\n```\nProgramming language: python\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## code ## ]]\n{code}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `reasoning`, `code`.",
        },
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {
            "role": "assistant",
            "content": "[[ ## reasoning ## ]]\nThink\n\n[[ ## code ## ]]\nprint('hi')\n\n[[ ## completed ## ]]\n",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ2\n\nRespond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]` (must be formatted as a valid Python Reasoning), then `[[ ## code ## ]]` (must be formatted as a valid Python Code_python), and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_and_lm_kwargs_with_native_tool_calling():

    class FunctionCallingLM(DummyLM):
        @property
        @override
        def supports_function_calling(self):
            return True

    def search(query: str, k: int = 3) -> str:
        return query

    NativeToolSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=True),
        task_spec=NativeToolSignature,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
        lm=FunctionCallingLM([{}]),
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `tools`, produce the fields `tool_calls`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ?\n\nRespond with the corresponding output fields, starting with the field , and then ending with the marker for `[[ ## completed ## ]]`.",
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
                        "required": ["query"],
                    },
                },
            }
        ]
    }
    assert lm_kwargs == expected_lm_kwargs


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
                {
                    "question": "Q1",
                    "next_thought": Reasoning(content="I should search."),
                    "tool_calls": ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                }
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
                {
                    "question": "Q1",
                    "next_thought": Reasoning(content="I should search twice."),
                    "tool_calls": tool_calls.model_copy(update={"tool_call_results": tool_call_results}),
                }
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
            "turns": [{"tool_calls": ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results)}],
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
                {"question": "Q1", "next_thought": Reasoning(content="I should search."), "tool_calls": tool_calls}
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


def test_chat_adapter_format_exact_messages_with_non_native_tool_history():

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
    history = TurnLog.model_validate(
        {
            "turns": [
                {
                    "question": "Q1",
                    "next_thought": "I should search.",
                    "tool_calls": ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                }
            ],
        }
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(use_native_function_calling=False),
        task_spec=ToolHistorySignature,
        demos=[],
        inputs={"question": "Q2", "turn_log": history, "tools": [Tool(search, description="Search for documents.")]},
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `question` (str): The question.\n2. `turn_log` (TurnLog): The history.\n3. `tools` (list[Tool]): The tools.\nYour output fields are:\n1. `next_thought` (str): The next thought.\n2. `tool_calls` (ToolCalls): The tool calls.\n    Type description of ToolCalls: Tool calls must be a JSON object with `tool_calls`, a list of calls. Each call must include `name` and `args`. Example: {"tool_calls": [{"name": "search", "args": {"query": "cats"}}]}\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## turn_log ## ]]\n{turn_log}\n\n[[ ## tools ## ]]\n{tools}\n\n[[ ## next_thought ## ]]\n{next_thought}\n\n[[ ## tool_calls ## ]]\n{tool_calls}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "$defs": {"ToolCall": {"type": "object", "properties": {"args": {"type": "object", "additionalProperties": true, "title": "Args"}, "name": {"type": "string", "title": "Name"}}, "required": ["name", "args"], "title": "ToolCall"}}, "properties": {"tool_calls": {"type": "array", "items": {"$ref": "#/$defs/ToolCall"}, "title": "Tool Calls"}}, "required": ["tool_calls"], "title": "ToolCalls"}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `turn_log`, `tools`, produce the fields `next_thought`, `tool_calls`.',
        },
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {
            "role": "assistant",
            "content": '[[ ## next_thought ## ]]\nI should search.\n\n[[ ## tool_calls ## ]]\n{"tool_calls": [{"name": "search", "args": {"query": "cats"}, "id": "call_1"}]}\n\n[[ ## completed ## ]]\n',
        },
        {
            "role": "user",
            "content": '[[ ## tool_call_results ## ]]\n{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}',
        },
        {
            "role": "user",
            "content": '[[ ## question ## ]]\nQ2\n\n[[ ## tools ## ]]\n["search, whose description is <desc>Search for documents.</desc>. It takes arguments {\'query\': {\'type\': \'string\'}}."]\n\nRespond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then `[[ ## tool_calls ## ]]` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}), and then ending with the marker for `[[ ## completed ## ]]`.',
        },
    ]
    assert messages == expected_messages
    assert lm_kwargs == {}


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
                        {
                            "question": "Q1",
                            "next_thought": "I should search.",
                            "tool_calls": ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                        }
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
                    "turns": [{"question": "Q1"}],
                }
            ),
        },
    )
    assert messages[1] == {"role": "user", "content": "custom history"}
    assert messages[2]["role"] == "user"
    assert "Q2" in messages[2]["content"]


def test_chat_adapter_format_exact_messages_with_tool_input():

    def search(query: str, k: int = 3) -> str:
        return query

    ToolSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=ToolSignature,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
    )
    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n1. `question` (str): The question.\n2. `tools` (list[Tool]): The tools.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## tools ## ]]\n{tools}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `tools`, produce the fields `answer`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\nQ?\n\n[[ ## tools ## ]]\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_chat_adapter_format_exact_messages_kitchen_sink():

    def search(query: str, k: int = 3) -> str:
        return query

    class Event(FieldTypeMixin):
        label: str

        @override
        def format(self):
            return [{"type": "event", "event": {"label": self.label}}]

        @classmethod
        @override
        def description(cls) -> str:
            return "An event block."

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

    KitchenSinkSignature = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "image": input_field("image", type_=Image, desc="The image."),
            "audio": input_field("audio", type_=Audio, desc="The audio."),
            "file": input_field("file", type_=File, desc="The file."),
            "document": input_field("document", type_=Document, desc="The document."),
            "event": input_field("event", type_=Event, desc="The event."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "profile": input_field("profile", type_=Profile, desc="The profile."),
            "context": input_field("context", desc="The context."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", type_=AnswerCard, desc="The answer."),
            "verdict": output_field("verdict", type_=Literal["yes", "no"], desc="The verdict."),
            "confidence": output_field("confidence", type_=float, desc="The confidence."),
        },
        instructions="Answer carefully using every available signal.",
    )
    tool = Tool(search, description="Search for documents.")
    demo_profile = Profile(name="Ada", location=Location(city="London", country="UK"), interests=["math", "machines"])
    current_profile = Profile(
        name="Grace", location=Location(city="Arlington", country="USA"), interests=["compilers", "navy"]
    )
    history = TurnLog.model_validate(
        {
            "turns": [
                {
                    "profile": demo_profile,
                    "context": ["old note", "older note"],
                    "question": "Who is Ada?",
                    "answer": AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
                    "verdict": "yes",
                    "confidence": 0.8,
                }
            ],
        }
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=ChatAdapter(),
        task_spec=KitchenSinkSignature,
        demos=[
            {
                "image": Image("https://example.com/demo.png"),
                "audio": Audio(data="REVNTw==", audio_format="wav"),
                "file": File.from_file_id("file-demo", filename="demo.txt"),
                "document": Document(data="Demo document", title="Demo Doc"),
                "event": Event(label="demo-event"),
                "tools": [tool],
                "profile": demo_profile,
                "context": ["demo context one", "demo context two"],
                "question": "What should we mention?",
                "answer": AnswerCard(answer="Mention analytical engines.", sources=["demo"]),
                "verdict": "yes",
                "confidence": 0.9,
            },
            {
                "question": "Incomplete example question",
                "answer": AnswerCard(answer="Partial answer.", sources=["partial"]),
            },
        ],
        inputs={
            "turn_log": history,
            "image": Image("https://example.com/current.png"),
            "audio": Audio(data="Q1VSUkVOVA==", audio_format="wav"),
            "file": File.from_file_id("file-current", filename="current.txt"),
            "document": Document(data="Current document", title="Current Doc"),
            "event": Event(label="current-event"),
            "tools": [tool],
            "profile": current_profile,
            "context": ["current context one", "current context two"],
            "question": "What should the answer include?",
        },
    )
    expected_messages = [
        {
            "role": "system",
            "content": 'Your input fields are:\n1. `turn_log` (TurnLog): The history.\n2. `image` (Image): The image.\n3. `audio` (Audio): The audio.\n4. `file` (File): The file.\n5. `document` (Document): The document.\n    Type description of Document: A document containing text content that can be referenced and cited. Include the full text content and optionally a title for proper referencing.\n6. `event` (Event): The event.\n    Type description of Event: An event block.\n7. `tools` (list[Tool]): The tools.\n8. `profile` (Profile): The profile.\n9. `context` (str): The context.\n10. `question` (str): The question.\nYour output fields are:\n1. `answer` (AnswerCard): The answer.\n2. `verdict` (Literal[\'yes\', \'no\']): The verdict.\n3. `confidence` (float): The confidence.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## turn_log ## ]]\n{turn_log}\n\n[[ ## image ## ]]\n{image}\n\n[[ ## audio ## ]]\n{audio}\n\n[[ ## file ## ]]\n{file}\n\n[[ ## document ## ]]\n{document}\n\n[[ ## event ## ]]\n{event}\n\n[[ ## tools ## ]]\n{tools}\n\n[[ ## profile ## ]]\n{profile}\n\n[[ ## context ## ]]\n{context}\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, "sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, "required": ["answer", "sources"], "title": "AnswerCard"}\n\n[[ ## verdict ## ]]\n{verdict}        # note: the value you produce must exactly match (no extra characters) one of: yes; no\n\n[[ ## confidence ## ]]\n{confidence}        # note: the value you produce must be a single float value\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer carefully using every available signal.',
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
                {"type": "text", "text": "\n\n[[ ## audio ## ]]\n"},
                {"type": "input_audio", "input_audio": {"data": "REVNTw==", "format": "wav"}},
                {"type": "text", "text": "\n\n[[ ## file ## ]]\n"},
                {"type": "file", "file": {"file_id": "file-demo", "filename": "demo.txt"}},
                {"type": "text", "text": "\n\n[[ ## document ## ]]\n"},
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": "Demo document"},
                    "citations": {"enabled": True},
                    "title": "Demo Doc",
                },
                {"type": "text", "text": "\n\n[[ ## event ## ]]\n"},
                {"type": "event", "event": {"label": "demo-event"}},
                {
                    "type": "text",
                    "text": "\n\n[[ ## tools ## ]]\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]",
                },
                {
                    "type": "text",
                    "text": '\n\n[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}',
                },
                {"type": "text", "text": "\n\n[[ ## context ## ]]\n[1] «demo context one»\n[2] «demo context two»"},
                {"type": "text", "text": "\n\n[[ ## question ## ]]\nWhat should we mention?"},
            ],
        },
        {
            "role": "assistant",
            "content": '[[ ## answer ## ]]\n{"answer": "Mention analytical engines.", "sources": ["demo"]}\n\n[[ ## verdict ## ]]\nyes\n\n[[ ## confidence ## ]]\n0.9\n\n[[ ## completed ## ]]\n',
        },
        {
            "role": "user",
            "content": "This is an example of the task, though some input or output fields are not supplied.\n\n[[ ## question ## ]]\nIncomplete example question",
        },
        {
            "role": "assistant",
            "content": '[[ ## answer ## ]]\n{"answer": "Partial answer.", "sources": ["partial"]}\n\n[[ ## verdict ## ]]\nNot supplied for this particular example. \n\n[[ ## confidence ## ]]\nNot supplied for this particular example.\n\n[[ ## completed ## ]]\n',
        },
        {
            "role": "user",
            "content": '[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}\n\n[[ ## context ## ]]\n[1] «old note»\n[2] «older note»\n\n[[ ## question ## ]]\nWho is Ada?',
        },
        {
            "role": "assistant",
            "content": '[[ ## answer ## ]]\n{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n\n[[ ## verdict ## ]]\nyes\n\n[[ ## confidence ## ]]\n0.8\n\n[[ ## completed ## ]]\n',
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[[ ## image ## ]]\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                {"type": "text", "text": "\n\n[[ ## audio ## ]]\n"},
                {"type": "input_audio", "input_audio": {"data": "Q1VSUkVOVA==", "format": "wav"}},
                {"type": "text", "text": "\n\n[[ ## file ## ]]\n"},
                {"type": "file", "file": {"file_id": "file-current", "filename": "current.txt"}},
                {"type": "text", "text": "\n\n[[ ## document ## ]]\n"},
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": "Current document"},
                    "citations": {"enabled": True},
                    "title": "Current Doc",
                },
                {"type": "text", "text": "\n\n[[ ## event ## ]]\n"},
                {"type": "event", "event": {"label": "current-event"}},
                {
                    "type": "text",
                    "text": "\n\n[[ ## tools ## ]]\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]",
                },
                {
                    "type": "text",
                    "text": '\n\n[[ ## profile ## ]]\n{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, "interests": ["compilers", "navy"]}',
                },
                {
                    "type": "text",
                    "text": "\n\n[[ ## context ## ]]\n[1] «current context one»\n[2] «current context two»",
                },
                {"type": "text", "text": "\n\n[[ ## question ## ]]\nWhat should the answer include?"},
                {
                    "type": "text",
                    "text": "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]` (must be formatted as a valid Python AnswerCard), then `[[ ## verdict ## ]]` (must be formatted as a valid Python Literal['yes', 'no']), then `[[ ## confidence ## ]]` (must be formatted as a valid Python float), and then ending with the marker for `[[ ## completed ## ]]`.",
                },
            ],
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


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


def test_chat_adapter_formats_image():
    image = Image(url="https://example.com/image.jpg")
    MySignature = make_task_spec(
        {"image": input_field("image", type_=Image, desc="The image."), "text": output_field("text", desc="The text.")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(adapter=adapter, task_spec=MySignature, demos=[], inputs={"image": image})
    assert len(messages) == 2
    user_message_content = messages[1]["content"]
    assert user_message_content is not None
    assert len(user_message_content) == 3
    assert user_message_content[0]["type"] == "text"
    assert user_message_content[2]["type"] == "text"
    expected_image_content = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    assert expected_image_content in user_message_content


def test_chat_adapter_formats_image_with_few_shot_examples():
    MySignature = make_task_spec(
        {"image": input_field("image", type_=Image, desc="The image."), "text": output_field("text", desc="The text.")},
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    adapter = ChatAdapter()
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
    assert "[[ ## completed ## ]]\n" in messages[2]["content"]
    assert "[[ ## completed ## ]]\n" in messages[4]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in messages[1]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in messages[3]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}} in messages[5]["content"]


def test_chat_adapter_formats_image_with_nested_images():

    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    MySignature = make_task_spec(
        {
            "image": input_field("image", type_=ImageWrapper, desc="The image."),
            "text": output_field("text", desc="The text."),
        },
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    image1 = Image(url="https://example.com/image1.jpg")
    image2 = Image(url="https://example.com/image2.jpg")
    image3 = Image(url="https://example.com/image3.jpg")
    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    adapter = ChatAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=MySignature, demos=[], inputs={"image": image_wrapper}
    )
    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]


def test_chat_adapter_formats_image_with_few_shot_examples_with_nested_images():

    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    MySignature = make_task_spec(
        {
            "image": input_field("image", type_=ImageWrapper, desc="The image."),
            "text": output_field("text", desc="The text."),
        },
        instructions="Given the fields `image`, produce the fields `text`.",
    )
    image1 = Image(url="https://example.com/image1.jpg")
    image2 = Image(url="https://example.com/image2.jpg")
    image3 = Image(url="https://example.com/image3.jpg")
    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    demos = [Example.from_record({"image": image_wrapper, "text": "This is a test image"})]
    image_wrapper_2 = ImageWrapper(images=[Image(url="https://example.com/image4.jpg")], tag=["test", "example"])
    adapter = ChatAdapter()
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
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
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
