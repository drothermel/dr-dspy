import asyncio
import sys
from unittest import mock

import pydantic
import pytest

from dspy.utils.exceptions import AdapterParseError

try:
    from litellm import Choices, Message, ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]

from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.types.code import Code
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.clients.lm import LM
from dspy.primitives.example import Example
from dspy.task_spec import FieldSpec, make_task_spec
from dspy.task_spec.pydantic_bridge import task_spec_output_field_infos
from tests.adapters.conftest import adapter_format_as_openai, format_messages_and_lm_kwargs
from tests.task_spec.helpers import ts


def test_xml_adapter_format_and_parse_basic():
    TestSignature = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    adapter = XMLAdapter()
    # Format output fields as XML
    fields_with_values = {
        FieldInfoWithName(name="answer", info=task_spec_output_field_infos(TestSignature)["answer"]): "Paris"
    }
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip() == "<answer>\nParis\n</answer>"

    # Parse XML output
    completion = "<answer>Paris</answer>"
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert parsed == {"answer": "Paris"}


def test_xml_adapter_parse_multiple_fields():
    TestSignature = ts(
        "question -> answer, explanation",
        instructions="Given the fields `question`, produce the fields `answer`, `explanation`.",
    )
    adapter = XMLAdapter()
    completion = """
<answer>Paris</answer>
<explanation>The capital of France is Paris.</explanation>
"""
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert parsed == {"answer": "Paris", "explanation": "The capital of France is Paris."}


def test_xml_adapter_parse_raises_on_missing_field():
    TestSignature = ts(
        "question -> answer, explanation",
        instructions="Given the fields `question`, produce the fields `answer`, `explanation`.",
    )
    adapter = XMLAdapter()
    completion = "<answer>Paris</answer>"
    with pytest.raises(AdapterParseError) as e:
        adapter.parse(task_spec=TestSignature, completion=completion)
    assert e.value.adapter_name == "XMLAdapter"
    assert e.value.task_spec == TestSignature
    assert e.value.lm_response == "<answer>Paris</answer>"
    assert "explanation" in str(e.value)


def test_xml_adapter_parse_casts_types():
    TestSignature = make_task_spec(
        {
            "number": FieldSpec.output("number", type_=int),
            "flag": FieldSpec.output("flag", type_=bool),
        },
        instructions="Given the fields , produce the fields `number`, `flag`.",
    )
    adapter = XMLAdapter()
    completion = """
<number>42</number>
<flag>true</flag>
"""
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert parsed == {"number": 42, "flag": True}


def test_xml_adapter_parse_raises_on_type_error():
    TestSignature = make_task_spec(
        {
            "number": FieldSpec.output("number", type_=int),
        },
        instructions="Given the fields , produce the fields `number`.",
    )
    adapter = XMLAdapter()
    completion = "<number>not_a_number</number>"
    with pytest.raises(AdapterParseError) as e:
        adapter.parse(task_spec=TestSignature, completion=completion)
    assert "Failed to parse field" in str(e.value)


def test_xml_adapter_format_and_parse_nested_model():
    class InnerModel(pydantic.BaseModel):
        value: int
        label: str

    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "result": FieldSpec.output("result", type_=InnerModel),
        },
        instructions="Given the fields `question`, produce the fields `result`.",
    )
    adapter = XMLAdapter()
    # Format output fields as XML
    fields_with_values = {
        FieldInfoWithName(name="result", info=task_spec_output_field_infos(TestSignature)["result"]): InnerModel(
            value=5, label="foo"
        )
    }
    xml = adapter.format_field_with_value(fields_with_values)
    # The output will be a JSON string inside the XML tag
    assert xml.strip().startswith("<result>")
    assert '"value": 5' in xml
    assert '"label": "foo"' in xml
    assert xml.strip().endswith("</result>")

    # Parse XML output (should parse as string, not as model)
    completion = '<result>{"value": 5, "label": "foo"}</result>'
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    # The parse_value helper will try to cast to InnerModel
    assert isinstance(parsed["result"], InnerModel)
    assert parsed["result"].value == 5
    assert parsed["result"].label == "foo"


def test_xml_adapter_format_and_parse_list_of_models():
    class Item(pydantic.BaseModel):
        name: str
        score: float

    TestSignature = make_task_spec(
        {
            "items": FieldSpec.output("items", type_=list[Item]),
        },
        instructions="Given the fields , produce the fields `items`.",
    )
    adapter = XMLAdapter()
    items = [Item(name="a", score=1.1), Item(name="b", score=2.2)]
    fields_with_values = {
        FieldInfoWithName(name="items", info=task_spec_output_field_infos(TestSignature)["items"]): items
    }
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip().startswith("<items>")
    assert '"name": "a"' in xml
    assert '"score": 2.2' in xml
    assert xml.strip().endswith("</items>")

    # Parse XML output
    import json

    completion = f"<items>{json.dumps([i.model_dump() for i in items])}</items>"
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert isinstance(parsed["items"], list)
    assert all(isinstance(i, Item) for i in parsed["items"])
    assert parsed["items"][0].name == "a"
    assert parsed["items"][1].score == 2.2


def test_xml_adapter_with_tool_like_output():
    # XMLAdapter does not natively support tool calls, but we can test structured output
    class ToolCall(pydantic.BaseModel):
        name: str
        args: dict
        result: str

    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "tool_calls": FieldSpec.output("tool_calls", type_=list[ToolCall]),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `question`, produce the fields `tool_calls`, `answer`.",
    )
    adapter = XMLAdapter()
    tool_calls = [
        ToolCall(name="get_weather", args={"city": "Tokyo"}, result="Sunny"),
        ToolCall(name="get_population", args={"country": "Japan", "year": 2023}, result="125M"),
    ]
    fields_with_values = {
        FieldInfoWithName(
            name="tool_calls", info=task_spec_output_field_infos(TestSignature)["tool_calls"]
        ): tool_calls,
        FieldInfoWithName(
            name="answer", info=TestSignature.output_fields["answer"]
        ): "The weather is Sunny. Population is 125M.",
    }
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip().startswith("<tool_calls>")
    assert '"name": "get_weather"' in xml
    assert '"result": "125M"' in xml
    assert xml.strip().endswith("</answer>")

    import json

    completion = (
        f"<tool_calls>{json.dumps([tc.model_dump() for tc in tool_calls])}</tool_calls>"
        f"\n<answer>The weather is Sunny. Population is 125M.</answer>"
    )
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert isinstance(parsed["tool_calls"], list)
    assert parsed["tool_calls"][0].name == "get_weather"
    assert parsed["tool_calls"][1].result == "125M"
    assert parsed["answer"] == "The weather is Sunny. Population is 125M."


def test_xml_adapter_formats_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[Image]
        tag: list[str]

    MySignature = make_task_spec(
        {
            "image": FieldSpec.input("image", type_=ImageWrapper),
            "text": FieldSpec.output("text"),
        },
        instructions="Given the fields `image`, produce the fields `text`.",
    )
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
    adapter = XMLAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=MySignature, demos=demos, inputs={"image": image_wrapper_2}
    )

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


def test_xml_adapter_with_code():
    # Test with code as input field
    CodeAnalysis = make_task_spec(
        {
            "code": FieldSpec.input("code", type_=Code),
            "result": FieldSpec.output("result"),
        },
        instructions="Analyze the time complexity of the code",
    )
    adapter = XMLAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=CodeAnalysis, demos=[], inputs={"code": "print('Hello, world!')"}
    )

    assert len(messages) == 2

    # The output field type description should be included in the system message even if the output field is nested
    assert Code.description() in messages[0]["content"]

    # The user message should include the question and the tools
    assert "print('Hello, world!')" in messages[1]["content"]

    # Test with code as output field
    CodeGeneration = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "code": FieldSpec.output("code", type_=Code),
        },
        instructions="Generate code to answer the question",
    )
    adapter = XMLAdapter()
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='<code>print("Hello, world!")</code>'))],
            model="openai/gpt-4o-mini",
        )
        result = asyncio.run(
            adapter.acall(
                lm=LM(model="openai/gpt-4o-mini", cache=False),
                config={},
                task_spec=CodeGeneration,
                demos=[],
                inputs={"question": "Write a python program to print 'Hello, world!'"},
            )
        )
        assert result[0]["code"].code == 'print("Hello, world!")'


def test_xml_adapter_full_prompt():
    QA = make_task_spec(
        {
            "query": FieldSpec.input("query"),
            "context": FieldSpec.input("context", type_=str | None),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `query`, `context`, produce the fields `answer`.",
    )
    adapter = XMLAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=QA, demos=[], inputs={"query": "when was Marie Curie born"}
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    union_type_repr = "Union[str, NoneType]" if sys.version_info >= (3, 14) else "UnionType[str, NoneType]"

    expected_system = (
        "Your input fields are:\n"
        "1. `query` (str): \n"
        f"2. `context` ({union_type_repr}):\n"
        "Your output fields are:\n"
        "1. `answer` (str):\n"
        "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        "<query>\n{query}\n</query>\n\n"
        "<context>\n{context}\n</context>\n\n"
        "<answer>\n{answer}\n</answer>\n"
        "In adhering to this structure, your objective is: \n"
        "        Given the fields `query`, `context`, produce the fields `answer`."
    )

    expected_user = (
        "<query>\nwhen was Marie Curie born\n</query>\n\n"
        "Respond with the corresponding output fields wrapped in XML tags `<answer>`."
    )

    assert messages[0]["content"] == expected_system
    assert messages[1]["content"] == expected_user


def test_xml_adapter_format_exact_messages_for_simple_signature():
    StringSignature = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=XMLAdapter(),
        task_spec=StringSignature,
        demos=[],
        inputs={"question": "why did a chicken cross the kitchen?"},
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

<question>
{question}
</question>

<answer>
{answer}
</answer>
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`.""",
        },
        {
            "role": "user",
            "content": """<question>
why did a chicken cross the kitchen?
</question>

Respond with the corresponding output fields wrapped in XML tags `<answer>`.""",
        },
    ]


def test_xml_adapter_format_exact_non_native_tool_result_history_field():
    def search(query: str) -> str:
        return query

    ToolHistorySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "history": FieldSpec.input("history", type_=History),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "next_thought": FieldSpec.output("next_thought"),
            "tool_calls": FieldSpec.output("tool_calls", type_=ToolCalls),
        },
        instructions="Given the fields `question`, `history`, `tools`, produce the fields `next_thought`, `tool_calls`.",
    )
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = ToolCallResults.from_tool_calls_and_values([tool_call], ["cat"])

    messages, _lm_kwargs = format_messages_and_lm_kwargs(
        adapter=XMLAdapter(use_native_function_calling=False),
        task_spec=ToolHistorySignature,
        demos=[],
        inputs={
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
        "<tool_call_results>\n"
        '{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}\n'
        "</tool_call_results>"
    )
    assert messages[4]["content"] == (
        "<question>\n"
        "Q2\n"
        "</question>\n"
        "\n"
        "<tools>\n"
        "[\"search. It takes arguments {'query': {'type': 'string'}}.\"]\n"
        "</tools>\n"
        "\n"
        "Respond with the corresponding output fields wrapped in XML tags `<next_thought>`, then `<tool_calls>`."
    )


def test_xml_adapter_format_exact_messages_for_two_input_signature():
    StringSignature = ts(
        "question, answer -> judgement",
        instructions="Given the fields `question`, `answer`, produce the fields `judgement`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=XMLAdapter(),
        task_spec=StringSignature,
        demos=[],
        inputs={"question": "why did a chicken cross the kitchen?", "answer": "To get to the other side!"},
    )

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):\x20
2. `answer` (str):
Your output fields are:
1. `judgement` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<judgement>
{judgement}
</judgement>
In adhering to this structure, your objective is:\x20
        Given the fields `question`, `answer`, produce the fields `judgement`.""",
        },
        {
            "role": "user",
            "content": """<question>
why did a chicken cross the kitchen?
</question>

<answer>
To get to the other side!
</answer>

Respond with the corresponding output fields wrapped in XML tags `<judgement>`.""",
        },
    ]


def test_xml_adapter_format_exact_messages_with_demo_and_typed_output():
    MultiAnswer = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "answer": FieldSpec.output("answer"),
            "score": FieldSpec.output("score", type_=float),
        },
        instructions="Given the fields `question`, produce the fields `answer`, `score`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=XMLAdapter(),
        task_spec=MultiAnswer,
        demos=[{"question": "Q1", "answer": "A1", "score": 0.9}],
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
2. `score` (float):
All interactions will be structured in the following way, with the appropriate values filled in.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<score>
{score}        # note: the value you produce must be a single float value
</score>
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`, `score`.""",
        },
        {
            "role": "user",
            "content": """<question>
Q1
</question>""",
        },
        {
            "role": "assistant",
            "content": """<answer>
A1
</answer>

<score>
0.9
</score>""",
        },
        {
            "role": "user",
            "content": """<question>
Q2
</question>

Respond with the corresponding output fields wrapped in XML tags `<answer>`, then `<score>`.""",
        },
    ]


def test_xml_adapter_format_exact_messages_with_history_demo_pydantic_tools_and_image():
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

    RichRenderingSignature = make_task_spec(
        {
            "history": FieldSpec.input("history", type_=History),
            "image": FieldSpec.input("image", type_=Image),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "profile": FieldSpec.input("profile", type_=Profile),
            "question": FieldSpec.input("question"),
            "answer": FieldSpec.output("answer", type_=AnswerCard),
        },
        instructions="Answer using all supplied context.",
    )
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
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=XMLAdapter(),
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
            "content": "Your input fields are:\n"
            "1. `history` (History): \n"
            "2. `image` (Image): \n"
            "3. `tools` (list[Tool]): \n"
            "4. `profile` (Profile): \n"
            "5. `question` (str):\n"
            "Your output fields are:\n"
            "1. `answer` (AnswerCard):\n"
            "All interactions will be structured in the following way, with the appropriate "
            "values filled in.\n"
            "\n"
            "<history>\n"
            "{history}\n"
            "</history>\n"
            "\n"
            "<image>\n"
            "{image}\n"
            "</image>\n"
            "\n"
            "<tools>\n"
            "{tools}\n"
            "</tools>\n"
            "\n"
            "<profile>\n"
            "{profile}\n"
            "</profile>\n"
            "\n"
            "<question>\n"
            "{question}\n"
            "</question>\n"
            "\n"
            "<answer>\n"
            "{answer}        # note: the value you produce must adhere to the JSON schema: "
            '{"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, '
            '"sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, '
            '"required": ["answer", "sources"], "title": "AnswerCard"}\n'
            "</answer>\n"
            "In adhering to this structure, your objective is: \n"
            "        Answer using all supplied context.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is an example of the task, though some input or output fields are not supplied.",
                },
                {"type": "text", "text": "<image>\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                {"type": "text", "text": "\n</image>"},
                {
                    "type": "text",
                    "text": "\n\n<tools>\n"
                    '["search, whose description is <desc>Search for documents.</desc>. It '
                    "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                    "'default': 3}}.\"]\n"
                    "</tools>",
                },
                {
                    "type": "text",
                    "text": "\n\n<profile>\n"
                    '{"name": "Ada", "location": {"city": "London", "country": "UK"}, '
                    '"interests": ["math", "machines"]}\n'
                    "</profile>",
                },
                {"type": "text", "text": "\n\n<question>\nWhat should we mention?\n</question>"},
            ],
        },
        {
            "role": "assistant",
            "content": '<answer>\n{"answer": "Mention analytical engines.", "sources": ["demo"]}\n</answer>',
        },
        {
            "role": "user",
            "content": "<profile>\n"
            '{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": '
            '["math", "machines"]}\n'
            "</profile>\n"
            "\n"
            "<question>\n"
            "Who is Ada?\n"
            "</question>",
        },
        {
            "role": "assistant",
            "content": '<answer>\n{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n</answer>',
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\n"},
                {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                {"type": "text", "text": "\n</image>"},
                {
                    "type": "text",
                    "text": "\n\n<tools>\n"
                    '["search, whose description is <desc>Search for documents.</desc>. It '
                    "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                    "'default': 3}}.\"]\n"
                    "</tools>",
                },
                {
                    "type": "text",
                    "text": "\n\n<profile>\n"
                    '{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, '
                    '"interests": ["compilers", "navy"]}\n'
                    "</profile>",
                },
                {
                    "type": "text",
                    "text": "\n\n<question>\nWhat should the answer include?\n</question>",
                },
                {
                    "type": "text",
                    "text": "\n\nRespond with the corresponding output fields wrapped in XML tags `<answer>`.",
                },
            ],
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_xml_adapter_format_exact_messages_with_nested_pydantic_output():
    class XmlAddress(pydantic.BaseModel):
        city: str
        country: str

    class XmlSummary(pydantic.BaseModel):
        title: str
        address: XmlAddress

    PydanticSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "summary": FieldSpec.output("summary", type_=XmlSummary),
        },
        instructions="Given the fields `question`, produce the fields `summary`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=XMLAdapter(), task_spec=PydanticSignature, demos=[], inputs={"question": "Summarize"}
    )

    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n"
            "1. `question` (str):\n"
            "Your output fields are:\n"
            "1. `summary` (XmlSummary):\n"
            "All interactions will be structured in the following way, with the appropriate "
            "values filled in.\n"
            "\n"
            "<question>\n"
            "{question}\n"
            "</question>\n"
            "\n"
            "<summary>\n"
            "{summary}        # note: the value you produce must adhere to the JSON schema: "
            '{"type": "object", "$defs": {"XmlAddress": {"type": "object", "properties": {"city": '
            '{"type": "string", "title": "City"}, "country": {"type": "string", "title": '
            '"Country"}}, "required": ["city", "country"], "title": "XmlAddress"}}, "properties": '
            '{"address": {"$ref": "#/$defs/XmlAddress"}, "title": {"type": "string", "title": '
            '"Title"}}, "required": ["title", "address"], "title": "XmlSummary"}\n'
            "</summary>\n"
            "In adhering to this structure, your objective is: \n"
            "        Given the fields `question`, produce the fields `summary`.",
        },
        {
            "role": "user",
            "content": "<question>\n"
            "Summarize\n"
            "</question>\n"
            "\n"
            "Respond with the corresponding output fields wrapped in XML tags `<summary>`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_xml_adapter_format_exact_messages_with_incomplete_demo():
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
        adapter=XMLAdapter(),
        task_spec=IncompleteDemoSignature,
        demos=[{"question": "Q1", "answer": "A1"}],
        inputs={"question": "Q2", "context": "C2"},
    )

    expected_messages = [
        {
            "role": "system",
            "content": "Your input fields are:\n"
            "1. `question` (str): \n"
            "2. `context` (str):\n"
            "Your output fields are:\n"
            "1. `answer` (str): \n"
            "2. `score` (float):\n"
            "All interactions will be structured in the following way, with the appropriate "
            "values filled in.\n"
            "\n"
            "<question>\n"
            "{question}\n"
            "</question>\n"
            "\n"
            "<context>\n"
            "{context}\n"
            "</context>\n"
            "\n"
            "<answer>\n"
            "{answer}\n"
            "</answer>\n"
            "\n"
            "<score>\n"
            "{score}        # note: the value you produce must be a single float value\n"
            "</score>\n"
            "In adhering to this structure, your objective is: \n"
            "        Given the fields `question`, `context`, produce the fields `answer`, "
            "`score`.",
        },
        {
            "role": "user",
            "content": "This is an example of the task, though some input or output fields are not "
            "supplied.\n"
            "\n"
            "<question>\n"
            "Q1\n"
            "</question>",
        },
        {
            "role": "assistant",
            "content": "<answer>\nA1\n</answer>\n\n<score>\nNot supplied for this particular example. \n</score>",
        },
        {
            "role": "user",
            "content": "<question>\n"
            "Q2\n"
            "</question>\n"
            "\n"
            "<context>\n"
            "C2\n"
            "</context>\n"
            "\n"
            "Respond with the corresponding output fields wrapped in XML tags `<answer>`, then "
            "`<score>`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_format_system_message():
    MySignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "answers": FieldSpec.output("answers", type_=list[str]),
            "scores": FieldSpec.output("scores", type_=list[float]),
        },
        instructions="Answer the question with multiple answers and scores",
    )
    adapter = XMLAdapter()
    system_message = adapter.format_system_message(MySignature)

    expected_system_message = """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answers` (list[str]):\x20
2. `scores` (list[float]):
All interactions will be structured in the following way, with the appropriate values filled in.

<question>
{question}
</question>

<answers>
{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}
</answers>

<scores>
{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}
</scores>
In adhering to this structure, your objective is:\x20
        Answer the question with multiple answers and scores"""
    assert system_message == expected_system_message
