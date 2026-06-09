import asyncio
import sys
from unittest import mock

import pydantic
import pytest

from dspy.errors import AdapterParseError

try:
    from litellm import Choices, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.adapters.types.code import Code
from dspy.adapters.types.image import Image
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.clients.lm import LM
from dspy.primitives import Example
from dspy.task_spec import FieldBinding, input_field, make_task_spec, output_field
from tests.adapters.conftest import adapter_format_as_openai, make_adapter_run
from tests.task_spec.helpers import ts


def test_xml_adapter_format_and_parse_basic():
    TestSignature = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    adapter = XMLAdapter()
    fields_with_values = {FieldBinding(name="answer", field=TestSignature.output_fields["answer"]): "Paris"}
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip() == "<answer>\nParis\n</answer>"
    completion = "<answer>Paris</answer>"
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert parsed == {"answer": "Paris"}


def test_xml_adapter_parse_hyphenated_field_name():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="q"),
            "my-answer": output_field("my-answer", desc="a"),
        },
        instructions="answer",
    )
    adapter = XMLAdapter()
    completion = "<my-answer>Paris</my-answer>"
    parsed = adapter.parse(task_spec=task_spec, completion=completion)
    assert parsed == {"my-answer": "Paris"}


def test_xml_adapter_parse_multiple_fields():
    TestSignature = ts(
        "question -> answer, explanation",
        instructions="Given the fields `question`, produce the fields `answer`, `explanation`.",
    )
    adapter = XMLAdapter()
    completion = "\n<answer>Paris</answer>\n<explanation>The capital of France is Paris.</explanation>\n"
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
            "number": output_field("number", type_=int, desc="The number."),
            "flag": output_field("flag", type_=bool, desc="The flag."),
        },
        instructions="Given the fields , produce the fields `number`, `flag`.",
    )
    adapter = XMLAdapter()
    completion = "\n<number>42</number>\n<flag>true</flag>\n"
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert parsed == {"number": 42, "flag": True}


def test_xml_adapter_parse_raises_on_type_error():
    TestSignature = make_task_spec(
        {"number": output_field("number", type_=int, desc="The number.")},
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
            "question": input_field("question", desc="The question."),
            "result": output_field("result", type_=InnerModel, desc="The result."),
        },
        instructions="Given the fields `question`, produce the fields `result`.",
    )
    adapter = XMLAdapter()
    fields_with_values = {
        FieldBinding(name="result", field=TestSignature.output_fields["result"]): InnerModel(value=5, label="foo")
    }
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip().startswith("<result>")
    assert '"value": 5' in xml
    assert '"label": "foo"' in xml
    assert xml.strip().endswith("</result>")
    completion = '<result>{"value": 5, "label": "foo"}</result>'
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert isinstance(parsed["result"], InnerModel)
    assert parsed["result"].value == 5
    assert parsed["result"].label == "foo"


def test_xml_adapter_format_and_parse_list_of_models():

    class Item(pydantic.BaseModel):
        name: str
        score: float

    TestSignature = make_task_spec(
        {"items": output_field("items", type_=list[Item], desc="The items.")},
        instructions="Given the fields , produce the fields `items`.",
    )
    adapter = XMLAdapter()
    items = [Item(name="a", score=1.1), Item(name="b", score=2.2)]
    fields_with_values = {FieldBinding(name="items", field=TestSignature.output_fields["items"]): items}
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip().startswith("<items>")
    assert '"name": "a"' in xml
    assert '"score": 2.2' in xml
    assert xml.strip().endswith("</items>")
    import json

    completion = f"<items>{json.dumps([i.model_dump() for i in items])}</items>"
    parsed = adapter.parse(task_spec=TestSignature, completion=completion)
    assert isinstance(parsed["items"], list)
    assert all(isinstance(i, Item) for i in parsed["items"])
    assert parsed["items"][0].name == "a"
    assert parsed["items"][1].score == 2.2


def test_xml_adapter_with_tool_like_output():

    class ToolCall(pydantic.BaseModel):
        name: str
        args: dict
        result: str

    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tool_calls": output_field("tool_calls", type_=list[ToolCall], desc="The tool calls."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `tool_calls`, `answer`.",
    )
    adapter = XMLAdapter()
    tool_calls = [
        ToolCall(name="get_weather", args={"city": "Tokyo"}, result="Sunny"),
        ToolCall(name="get_population", args={"country": "Japan", "year": 2023}, result="125M"),
    ]
    fields_with_values = {
        FieldBinding(name="tool_calls", field=TestSignature.output_fields["tool_calls"]): tool_calls,
        FieldBinding(
            name="answer", field=TestSignature.output_fields["answer"]
        ): "The weather is Sunny. Population is 125M.",
    }
    xml = adapter.format_field_with_value(fields_with_values)
    assert xml.strip().startswith("<tool_calls>")
    assert '"name": "get_weather"' in xml
    assert '"result": "125M"' in xml
    assert xml.strip().endswith("</answer>")
    import json

    completion = f"<tool_calls>{json.dumps([tc.model_dump() for tc in tool_calls])}</tool_calls>\n<answer>The weather is Sunny. Population is 125M.</answer>"
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
    adapter = XMLAdapter()
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


def test_xml_adapter_with_code():
    CodeAnalysis = make_task_spec(
        {
            "code": input_field("code", type_=Code, desc="The code."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Analyze the time complexity of the code",
    )
    adapter = XMLAdapter()
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
    adapter = XMLAdapter()
    lm = LM(model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='<code>print("Hello, world!")</code>'))],
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


def test_xml_adapter_full_prompt():
    QA = make_task_spec(
        {
            "query": input_field("query", desc="The query."),
            "context": input_field("context", type_=str | None, desc="The context."),
            "answer": output_field("answer", desc="The answer."),
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
    expected_system = f"Your input fields are:\n1. `query` (str): The query.\n2. `context` ({union_type_repr}): The context.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n<query>\n{{query}}\n</query>\n\n<context>\n{{context}}\n</context>\n\n<answer>\n{{answer}}\n</answer>\nIn adhering to this structure, your objective is: \n        Given the fields `query`, `context`, produce the fields `answer`."
    expected_user = "<query>\nwhen was Marie Curie born\n</query>\n\nRespond with the corresponding output fields wrapped in XML tags `<answer>`."
    assert messages[0]["content"] == expected_system
    assert messages[1]["content"] == expected_user


def test_format_system_message():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answers": output_field("answers", type_=list[str], desc="The answers."),
            "scores": output_field("scores", type_=list[float], desc="The scores."),
        },
        instructions="Answer the question with multiple answers and scores",
    )
    adapter = XMLAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answers` (list[str]): The answers.\n2. `scores` (list[float]): The scores.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n<question>\n{question}\n</question>\n\n<answers>\n{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}\n</answers>\n\n<scores>\n{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}\n</scores>\nIn adhering to this structure, your objective is: \n        Answer the question with multiple answers and scores'
    assert system_message == expected_system_message
