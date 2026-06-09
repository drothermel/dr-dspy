import pytest

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from tests.adapters.assertions import assert_multimodal_blocks
from tests.adapters.conftest import adapter_format_as_openai
from tests.adapters.scenarios.multimodal import (
    image_with_demos_case,
    nested_documents_case,
    nested_images_case,
    nested_images_with_demos_case,
    single_image_case,
)


@pytest.mark.parametrize("adapter_factory", [ChatAdapter, JSONAdapter, XMLAdapter])
def test_single_image_in_user_message(adapter_factory):
    scenario = single_image_case()
    messages = adapter_format_as_openai(
        adapter=adapter_factory(),
        task_spec=scenario.task_spec,
        demos=list(scenario.demos),
        inputs=scenario.inputs,
    )
    assert len(messages) == 2
    user_message = messages[1]
    content = user_message["content"]
    assert content is not None
    if adapter_factory is not XMLAdapter:
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[2]["type"] == "text"
    assert_multimodal_blocks(
        message=user_message,
        expected_blocks=[{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}],
    )


@pytest.mark.parametrize("adapter_factory", [ChatAdapter, JSONAdapter, XMLAdapter])
def test_image_with_few_shot_demos(adapter_factory):
    scenario = image_with_demos_case()
    messages = adapter_format_as_openai(
        adapter=adapter_factory(),
        task_spec=scenario.task_spec,
        demos=list(scenario.demos),
        inputs=scenario.inputs,
    )
    assert len(messages) == 6
    assert_multimodal_blocks(
        message=messages[1],
        expected_blocks=[{"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}],
    )
    assert_multimodal_blocks(
        message=messages[3],
        expected_blocks=[{"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}],
    )
    assert_multimodal_blocks(
        message=messages[5],
        expected_blocks=[{"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}],
    )
    if adapter_factory is ChatAdapter:
        assert "[[ ## completed ## ]]\n" in messages[2]["content"]
        assert "[[ ## completed ## ]]\n" in messages[4]["content"]


@pytest.mark.parametrize("adapter_factory", [ChatAdapter, JSONAdapter, XMLAdapter])
def test_nested_images_in_pydantic_wrapper(adapter_factory):
    scenario = nested_images_case()
    messages = adapter_format_as_openai(
        adapter=adapter_factory(),
        task_spec=scenario.task_spec,
        demos=list(scenario.demos),
        inputs=scenario.inputs,
    )
    assert_multimodal_blocks(
        message=messages[1],
        expected_blocks=[
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}},
        ],
    )


@pytest.mark.parametrize("adapter_factory", [ChatAdapter, JSONAdapter, XMLAdapter])
def test_nested_images_with_few_shot_demos(adapter_factory):
    scenario = nested_images_with_demos_case()
    messages = adapter_format_as_openai(
        adapter=adapter_factory(),
        task_spec=scenario.task_spec,
        demos=list(scenario.demos),
        inputs=scenario.inputs,
    )
    assert len(messages) == 4
    assert_multimodal_blocks(
        message=messages[1],
        expected_blocks=[
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}},
        ],
    )
    assert_multimodal_blocks(
        message=messages[-1],
        expected_blocks=[{"type": "image_url", "image_url": {"url": "https://example.com/image4.jpg"}}],
    )


@pytest.mark.parametrize("adapter_factory", [JSONAdapter, ChatAdapter])
def test_nested_documents(adapter_factory):
    scenario = nested_documents_case()
    messages = adapter_format_as_openai(
        adapter=adapter_factory(),
        task_spec=scenario.task_spec,
        demos=list(scenario.demos),
        inputs=scenario.inputs,
    )
    assert_multimodal_blocks(
        message=messages[1],
        expected_blocks=[
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "Hello, world!"},
                "citations": {"enabled": True},
            },
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "Hello, world 2!"},
                "citations": {"enabled": True},
            },
        ],
    )
