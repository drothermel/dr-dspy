import pytest

from dspy.clients.openai_format.chat_request import request_messages_as_openai
from dspy.clients.openai_format.serialize import part_to_openai_blocks, parts_to_openai_content
from dspy.core.types import (
    LMBinaryPart,
    LMCitationPart,
    LMDocumentPart,
    LMImagePart,
    LMMessage,
    LMMessageRole,
    LMOpaquePart,
    LMRefusalPart,
    LMTextPart,
    LMThinkingPart,
    LMToolResultPart,
)
from dspy.core.types.request import LMRequest
from tests.core.types.conftest import history_messages_as_openai


def _part_fixtures():
    return [
        LMTextPart(text="Hello."),
        LMThinkingPart(text="Let me think."),
        LMRefusalPart(text="I cannot help with that."),
        LMCitationPart(title="Paper", text="Summary", url="https://example.com"),
        LMImagePart(url="https://example.com/image.png", media_type="image/png"),
        LMDocumentPart(url="https://example.com/report.pdf", title="Report", media_type="application/pdf"),
        LMBinaryPart(data="YWJj", media_type="application/octet-stream"),
        LMOpaquePart(block={"type": "custom", "value": "payload"}),
    ]


@pytest.mark.parametrize("part", _part_fixtures())
def test_part_serializes_to_single_openai_block(part):
    assert part_to_openai_blocks(part)[0]


def test_tool_result_with_nested_image_matches_live_serializer():
    result = LMToolResultPart(
        call_id="call_1",
        name="search",
        content=[LMImagePart(url="https://example.com/image.png", media_type="image/png")],
    )
    message = LMMessage(role=LMMessageRole.TOOL, parts=[result])
    expected_content = parts_to_openai_content(result.content)
    assert history_messages_as_openai(message) == [
        {
            "role": "tool",
            "content": expected_content,
            "tool_call_id": "call_1",
            "name": "search",
        }
    ]
    assert expected_content == [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]


def test_request_messages_as_openai_serializes_thinking_part():
    message = LMMessage(role=LMMessageRole.ASSISTANT, parts=[LMThinkingPart(text="Reasoning trace.")])
    request = LMRequest(model="model", messages=[message])
    assert request_messages_as_openai(request) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Reasoning trace."}]},
    ]
