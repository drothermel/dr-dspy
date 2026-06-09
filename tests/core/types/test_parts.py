from typing import Any, cast

import pytest

from dspy.core.types import LMAudioPart, LMDocumentPart, LMMessage, LMToolCallPart, LMVideoPart, User
from tests.core.types.conftest import history_entry


def test_message_content_and_tool_calls_normalize_for_dspy_history_surface():
    message = cast("Any", LMMessage)(
        role="assistant",
        content="Use search.",
        tool_calls=[{"id": "call_1", "function": {"name": "search", "arguments": '{"query": "dspy"}'}}],
    )
    assert message.text == "Use search."
    assert [part for part in message.parts if isinstance(part, LMToolCallPart)] == [
        LMToolCallPart(id="call_1", name="search", args={"query": "dspy"})
    ]
    assert history_entry(message).messages_as_openai == [
        {
            "role": "assistant",
            "content": "Use search.",
            "tool_calls": [
                {"type": "function", "function": {"name": "search", "arguments": '{"query": "dspy"}'}, "id": "call_1"}
            ],
        }
    ]


def test_tool_result_content_none_normalizes_to_empty_parts():
    from dspy.core.types import LMToolResultPart

    assert LMToolResultPart(content=cast("Any", None)).content == []
    message = cast("Any", LMMessage)(role="tool", content=None, tool_call_id="call_1", name="search")
    result = message.parts[0]
    assert isinstance(result, LMToolResultPart)
    assert result.call_id == "call_1"
    assert result.name == "search"
    assert result.content == []
    assert history_entry(message).messages_as_openai == [
        {"role": "tool", "content": "", "tool_call_id": "call_1", "name": "search"}
    ]


def test_audio_content_accepts_url_and_history_preserves_url():
    message = cast("Any", LMMessage)(
        role="user",
        content=[{"type": "input_audio", "input_audio": {"url": "https://example.com/audio.wav", "format": "wav"}}],
    )
    audio = message.parts[0]
    assert isinstance(audio, LMAudioPart)
    assert audio.url == "https://example.com/audio.wav"
    assert history_entry(message).messages_as_openai[0]["content"][0]["input_audio"] == {
        "format": "wav",
        "url": "https://example.com/audio.wav",
    }


def test_audio_content_defaults_null_format_to_wav():
    message = cast("Any", LMMessage)(
        role="user",
        content=[{"type": "input_audio", "input_audio": {"url": "https://example.com/audio.wav", "format": None}}],
    )
    audio = message.parts[0]
    assert isinstance(audio, LMAudioPart)
    assert audio.media_type == "audio/wav"


def test_image_content_requires_mapping_with_url():
    with pytest.raises(TypeError, match="Image content block"):
        cast("Any", LMMessage)(
            role="user", content=[{"type": "image_url", "image_url": "https://example.com/image.png"}]
        )
    with pytest.raises(ValueError, match="requires url"):
        cast("Any", LMMessage)(role="user", content=[{"type": "image_url", "image_url": {}}])


def test_video_data_round_trips_through_history_messages():
    message = User(LMVideoPart(data="YWJj", media_type="video/mp4"))
    content = history_entry(message).messages_as_openai[0]["content"][0]
    round_tripped = cast("Any", LMMessage)(role="user", content=[content]).parts[0]
    assert content == {"type": "video", "video": {"media_type": "video/mp4", "data": "data:video/mp4;base64,YWJj"}}
    assert isinstance(round_tripped, LMVideoPart)
    assert round_tripped.data == "YWJj"
    assert round_tripped.media_type == "video/mp4"


def test_document_source_url_stays_url_and_round_trips_through_history_messages():
    message = cast("Any", LMMessage)(
        role="user", content=[{"type": "document", "source": "https://example.com/report.pdf", "title": "Report"}]
    )
    document = message.parts[0]
    content = history_entry(message).messages_as_openai[0]["content"][0]
    round_tripped = cast("Any", LMMessage)(role="user", content=[content]).parts[0]
    assert isinstance(document, LMDocumentPart)
    assert document.url == "https://example.com/report.pdf"
    assert document.data is None
    assert isinstance(round_tripped, LMDocumentPart)
    assert round_tripped.url == "https://example.com/report.pdf"
    assert round_tripped.data is None
