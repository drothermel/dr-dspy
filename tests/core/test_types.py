import pydantic
import pytest

from dspy.core.types import (
    LMAudioPart,
    LMConfig,
    LMDocumentPart,
    LMHistoryEntry,
    LMMessage,
    LMOutput,
    LMOutputBuilder,
    LMPromptCacheConfig,
    LMReasoningConfig,
    LMRequest,
    LMResponse,
    LMStreamDeltaEvent,
    LMTextDelta,
    LMThinkingDelta,
    LMThinkingPart,
    LMToolCallDelta,
    LMToolCallPart,
    LMToolChoice,
    LMToolResultPart,
    LMUsage,
    LMVideoPart,
    User,
)


def _history_entry(message: LMMessage) -> LMHistoryEntry:
    return LMHistoryEntry(
        request=LMRequest(model="model", messages=[message]),
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )


def test_message_content_and_tool_calls_normalize_for_dspy_history_surface():
    message = LMMessage(
        role="assistant",
        content="Use search.",  # ty:ignore[unknown-argument]
        tool_calls=[
            {
                "id": "call_1",
                "function": {"name": "search", "arguments": '{"query": "dspy"}'},
            }
        ],  # ty:ignore[unknown-argument]
    )  # ty:ignore[missing-argument]

    assert message.text == "Use search."
    assert [part for part in message.parts if isinstance(part, LMToolCallPart)] == [
        LMToolCallPart(id="call_1", name="search", args={"query": "dspy"})
    ]
    assert _history_entry(message).messages_as_openai == [
        {
            "role": "assistant",
            "content": "Use search.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "dspy"}'},
                    "id": "call_1",
                }
            ],
        }
    ]


def test_tool_result_content_none_normalizes_to_empty_parts():
    assert LMToolResultPart(content=None).content == []  # ty:ignore[invalid-argument-type]

    message = LMMessage(role="tool", content=None, tool_call_id="call_1", name="search")  # ty:ignore[missing-argument, unknown-argument]
    result = message.parts[0]

    assert isinstance(result, LMToolResultPart)
    assert result.call_id == "call_1"
    assert result.name == "search"
    assert result.content == []
    assert _history_entry(message).messages_as_openai == [
        {"role": "tool", "content": "", "tool_call_id": "call_1", "name": "search"}
    ]


def test_audio_content_accepts_url_and_history_preserves_url():
    message = LMMessage(
        role="user",
        content=[
            {
                "type": "input_audio",
                "input_audio": {"url": "https://example.com/audio.wav", "format": "wav"},
            }
        ],  # ty:ignore[unknown-argument]
    )  # ty:ignore[missing-argument]

    audio = message.parts[0]
    assert isinstance(audio, LMAudioPart)
    assert audio.url == "https://example.com/audio.wav"
    assert _history_entry(message).messages_as_openai[0]["content"][0]["input_audio"] == {
        "format": "wav",
        "url": "https://example.com/audio.wav",
    }


def test_audio_content_defaults_null_format_to_wav():
    message = LMMessage(
        role="user",
        content=[
            {
                "type": "input_audio",
                "input_audio": {"url": "https://example.com/audio.wav", "format": None},
            }
        ],  # ty:ignore[unknown-argument]
    )  # ty:ignore[missing-argument]

    audio = message.parts[0]

    assert isinstance(audio, LMAudioPart)
    assert audio.media_type == "audio/wav"


def test_image_content_requires_mapping_with_url():
    with pytest.raises(TypeError, match="Image content block"):
        LMMessage(role="user", content=[{"type": "image_url", "image_url": "https://example.com/image.png"}])  # ty:ignore[missing-argument, unknown-argument]

    with pytest.raises(ValueError, match="requires url"):
        LMMessage(role="user", content=[{"type": "image_url", "image_url": {}}])  # ty:ignore[missing-argument, unknown-argument]


def test_video_data_round_trips_through_history_messages():
    message = User(LMVideoPart(data="YWJj", media_type="video/mp4"))
    content = _history_entry(message).messages_as_openai[0]["content"][0]
    round_tripped = LMMessage(role="user", content=[content]).parts[0]  # ty:ignore[missing-argument, unknown-argument]

    assert content == {
        "type": "video",
        "video": {"media_type": "video/mp4", "data": "data:video/mp4;base64,YWJj"},
    }
    assert isinstance(round_tripped, LMVideoPart)
    assert round_tripped.data == "YWJj"
    assert round_tripped.media_type == "video/mp4"


def test_document_source_url_stays_url_and_round_trips_through_history_messages():
    message = LMMessage(
        role="user",
        content=[
            {
                "type": "document",
                "source": "https://example.com/report.pdf",
                "title": "Report",
            }
        ],  # ty:ignore[unknown-argument]
    )  # ty:ignore[missing-argument]
    document = message.parts[0]
    content = _history_entry(message).messages_as_openai[0]["content"][0]
    round_tripped = LMMessage(role="user", content=[content]).parts[0]  # ty:ignore[missing-argument, unknown-argument]

    assert isinstance(document, LMDocumentPart)
    assert document.url == "https://example.com/report.pdf"
    assert document.data is None
    assert isinstance(round_tripped, LMDocumentPart)
    assert round_tripped.url == "https://example.com/report.pdf"
    assert round_tripped.data is None


def test_config_extensions_surface_in_history_kwargs():
    config = LMConfig(temperature=0.2, extensions={"provider_flag": True})
    request = LMRequest(model="model", messages=[], config=config)
    entry = LMHistoryEntry(
        request=request,
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )

    assert entry.kwargs == {"provider_flag": True, "temperature": 0.2}


def test_lm_config_rejects_unknown_top_level_keys():
    with pytest.raises(pydantic.ValidationError):
        LMConfig.from_kwargs(temperature=0.2, provider_flag=True)


def test_lm_config_accepts_canonical_nested_fields():
    config = LMConfig(
        reasoning=LMReasoningConfig(effort="high", summary="auto"),
        tool_choice=LMToolChoice(mode="auto", parallel=False),
        prompt_cache=LMPromptCacheConfig(enabled=True, key="prompt-cache"),
        extensions={"provider_flag": True},
    )

    assert config.reasoning.effort == "high"  # ty:ignore[unresolved-attribute]
    assert config.reasoning.summary == "auto"  # ty:ignore[unresolved-attribute]
    assert config.tool_choice.mode == "auto"  # ty:ignore[unresolved-attribute]
    assert config.tool_choice.parallel is False  # ty:ignore[unresolved-attribute]
    assert config.prompt_cache.enabled is True  # ty:ignore[unresolved-attribute]
    assert config.prompt_cache.key == "prompt-cache"  # ty:ignore[unresolved-attribute]
    assert config.extensions == {"provider_flag": True}


def test_usage_normalizes_existing_user_visible_token_aliases():
    provider_usage = LMUsage(prompt_tokens=1, completion_tokens=2)
    canonical_usage = LMUsage(input_tokens=1, output_tokens=2)

    assert provider_usage.input_tokens == 1
    assert provider_usage.output_tokens == 2
    assert provider_usage.total_tokens == 3
    assert canonical_usage.prompt_tokens == 1
    assert canonical_usage.completion_tokens == 2
    assert canonical_usage.total_tokens == 3


def test_default_config_does_not_serialize_empty_stop_sequences():
    request = LMRequest.from_call(model="model", prompt="hi")
    entry = LMHistoryEntry(
        request=request,
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )

    assert request.config.stop is None
    assert entry.kwargs == {}


def test_history_entry_exposes_typed_derived_properties():
    message = User("hi")
    request = LMRequest.from_call(model="model", messages=[message], temperature=0.2)
    response = LMResponse.from_text("ok", model="response-model", usage={"input_tokens": 1}, cost=0.5)
    entry = LMHistoryEntry(request=request, response=response, timestamp="timestamp", uuid="uuid")

    assert entry.model == "model"
    assert entry.prompt == "hi"
    assert entry.messages == [message]
    assert entry.messages_as_openai == [{"role": "user", "content": "hi"}]
    assert entry.outputs == ["ok"]
    assert entry.usage["input_tokens"] == 1
    assert entry.cost == 0.5
    assert entry.kwargs == {"temperature": 0.2}
    assert entry.response_model == "response-model"


def test_response_rejects_empty_outputs():
    with pytest.raises(pydantic.ValidationError):
        LMResponse(model="model", outputs=[])


def test_output_to_value_preserves_redacted_thinking_part():
    thinking = LMThinkingPart(text="hidden", redacted=True)
    output = LMOutput(parts=[thinking])

    assert output.to_value() == [thinking]


def test_stream_event_indices_must_be_non_negative():
    with pytest.raises(pydantic.ValidationError):
        LMStreamDeltaEvent(output_index=-1, part_index=0, delta=LMTextDelta(text="x"))

    with pytest.raises(pydantic.ValidationError):
        LMStreamDeltaEvent(output_index=0, part_index=-1, delta=LMTextDelta(text="x"))


def test_stream_builder_rejects_sparse_output_indices():
    builder = LMOutputBuilder()
    builder.apply(LMStreamDeltaEvent(output_index=2, part_index=0, delta=LMTextDelta(text="third")))

    with pytest.raises(ValueError, match="output indices"):
        builder.to_response()


def test_stream_builder_rejects_sparse_part_indices():
    builder = LMOutputBuilder()
    builder.apply(LMStreamDeltaEvent(output_index=0, part_index=1, delta=LMTextDelta(text="second")))

    with pytest.raises(ValueError, match="part indices"):
        builder.to_response()


def test_stream_builder_rejects_delta_type_changes():
    builder = LMOutputBuilder()
    builder.apply(LMStreamDeltaEvent(output_index=0, part_index=0, delta=LMTextDelta(text="text")))

    with pytest.raises(ValueError, match="thinking delta"):
        builder.apply(LMStreamDeltaEvent(output_index=0, part_index=0, delta=LMThinkingDelta(text="thought")))


def test_stream_builder_rejects_incomplete_tool_call_arguments():
    builder = LMOutputBuilder()
    builder.apply(
        LMStreamDeltaEvent(
            output_index=0,
            part_index=0,
            delta=LMToolCallDelta(id="call_1", name="search", args_delta='{"query": '),
        )
    )

    with pytest.raises(ValueError, match="tool-call arguments"):
        builder.to_response()
