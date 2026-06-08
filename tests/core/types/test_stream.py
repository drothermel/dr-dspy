import pydantic
import pytest

from dspy.core.types import (
    LMOutputBuilder,
    LMStreamDeltaEvent,
    LMTextDelta,
    LMThinkingDelta,
    LMToolCallDelta,
)


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
