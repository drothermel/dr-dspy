import pydantic
import pytest

from dspy.core.types import LMOutput, LMResponse, LMThinkingPart


def test_response_rejects_empty_outputs():
    with pytest.raises(pydantic.ValidationError):
        LMResponse(model="model", outputs=[])


def test_output_to_value_preserves_redacted_thinking_part():
    thinking = LMThinkingPart(text="hidden", redacted=True)
    output = LMOutput(parts=[thinking])
    assert output.to_value() == [thinking]
