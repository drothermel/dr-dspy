import pydantic
import pytest

from dspy.core.types import LMOutput, LMResponse, LMThinkingPart, LMUsage


def test_response_rejects_empty_outputs():
    with pytest.raises(pydantic.ValidationError):
        LMResponse(model="model", outputs=[])


def test_usage_normalizes_existing_user_visible_token_aliases():
    provider_usage = LMUsage(prompt_tokens=1, completion_tokens=2)
    canonical_usage = LMUsage(input_tokens=1, output_tokens=2)
    assert provider_usage.input_tokens == 1
    assert provider_usage.output_tokens == 2
    assert provider_usage.total_tokens == 3
    assert canonical_usage.prompt_tokens == 1
    assert canonical_usage.completion_tokens == 2
    assert canonical_usage.total_tokens == 3


def test_output_to_value_preserves_redacted_thinking_part():
    thinking = LMThinkingPart(text="hidden", redacted=True)
    output = LMOutput(parts=[thinking])
    assert output.to_value() == [thinking]
