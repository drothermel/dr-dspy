import random

import pytest

from dspy.core.types import LMMessage, LMMessageRole, LMRequest, LMTextPart
from dspy.core.types.lm_config import LMConfig
from tests.test_utils import DummyLM, DummyVectorizer
from tests.test_utils.lm.answer_routing import follow_example_output


@pytest.mark.asyncio
async def test_dummy_lm_follow_examples_without_field_headers_returns_empty_output():
    lm = DummyLM([], follow_examples=True)
    request = LMRequest(
        model="dummy",
        messages=[LMMessage(role=LMMessageRole.USER, parts=[LMTextPart(text="plain question without field headers")])],
        config=LMConfig(),
    )
    response = await lm.aforward(request)
    assert response.outputs[0].parts == []


def test_follow_example_output_returns_none_when_no_field_headers():
    messages = [LMMessage(role=LMMessageRole.USER, parts=[LMTextPart(text="plain question")])]
    assert follow_example_output(messages) is None


def test_dummy_lm_supports_reasoning_property():
    lm = DummyLM([{}], supports_reasoning=True)
    assert lm.supports_reasoning is True


def test_dummy_vectorizer_does_not_mutate_global_rng():
    state_before = random.getstate()
    DummyVectorizer()
    state_after = random.getstate()
    assert state_before == state_after
