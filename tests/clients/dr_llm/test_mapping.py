from __future__ import annotations

from unittest.mock import patch

import pytest
from dr_llm.llm import EffortSpec

from dspy.clients.dr_llm.mapping import (
    backend_response_to_lm_response,
    lm_request_to_backend_request,
    probe_backend_request,
)
from dspy.clients.dr_llm.provider_name import parse_dr_llm_provider
from dspy.core.types import LMMessage, LMMessageRole, LMRequest, User
from dspy.core.types.config import LMConfig, LMReasoningConfig, ReasoningEffort
from dspy.core.types.parts import LMImagePart, LMTextPart
from dspy.errors import LMConfigurationError, LMUnsupportedFeatureError
from dspy.testing import DummyLM
from tests.clients.dr_llm._helpers import make_backend_response, make_lm_request


def test_parse_dr_llm_provider_rejects_unknown() -> None:
    with pytest.raises(LMUnsupportedFeatureError, match="not-a-provider"):
        parse_dr_llm_provider("not-a-provider", model="not-a-provider/gpt-4.1-mini")


def test_lm_request_rejects_unknown_provider() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = make_lm_request()
    request = request.model_copy(update={"model": "not-a-provider/gpt-4.1-mini"})
    with pytest.raises(LMUnsupportedFeatureError, match="provider"):
        lm_request_to_backend_request(request, lm=lm)


def test_probe_backend_request_rejects_unknown_provider() -> None:
    lm = DummyLM([{"answer": "x"}])
    lm.model = "not-a-provider/gpt-4.1-mini"
    with pytest.raises(LMUnsupportedFeatureError, match="provider"):
        probe_backend_request(lm)


def test_lm_request_to_backend_request_maps_text_messages() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = make_lm_request(content="hello world")
    backend_request = lm_request_to_backend_request(request, lm=lm)
    assert backend_request.provider == "openai"
    assert backend_request.model == "gpt-4.1-mini"
    assert backend_request.messages[0].content == "hello world"


def test_lm_request_to_backend_request_maps_sampling() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[User(LMTextPart(text="hi"))],
        config=LMConfig(temperature=0.5, top_p=0.9, max_tokens=128),
    )
    backend_request = lm_request_to_backend_request(request, lm=lm)
    assert backend_request.sampling is not None
    assert backend_request.sampling.temperature == 0.5
    assert backend_request.sampling.top_p == 0.9
    assert backend_request.max_tokens == 128


def test_lm_request_rejects_tools() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = make_lm_request()
    request = request.model_copy(update={"tools": [{"type": "function", "name": "f", "parameters": {}}]})
    with pytest.raises(LMUnsupportedFeatureError):
        lm_request_to_backend_request(request, lm=lm)


def test_lm_request_rejects_unsupported_roles() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[LMMessage(role=LMMessageRole.TOOL, parts=[LMTextPart(text="result")])],
    )
    with pytest.raises(LMUnsupportedFeatureError, match="role"):
        lm_request_to_backend_request(request, lm=lm)


def test_lm_request_rejects_multimodal_parts() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[User(LMImagePart(url="https://example.com/x.png"))],
    )
    with pytest.raises(LMUnsupportedFeatureError):
        lm_request_to_backend_request(request, lm=lm)


@pytest.mark.parametrize(
    ("config_kwargs", "feature"),
    [
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"stop": ["END"]}, "stop"),
        ({"n": 2}, "n"),
        ({"logprobs": True}, "logprobs"),
        ({"tool_choice": {"mode": "auto"}}, "tool_choice"),
        ({"prompt_cache": {"enabled": True}}, "prompt_cache"),
        ({"extensions": {"foo": "bar"}}, "extensions"),
    ],
)
def test_lm_request_rejects_unsupported_merged_config(config_kwargs: dict, feature: str) -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[User(LMTextPart(text="hi"))],
        config=LMConfig(**config_kwargs),
    )
    with pytest.raises(LMUnsupportedFeatureError, match=feature):
        lm_request_to_backend_request(request, lm=lm)


def test_lm_request_rejects_unsupported_reasoning_fields() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[User(LMTextPart(text="hi"))],
        config=LMConfig(reasoning=LMReasoningConfig(max_tokens=1000)),
    )
    with pytest.raises(LMUnsupportedFeatureError, match="reasoning"):
        lm_request_to_backend_request(request, lm=lm)


def test_lm_request_rejects_reasoning_summary() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[User(LMTextPart(text="hi"))],
        config=LMConfig(reasoning=LMReasoningConfig(summary="auto")),
    )
    with pytest.raises(LMUnsupportedFeatureError, match=r"reasoning\.summary"):
        lm_request_to_backend_request(request, lm=lm)


def test_lm_request_maps_reasoning_effort() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[User(LMTextPart(text="hi"))],
        config=LMConfig(reasoning=LMReasoningConfig(effort=ReasoningEffort.HIGH)),
    )
    backend_request = lm_request_to_backend_request(request, lm=lm)
    assert backend_request.effort == EffortSpec.HIGH
    assert backend_request.reasoning is None


def test_effort_from_config_raises_on_invalid_effort() -> None:
    lm = DummyLM([{"answer": "x"}])
    request = LMRequest(
        model="openai/gpt-4.1-mini",
        messages=[User(LMTextPart(text="hi"))],
        config=LMConfig(reasoning=LMReasoningConfig(effort=ReasoningEffort.LOW)),
    )
    with (
        patch(
            "dspy.clients.dr_llm.mapping.EffortSpec",
            side_effect=ValueError("invalid effort"),
        ),
        pytest.raises(LMConfigurationError, match="reasoning effort"),
    ):
        lm_request_to_backend_request(request, lm=lm)


def test_backend_response_to_lm_response() -> None:
    request = make_lm_request()
    response = backend_response_to_lm_response(
        make_backend_response(text="answer", source="pool_cache"),
        request=request,
    )
    assert response.text == "answer"
    assert response.output.provider_data["source"] == "pool_cache"
