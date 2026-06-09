from __future__ import annotations

import pytest

from dspy.clients.dr_llm.mapping import (
    backend_response_to_lm_response,
    lm_request_to_backend_request,
    probe_backend_request,
    split_provider_model,
)
from dspy.core.types import LMMessage, LMRequest, User
from dspy.core.types.config import LMConfig
from dspy.core.types.parts import LMImagePart, LMTextPart
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import LMUnsupportedFeatureError
from tests.clients.dr_llm._helpers import make_backend_response, make_lm_request


def test_probe_backend_request_rejects_unknown_provider() -> None:
    lm = DummyLM([{"answer": "x"}])
    lm.model = "not-a-provider/gpt-4.1-mini"
    with pytest.raises(LMUnsupportedFeatureError, match="provider"):
        probe_backend_request(lm)


def test_split_provider_model() -> None:
    assert split_provider_model("openai/gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert split_provider_model("gpt-4.1-mini") == ("openai", "gpt-4.1-mini")


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
        messages=[LMMessage(role="tool", parts=[LMTextPart(text="result")])],
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


def test_backend_response_to_lm_response() -> None:
    request = make_lm_request()
    response = backend_response_to_lm_response(
        make_backend_response(text="answer", source="pool_cache"),
        request=request,
    )
    assert response.text == "answer"
    assert response.output.provider_data["source"] == "pool_cache"
