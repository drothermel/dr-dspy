from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from dr_llm.backends.models import BackendCapabilities
from dr_llm.llm import CallMode

from dspy.clients.dr_llm import DrLlmDirectLM
from tests.clients.dr_llm._helpers import make_backend_response, make_lm_request


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        provider="openai",
        model="gpt-4.1-mini",
        mode=CallMode.api,
        control_mode="reasoning",
    )


def test_dr_llm_direct_lm_aforward() -> None:
    backend_response = make_backend_response(text="from dr-llm")
    mock_backend = MagicMock()
    mock_backend.acomplete = AsyncMock(return_value=backend_response)
    mock_backend.capabilities.return_value = _capabilities()

    with patch("dspy.clients.dr_llm.direct.DirectBackend", return_value=mock_backend):
        lm = DrLlmDirectLM("openai/gpt-4.1-mini", temperature=0.0)
        response = asyncio.run(lm.aforward(make_lm_request()))

    assert response.text == "from dr-llm"
    mock_backend.acomplete.assert_awaited_once()


def test_dr_llm_direct_lm_capabilities() -> None:
    mock_backend = MagicMock()
    mock_backend.capabilities.return_value = _capabilities()

    with patch("dspy.clients.dr_llm.direct.DirectBackend", return_value=mock_backend):
        lm = DrLlmDirectLM("openai/gpt-4.1-mini")
        assert lm.supports_function_calling is False
        assert lm.supports_response_schema is False
        assert lm.supports_reasoning is True
        assert "temperature" in lm.supported_params
