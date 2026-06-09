from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from dspy.clients.dr_llm import DrLlmDirectLM
from tests.clients.dr_llm._helpers import make_lm_request

if TYPE_CHECKING:
    from dspy.core.types import LMResponse


def _require_env(*keys: str) -> None:
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing live LM credentials: {', '.join(missing)}")


def _text(response: LMResponse) -> str:
    assert response.text is not None
    return response.text.strip()


@pytest.mark.llm_call
async def test_live_dr_llm_direct_openai_exact_reply() -> None:
    _require_env("OPENAI_API_KEY")
    model = os.getenv("LM_FOR_TEST_DIRECT_DR_LLM", "openai/gpt-4.1-mini")
    lm = DrLlmDirectLM(model, temperature=0.0, max_tokens=32)
    response = await lm.aforward(make_lm_request(content="Reply with exactly: beta"))
    assert "beta" in _text(response).lower()
    assert response.output.provider_data.get("source") == "direct"
    assert response.output.finish_reason is not None
