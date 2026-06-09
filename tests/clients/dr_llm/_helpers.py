from __future__ import annotations

from typing import Literal

from dr_llm.backends.models import BackendResponse
from dr_llm.llm import CallMode, ProviderName, TokenUsage

from dspy.core.types import LMRequest, User
from dspy.core.types.parts import LMTextPart


def make_lm_request(*, model: str = "openai/gpt-4.1-mini", content: str = "hello") -> LMRequest:
    return LMRequest(
        model=model,
        messages=[User(LMTextPart(text=content))],
    )


def make_backend_response(
    *,
    text: str = "ok",
    source: Literal["direct", "pool_cache", "generated"] | None = "direct",
) -> BackendResponse:
    return BackendResponse(
        text=text,
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        source=source,
    )
