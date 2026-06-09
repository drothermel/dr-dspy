from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from dspy.runtime.run_context import RunContext
from dspy.utils.dummies import DummyLM
from tests.clients.dr_llm._helpers import make_lm_request
from tests.clients.dr_llm.conftest import backend_request_for_lm, seed_complete_samples

if TYPE_CHECKING:
    from dspy.clients.dr_llm import DrLlmPoolLM
    from dspy.core.types import LMRequest


@pytest.mark.integration
async def test_dr_llm_pool_lm_aforward_miss_then_cache_hit(dspy_pool_lm: DrLlmPoolLM) -> None:
    request = make_lm_request(content="dspy integration aforward unique prompt")

    first = await dspy_pool_lm.aforward(request)
    assert first.text is not None
    assert first.text.startswith("generated-")
    assert first.output.provider_data["source"] == "generated"

    second = await dspy_pool_lm.aforward(request)
    assert second.output.provider_data["source"] == "pool_cache"
    assert second.text == first.text


@pytest.mark.integration
async def test_dr_llm_pool_lm_acquire_samples_session_semantics(
    dspy_pool_lm: DrLlmPoolLM,
    dspy_lm_request: LMRequest,
) -> None:
    backend_request = backend_request_for_lm(dspy_pool_lm, dspy_lm_request)
    seed_complete_samples(dspy_pool_lm._backend.store, backend_request, count=12)

    run = RunContext.create(lm=DummyLM([{"answer": "x"}]), adapter=MagicMock(), init_run_log=False)

    first = await dspy_pool_lm.acquire_samples(dspy_lm_request, n=10, run=run, session_id="s1")
    assert len(first) == 10
    first_texts = {response.text for response in first}
    assert len(first_texts) == 10

    second = await dspy_pool_lm.acquire_samples(dspy_lm_request, n=3, run=run, session_id="s1")
    assert len(second) == 3
    assert first_texts.isdisjoint({response.text for response in second})

    third = await dspy_pool_lm.acquire_samples(dspy_lm_request, n=3, run=run, session_id="s2")
    assert len(third) == 3
    assert all(response.text is not None and response.text.startswith("seed-") for response in third)
