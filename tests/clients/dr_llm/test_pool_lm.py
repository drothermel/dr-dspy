from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from dr_llm.backends.models import AcquireResult, BackendCapabilities, PoolBackendConfig
from dr_llm.llm import CallMode

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.dr_llm import DrLlmPoolLM, resolve_pool_session_id
from dspy.errors import LMUnexpectedError
from dspy.runtime import CallLogMode, TelemetryConfig
from dspy.runtime.run_context import RunContext
from dspy.testing import DummyLM
from tests.clients.dr_llm._helpers import make_backend_response, make_lm_request


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        provider="openai",
        model="gpt-4.1-mini",
        mode=CallMode.api,
        control_mode="reasoning",
    )


def test_resolve_pool_session_id_fallback() -> None:
    run = RunContext.create(
        lm=DummyLM([{"answer": "x"}]),
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    assert resolve_pool_session_id(run, fallback="session-1") == "session-1"


def test_dr_llm_pool_lm_aforward() -> None:
    pool_backend = MagicMock()
    pool_backend.acomplete = AsyncMock(return_value=make_backend_response(source="pool_cache"))
    direct_backend = MagicMock()
    direct_backend.capabilities.return_value = _capabilities()

    config = PoolBackendConfig(pool_name="test_pool", database_url="postgresql://localhost/test")

    with (
        patch("dspy.clients.dr_llm.pool.PoolBackend", return_value=pool_backend),
        patch("dspy.clients.dr_llm.pool.DirectBackend", return_value=direct_backend),
    ):
        lm = DrLlmPoolLM("openai/gpt-4.1-mini", pool_config=config)
        response = asyncio.run(lm.aforward(make_lm_request()))

    assert response.text == "ok"
    assert response.output.provider_data["source"] == "pool_cache"
    pool_backend.acomplete.assert_awaited_once()


def test_dr_llm_pool_lm_acquire_samples() -> None:
    pool_backend = MagicMock()
    pool_backend.aacquire = AsyncMock(
        return_value=AcquireResult(
            responses=[make_backend_response(text="a"), make_backend_response(text="b")],
            claimed_from_cache=1,
            generated=1,
        )
    )
    direct_backend = MagicMock()
    direct_backend.capabilities.return_value = _capabilities()

    config = PoolBackendConfig(pool_name="test_pool", database_url="postgresql://localhost/test")
    run = RunContext.create(
        lm=DummyLM([{"answer": "x"}]),
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )

    with (
        patch("dspy.clients.dr_llm.pool.PoolBackend", return_value=pool_backend),
        patch("dspy.clients.dr_llm.pool.DirectBackend", return_value=direct_backend),
    ):
        lm = DrLlmPoolLM("openai/gpt-4.1-mini", pool_config=config, session_id="fixed-session")
        responses = asyncio.run(lm.acquire_samples(make_lm_request(), n=2, run=run))

    assert len(responses) == 2
    pool_backend.aacquire.assert_awaited_once()
    await_args = pool_backend.aacquire.await_args
    assert await_args is not None
    assert await_args.args[1] == "fixed-session"


def test_dr_llm_pool_lm_close_closes_pool_backend_only() -> None:
    pool_backend = MagicMock()
    direct_backend = MagicMock()
    direct_backend.capabilities.return_value = _capabilities()

    config = PoolBackendConfig(pool_name="test_pool", database_url="postgresql://localhost/test")

    with (
        patch("dspy.clients.dr_llm.pool.PoolBackend", return_value=pool_backend),
        patch("dspy.clients.dr_llm.pool.DirectBackend", return_value=direct_backend),
    ):
        lm = DrLlmPoolLM("openai/gpt-4.1-mini", pool_config=config)
        lm.close()

    pool_backend.close.assert_called_once()


def test_dr_llm_pool_lm_close_is_idempotent() -> None:
    pool_backend = MagicMock()
    direct_backend = MagicMock()
    direct_backend.capabilities.return_value = _capabilities()

    config = PoolBackendConfig(pool_name="test_pool", database_url="postgresql://localhost/test")

    with (
        patch("dspy.clients.dr_llm.pool.PoolBackend", return_value=pool_backend),
        patch("dspy.clients.dr_llm.pool.DirectBackend", return_value=direct_backend),
    ):
        lm = DrLlmPoolLM("openai/gpt-4.1-mini", pool_config=config)
        lm.close()
        lm.close()

    pool_backend.close.assert_called_once()


def test_dr_llm_pool_lm_context_manager_closes_on_exit() -> None:
    pool_backend = MagicMock()
    direct_backend = MagicMock()
    direct_backend.capabilities.return_value = _capabilities()

    config = PoolBackendConfig(pool_name="test_pool", database_url="postgresql://localhost/test")

    with (
        patch("dspy.clients.dr_llm.pool.PoolBackend", return_value=pool_backend),
        patch("dspy.clients.dr_llm.pool.DirectBackend", return_value=direct_backend),
        DrLlmPoolLM("openai/gpt-4.1-mini", pool_config=config) as lm,
    ):
        assert lm.model == "openai/gpt-4.1-mini"

    pool_backend.close.assert_called_once()


def test_dr_llm_pool_lm_close_after_failed_acquire() -> None:
    pool_backend = MagicMock()
    pool_backend.aacquire = AsyncMock(side_effect=RuntimeError("acquire failed"))
    direct_backend = MagicMock()
    direct_backend.capabilities.return_value = _capabilities()

    config = PoolBackendConfig(pool_name="test_pool", database_url="postgresql://localhost/test")
    run = RunContext.create(
        lm=DummyLM([{"answer": "x"}]),
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )

    with (
        patch("dspy.clients.dr_llm.pool.PoolBackend", return_value=pool_backend),
        patch("dspy.clients.dr_llm.pool.DirectBackend", return_value=direct_backend),
    ):
        lm = DrLlmPoolLM("openai/gpt-4.1-mini", pool_config=config)
        try:
            asyncio.run(lm.acquire_samples(make_lm_request(), n=1, run=run))
        except LMUnexpectedError:
            pass
        lm.close()

    pool_backend.close.assert_called_once()
