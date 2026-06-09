from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import psycopg
import pytest
from dr_llm.backends.converters import backend_request_payload
from dr_llm.backends.fingerprint import fingerprint_request
from dr_llm.backends.models import BackendRequest, PoolBackendConfig
from dr_llm.backends.schema import BACKENDS_KEY_COLUMN, backends_pool_schema
from dr_llm.errors import TransientPersistenceError
from dr_llm.llm import CallMode, LlmResponse, Message, ProviderName, TokenUsage
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.sampling.db.names import claims_table_name
from psycopg import sql

from dspy.clients.dr_llm import DrLlmPoolLM
from dspy.clients.dr_llm.mapping import lm_request_to_backend_request
from tests.clients.dr_llm._helpers import make_lm_request

if TYPE_CHECKING:
    from collections.abc import Generator

    from dr_llm.pool.pool_store import PoolStore

    from dspy.core.types import LMRequest

_POOL_NAME = "itest_dspy"
_SCHEMA = backends_pool_schema(_POOL_NAME)
_CONSUMER_ID = _POOL_NAME


def integration_database_url() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _drop_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for tbl in reversed(_SCHEMA.table_names()):
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier("public", tbl)))
        claims_tbl = claims_table_name(_SCHEMA.name, _CONSUMER_ID)
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier("public", claims_tbl)))
        catalog_row = conn.execute("SELECT to_regclass('public.pool_catalog') IS NOT NULL").fetchone()
        catalog_exists = bool(catalog_row and catalog_row[0])
        if catalog_exists:
            conn.execute("DELETE FROM pool_catalog WHERE pool_name = %s", [_POOL_NAME])
        conn.commit()


def _backend_request(*, content: str) -> BackendRequest:
    return BackendRequest(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        messages=[Message(role="user", content=content)],
    )


def _llm_response(*, text: str) -> LlmResponse:
    return LlmResponse(
        text=text,
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )


def mock_openai_registry() -> MagicMock:
    counter = {"n": 0}

    def _generate(_request: object) -> LlmResponse:
        counter["n"] += 1
        return _llm_response(text=f"generated-{counter['n']}")

    orchestrator = MagicMock()
    orchestrator.generate.side_effect = _generate
    registry = MagicMock()
    registry.get.return_value = orchestrator
    return registry


def seed_complete_samples(store: PoolStore, request: BackendRequest, *, count: int) -> str:
    fingerprint = fingerprint_request(request)
    for idx in range(count):
        store.insert_sample(
            PoolSample(
                key_values={BACKENDS_KEY_COLUMN: fingerprint},
                request=backend_request_payload(request),
                response=_llm_response(text=f"seed-{idx}").model_dump(mode="json"),
                finish_reason="stop",
                sample_idx=idx,
            )
        )
    return fingerprint


@pytest.fixture
def dspy_pool_lm() -> Generator[DrLlmPoolLM, None, None]:
    dsn = integration_database_url()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run dr-llm pool integration tests")
    try:
        _drop_tables(dsn)
        lm = DrLlmPoolLM(
            "openai/gpt-4.1-mini",
            pool_config=PoolBackendConfig(
                pool_name=_POOL_NAME,
                database_url=dsn,
                consumer_id=_CONSUMER_ID,
                num_workers=2,
                lease_seconds=30,
                acquire_timeout_seconds=30,
            ),
            registry=mock_openai_registry(),
        )
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        pytest.skip(f"Postgres unavailable for dr-llm pool integration tests: {exc}")
    yield lm
    lm.close()
    dsn_after = integration_database_url()
    if dsn_after:
        _drop_tables(dsn_after)


@pytest.fixture
def pool_lm_request() -> BackendRequest:
    return _backend_request(content="dspy pool integration prompt")


@pytest.fixture
def dspy_lm_request() -> LMRequest:
    return make_lm_request(content="dspy pool integration prompt")


def backend_request_for_lm(lm: DrLlmPoolLM, request: LMRequest) -> BackendRequest:
    return lm_request_to_backend_request(request, lm=lm)
