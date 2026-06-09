from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from dr_llm.backends.models import PoolBackendConfig

from dspy.clients.base_lm import LM_CLASS_STATE_KEY, PROVIDER_OPTIONS_STATE_KEY
from dspy.clients.dr_llm import DrLlmDirectLM, DrLlmPoolLM
from dspy.errors import LMConfigurationError


def test_dr_llm_direct_lm_rejects_num_retries_kwarg() -> None:
    with pytest.raises(TypeError, match="num_retries"):
        DrLlmDirectLM("openai/gpt-4.1-mini", num_retries=3)


def test_dr_llm_direct_lm_rejects_unknown_ctor_kwargs() -> None:
    with pytest.raises(TypeError, match="timeout"):
        DrLlmDirectLM("openai/gpt-4.1-mini", timeout=30)


def test_dr_llm_direct_lm_rejects_provider_options_kwarg() -> None:
    with pytest.raises(TypeError, match="provider_options"):
        DrLlmDirectLM("openai/gpt-4.1-mini", provider_options={"api_key": "x"})


def test_dr_llm_direct_lm_happy_path_ctor() -> None:
    lm = DrLlmDirectLM("openai/gpt-4.1-mini", temperature=0.0, max_tokens=128)
    assert lm.model == "openai/gpt-4.1-mini"
    assert lm.kwargs.get("temperature") == 0.0
    assert lm.kwargs.get("max_tokens") == 128


def test_dr_llm_pool_lm_happy_path_ctor() -> None:
    config = PoolBackendConfig(pool_name="test_pool", database_url="postgresql://localhost/test")
    with (
        patch("dspy.clients.dr_llm.pool.PoolBackend", return_value=MagicMock()),
        patch("dspy.clients.dr_llm.pool.DirectBackend", return_value=MagicMock()),
    ):
        lm = DrLlmPoolLM(
            "openai/gpt-4.1-mini",
            pool_config=config,
            session_id="session-1",
            temperature=0.5,
        )
    assert lm.model == "openai/gpt-4.1-mini"
    assert lm._default_session_id == "session-1"


def test_dr_llm_direct_lm_load_state_rejects_legacy_provider_options() -> None:
    state = {
        LM_CLASS_STATE_KEY: "dspy.clients.dr_llm.direct.DrLlmDirectLM",
        "model": "openai/gpt-4.1-mini",
        "model_type": "chat",
        "num_retries": 3,
        PROVIDER_OPTIONS_STATE_KEY: {"api_key": "legacy-key"},
        "dr_llm_mode": "api",
    }
    with pytest.raises(LMConfigurationError, match="LMProviderOptions"):
        DrLlmDirectLM.load_state(state)


def test_dr_llm_pool_lm_load_state_rejects_legacy_provider_options() -> None:
    state = {
        LM_CLASS_STATE_KEY: "dspy.clients.dr_llm.pool.DrLlmPoolLM",
        "model": "openai/gpt-4.1-mini",
        "model_type": "chat",
        "num_retries": 3,
        PROVIDER_OPTIONS_STATE_KEY: {"timeout": 30.0},
        "dr_llm_mode": "api",
        "dr_llm_pool_config": {
            "pool_name": "test_pool",
            "database_url": "postgresql://localhost/test",
        },
    }
    with pytest.raises(LMConfigurationError, match="LMProviderOptions"):
        DrLlmPoolLM.load_state(state)
