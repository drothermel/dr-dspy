from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from dr_llm.backends import DirectBackend, PoolBackend
from dr_llm.backends.models import BackendCapabilities, PoolBackendConfig
from dr_llm.llm import CallMode
from dr_llm.llm.providers.default_registry import build_default_registry
from typing_extensions import override

from dspy.clients.base_lm import LM_CLASS_STATE_KEY, PROVIDER_OPTIONS_STATE_KEY, BaseLM
from dspy.clients.dr_llm.capabilities import (
    supported_params_v1,
    supports_reasoning_from_capabilities,
)
from dspy.clients.dr_llm.contract import (
    provider_options_from_serialized_state,
    validate_dr_llm_ctor,
)
from dspy.clients.dr_llm.errors import wrap_backend_exception
from dspy.clients.dr_llm.mapping import (
    backend_response_to_lm_response,
    lm_request_to_backend_request,
    probe_backend_request,
)
from dspy.clients.lm_strict import validate_lm_state
from dspy.core.types.lm_provider import LMProviderOptions
from dspy.runtime.run_log import resolve_run_bucket

if TYPE_CHECKING:
    from dr_llm.llm.providers.core.registry import ProviderRegistry

    from dspy.core.types import LMRequest, LMResponse
    from dspy.runtime.callback import Callback
    from dspy.runtime.run_context import RunContext


def resolve_pool_session_id(run: RunContext, *, fallback: str | None = None) -> str:
    session = run.log_session
    if session is not None:
        return f"{resolve_run_bucket()}:{session.timestamp}"
    if fallback:
        return fallback
    return uuid.uuid4().hex


class DrLlmPoolLM(BaseLM):
    __module__ = "dspy.clients.dr_llm"

    def __init__(
        self,
        model: str,
        *,
        pool_config: PoolBackendConfig,
        mode: CallMode = CallMode.api,
        registry: ProviderRegistry | None = None,
        session_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        validate_dr_llm_ctor(model=model)
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            num_retries=0,
            provider_options=LMProviderOptions(),
        )
        self._mode = mode
        self._registry = registry or build_default_registry()
        self._pool_config = pool_config
        self._default_session_id = session_id
        self._backend = PoolBackend(pool_config, registry=self._registry)
        self._direct_backend = DirectBackend(self._registry)
        self._capabilities_cache: BackendCapabilities | None = None

    @property
    @override
    def supports_function_calling(self) -> bool:
        return False

    @property
    @override
    def supports_response_schema(self) -> bool:
        return False

    @property
    @override
    def supports_reasoning(self) -> bool:
        return supports_reasoning_from_capabilities(self._cached_capabilities())

    @property
    @override
    def supported_params(self) -> set[str]:
        return supported_params_v1()

    def _cached_capabilities(self) -> BackendCapabilities:
        if self._capabilities_cache is None:
            self._capabilities_cache = self._direct_backend.capabilities(probe_backend_request(self, mode=self._mode))
        return self._capabilities_cache

    def close(self) -> None:
        direct_close = getattr(self._direct_backend, "close", None)
        if callable(direct_close):
            direct_close()
        self._backend.close()

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        backend_request = lm_request_to_backend_request(request, lm=self, mode=self._mode)
        try:
            backend_response = await self._backend.acomplete(backend_request)
        except Exception as exc:
            raise wrap_backend_exception(exc, model=request.model) from exc
        return backend_response_to_lm_response(backend_response, request=request)

    async def acquire_samples(
        self,
        request: LMRequest,
        *,
        n: int,
        run: RunContext,
        session_id: str | None = None,
    ) -> list[LMResponse]:
        backend_request = lm_request_to_backend_request(request, lm=self, mode=self._mode)
        sid = session_id or resolve_pool_session_id(run, fallback=self._default_session_id)
        try:
            result = await self._backend.aacquire(backend_request, sid, n)
        except Exception as exc:
            raise wrap_backend_exception(exc, model=request.model) from exc
        return [backend_response_to_lm_response(response, request=request) for response in result.responses]

    @override
    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state["dr_llm_mode"] = self._mode.value if hasattr(self._mode, "value") else str(self._mode)
        state["dr_llm_pool_config"] = self._pool_config.model_dump(mode="json")
        if self._default_session_id is not None:
            state["dr_llm_session_id"] = self._default_session_id
        return state

    @classmethod
    @override
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> DrLlmPoolLM:
        state = validate_lm_state(dict(state))
        state.pop(LM_CLASS_STATE_KEY, None)
        mode_raw = state.pop("dr_llm_mode", CallMode.api)
        mode = CallMode(mode_raw) if isinstance(mode_raw, str) else mode_raw
        pool_config_raw = state.pop("dr_llm_pool_config")
        pool_config = PoolBackendConfig(**pool_config_raw)
        session_id = state.pop("dr_llm_session_id", None)
        model = state.pop("model")
        state.pop("model_type", "chat")
        state.pop("num_retries", 3)
        provider_data = state.pop(PROVIDER_OPTIONS_STATE_KEY, None)
        temperature = state.pop("temperature", None)
        max_tokens = state.pop("max_tokens", None)
        provider_options = provider_options_from_serialized_state(provider_data=provider_data, remaining=state)
        validate_dr_llm_ctor(model=model, provider_options=provider_options)
        return cls(
            model=model,
            pool_config=pool_config,
            mode=mode,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
