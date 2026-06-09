from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from dr_llm.backends import DirectBackend, PoolBackend
from dr_llm.backends.models import PoolBackendConfig
from dr_llm.llm import CallMode
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from dspy.clients.dr_llm.base import DrLlmLM
from dspy.clients.dr_llm.errors import wrap_backend_exception
from dspy.clients.dr_llm.mapping import (
    backend_response_to_lm_response,
    lm_request_to_backend_request,
)
from dspy.runtime.log_paths import resolve_run_bucket

if TYPE_CHECKING:
    from dr_llm.llm.providers.core.registry import ProviderRegistry

    from dspy.clients.dr_llm.controls import DrLlmProviderControls
    from dspy.clients.dr_llm.protocol import PoolSessionIdResolver
    from dspy.core.types import LMRequest, LMResponse
    from dspy.core.types.lm_provider import LMProviderOptions
    from dspy.runtime.callback import Callback
    from dspy.runtime.run_context import RunContext


class DrLlmAcquireResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    responses: list[Any] = Field(default_factory=list)
    claimed_from_cache: int = 0
    generated: int = 0


def resolve_pool_session_id(run: RunContext, *, fallback: str | None = None) -> str:
    session = run.log_session
    if session is not None:
        return f"{resolve_run_bucket()}:{session.timestamp}"
    if fallback:
        return fallback
    return uuid.uuid4().hex


class DrLlmPoolLM(DrLlmLM):
    def __init__(
        self,
        model: str,
        *,
        pool_config: PoolBackendConfig,
        mode: CallMode = CallMode.api,
        registry: ProviderRegistry | None = None,
        dr_llm_controls: DrLlmProviderControls | dict[str, Any] | None = None,
        session_id: str | None = None,
        session_id_resolver: PoolSessionIdResolver | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            mode=mode,
            registry=registry,
            dr_llm_controls=dr_llm_controls,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
        self._pool_config = pool_config
        self._default_session_id = session_id
        self._session_id_resolver = session_id_resolver or resolve_pool_session_id
        self._closed = False
        self._backend = PoolBackend(pool_config, registry=self._registry)
        # Capabilities probe only; DirectBackend has no lifecycle teardown.
        self._direct_backend = DirectBackend(self._registry)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DrLlmPoolLM is closed.")

    @property
    @override
    def _completion_backend(self):
        self._ensure_open()
        return self._backend

    @property
    @override
    def _capabilities_backend(self):
        return self._direct_backend

    def close(self) -> None:
        if self._closed:
            return
        self._backend.close()
        self._closed = True

    def __enter__(self) -> DrLlmPoolLM:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        self._ensure_open()
        return await super().aforward(request)

    async def acquire_samples(
        self,
        request: LMRequest,
        *,
        n: int,
        run: RunContext,
        session_id: str | None = None,
    ) -> list[LMResponse]:
        result = await self.acquire_samples_result(request, n=n, run=run, session_id=session_id)
        return result.responses

    async def acquire_samples_result(
        self,
        request: LMRequest,
        *,
        n: int,
        run: RunContext,
        session_id: str | None = None,
    ) -> DrLlmAcquireResult:
        self._ensure_open()
        backend_request = lm_request_to_backend_request(request, lm=self, mode=self._mode)
        sid = session_id or self._session_id_resolver(run, fallback=self._default_session_id)
        try:
            result = await self._backend.aacquire(backend_request, sid, n)
        except Exception as exc:
            raise wrap_backend_exception(exc, model=request.model) from exc
        return DrLlmAcquireResult(
            responses=[backend_response_to_lm_response(response, request=request) for response in result.responses],
            claimed_from_cache=result.claimed_from_cache,
            generated=result.generated,
        )

    @override
    def copy(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider_options: LMProviderOptions | None = None,
    ) -> DrLlmPoolLM:
        if provider_options is not None:
            raise TypeError("DrLlmPoolLM.copy() does not accept provider_options.")
        new_temperature = temperature if temperature is not None else self.kwargs.get("temperature")
        new_max_tokens = max_tokens if max_tokens is not None else self.kwargs.get("max_tokens")
        return type(self)(
            model=model or self.model,
            pool_config=self._pool_config,
            mode=self._mode,
            registry=self._registry,
            dr_llm_controls=self._dr_llm_controls,
            session_id=self._default_session_id,
            session_id_resolver=self._session_id_resolver,
            temperature=new_temperature,
            max_tokens=new_max_tokens,
            callbacks=list(getattr(self, "callbacks", []) or []),
        )

    @override
    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state["dr_llm_pool_config"] = self._pool_config.model_dump(mode="json")
        if self._default_session_id is not None:
            state["dr_llm_session_id"] = self._default_session_id
        return state

    @classmethod
    @override
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> DrLlmPoolLM:
        state = dict(state)
        pool_config_raw = state.pop("dr_llm_pool_config")
        pool_config = PoolBackendConfig(**pool_config_raw)
        session_id = state.pop("dr_llm_session_id", None)
        ctor_kwargs = cls._parse_dr_llm_ctor_state(state)
        return cls(
            pool_config=pool_config,
            session_id=session_id,
            **ctor_kwargs,
        )
