from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from dr_llm.backends import DirectBackend, PoolBackend
from dr_llm.backends.models import PoolBackendConfig
from dr_llm.llm import CallMode
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


class DrLlmPoolLM(DrLlmLM):
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
        super().__init__(
            model=model,
            mode=mode,
            registry=registry,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
        self._pool_config = pool_config
        self._default_session_id = session_id
        self._closed = False
        self._backend = PoolBackend(pool_config, registry=self._registry)
        # Capabilities probe only; DirectBackend has no lifecycle teardown.
        self._direct_backend = DirectBackend(self._registry)

    @property
    @override
    def _completion_backend(self):
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
