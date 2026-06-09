"""Shared dr-llm LM wiring.

Capabilities always probe a ``DirectBackend``: ``PoolBackend`` has no public
``.capabilities()`` API, so pool LMs keep a dedicated ``DirectBackend`` for
probe while completions use ``PoolBackend``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

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

if TYPE_CHECKING:
    from dr_llm.backends.models import BackendCapabilities
    from dr_llm.llm.providers.core.registry import ProviderRegistry

    from dspy.core.types import LMRequest, LMResponse
    from dspy.runtime.callback import Callback


class _CapabilitiesBackend(Protocol):
    def capabilities(self, request: Any) -> BackendCapabilities: ...


class _CompletionBackend(Protocol):
    async def acomplete(self, request: Any) -> Any: ...


class DrLlmLM(BaseLM):
    __module__ = "dspy.clients.dr_llm"

    _mode: CallMode
    _registry: ProviderRegistry
    _capabilities_cache: BackendCapabilities | None

    def __init__(
        self,
        model: str,
        *,
        mode: CallMode = CallMode.api,
        registry: ProviderRegistry | None = None,
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
        self._capabilities_cache = None

    @property
    def _completion_backend(self) -> _CompletionBackend:
        raise NotImplementedError

    @property
    def _capabilities_backend(self) -> _CapabilitiesBackend:
        raise NotImplementedError

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
            self._capabilities_cache = self._capabilities_backend.capabilities(
                probe_backend_request(self, mode=self._mode)
            )
        return self._capabilities_cache

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        backend_request = lm_request_to_backend_request(request, lm=self, mode=self._mode)
        try:
            backend_response = await self._completion_backend.acomplete(backend_request)
        except Exception as exc:
            raise wrap_backend_exception(exc, model=request.model) from exc
        return backend_response_to_lm_response(backend_response, request=request)

    @override
    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state["dr_llm_mode"] = self._mode.value if hasattr(self._mode, "value") else str(self._mode)
        return state

    @classmethod
    def _parse_dr_llm_ctor_state(cls, state: dict[str, Any]) -> dict[str, Any]:
        state = validate_lm_state(dict(state))
        state.pop(LM_CLASS_STATE_KEY, None)
        mode_raw = state.pop("dr_llm_mode", CallMode.api)
        mode = CallMode(mode_raw) if isinstance(mode_raw, str) else mode_raw
        model = state.pop("model")
        state.pop("model_type", "chat")
        state.pop("num_retries", 3)
        provider_data = state.pop(PROVIDER_OPTIONS_STATE_KEY, None)
        temperature = state.pop("temperature", None)
        max_tokens = state.pop("max_tokens", None)
        provider_options = provider_options_from_serialized_state(provider_data=provider_data, remaining=state)
        validate_dr_llm_ctor(model=model, provider_options=provider_options)
        return {
            "model": model,
            "mode": mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
