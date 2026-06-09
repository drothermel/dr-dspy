from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_llm.backends import DirectBackend
from dr_llm.llm import CallMode
from dr_llm.llm.providers.default_registry import build_default_registry
from typing_extensions import override

from dspy.clients.base_lm import BaseLM
from dspy.clients.dr_llm.capabilities import (
    supported_params_v1,
    supports_reasoning_from_capabilities,
)
from dspy.clients.dr_llm.errors import wrap_backend_exception
from dspy.clients.dr_llm.mapping import (
    backend_response_to_lm_response,
    lm_request_to_backend_request,
    probe_backend_request,
)

if TYPE_CHECKING:
    from dr_llm.backends.models import BackendCapabilities
    from dr_llm.llm.providers.core.registry import ProviderRegistry

    from dspy.core.types import LMRequest, LMResponse
    from dspy.runtime.callback import Callback


class DrLlmDirectLM(BaseLM):
    __module__ = "dspy.clients.dr_llm"

    def __init__(
        self,
        model: str,
        *,
        mode: CallMode = CallMode.api,
        registry: ProviderRegistry | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: list[Callback] | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            num_retries=num_retries,
            provider_options=kwargs.pop("provider_options", None),
        )
        self._mode = mode
        self._registry = registry or build_default_registry()
        self._backend = DirectBackend(self._registry)
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
            self._capabilities_cache = self._backend.capabilities(probe_backend_request(self, mode=self._mode))
        return self._capabilities_cache

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        backend_request = lm_request_to_backend_request(request, lm=self, mode=self._mode)
        try:
            backend_response = await self._backend.acomplete(backend_request)
        except Exception as exc:
            raise wrap_backend_exception(exc, model=request.model) from exc
        return backend_response_to_lm_response(backend_response, request=request)

    @override
    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state["dr_llm_mode"] = self._mode.value if hasattr(self._mode, "value") else str(self._mode)
        return state

    @classmethod
    @override
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> DrLlmDirectLM:
        state = dict(state)
        mode_raw = state.pop("dr_llm_mode", CallMode.api)
        mode = CallMode(mode_raw) if isinstance(mode_raw, str) else mode_raw
        base = super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)
        instance = cls(
            model=base.model,
            mode=mode,
            temperature=base.kwargs.get("temperature"),
            max_tokens=base.kwargs.get("max_tokens"),
            num_retries=base.num_retries,
            provider_options=base.provider_options,
        )
        instance.callbacks = list(base.callbacks)
        return instance
