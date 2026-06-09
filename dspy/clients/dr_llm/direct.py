from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_llm.backends import DirectBackend
from dr_llm.llm import CallMode
from typing_extensions import override

from dspy.clients.dr_llm.base import DrLlmLM

if TYPE_CHECKING:
    from dr_llm.llm.providers.core.registry import ProviderRegistry

    from dspy.runtime.callback import Callback


class DrLlmDirectLM(DrLlmLM):
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
        super().__init__(
            model=model,
            mode=mode,
            registry=registry,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
        self._backend = DirectBackend(self._registry)

    @property
    @override
    def _completion_backend(self):
        return self._backend

    @property
    @override
    def _capabilities_backend(self):
        return self._backend

    @classmethod
    @override
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> DrLlmDirectLM:
        ctor_kwargs = cls._parse_dr_llm_ctor_state(state)
        return cls(**ctor_kwargs)
