import logging
from typing import Any, Literal

from typing_extensions import override

from dspy.clients._litellm import is_litellm_context_window_error
from dspy.clients.errors import (
    _exception_message,
    _exception_provider_code,
    _exception_request_id,
    _exception_retry_after,
    _exception_status,
    _lm_error_class_from_status,
)
from dspy.clients.lm.errors import _lm_error_class_from_litellm_exception
from dspy.clients.lm.litellm_access import _get_litellm
from dspy.clients.lm.transport import (
    completion_fn_for_model_type,
    lm_response_for_model_type,
    provider_request_for_model_type,
)
from dspy.clients.lm_strict import validate_lm_kwargs, validate_lm_state
from dspy.clients.model_id import split_provider_model
from dspy.clients.openai_format.reasoning_models import is_openai_reasoning_model
from dspy.core.types import LMRequest, LMResponse, NativeAdaptationMode
from dspy.core.types.lm_provider import LMProviderOptions
from dspy.errors import ContextWindowExceededError, LMConfigurationError, LMError
from dspy.runtime.callback import Callback

from ..base_lm import BaseLM

logger = logging.getLogger(__name__)


class LM(BaseLM):
    __module__ = "dspy.clients.lm"

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: list[Callback] | None = None,
        num_retries: int = 3,
        use_developer_role: bool = False,
        provider_options: LMProviderOptions | None = None,
    ) -> None:
        merged_provider = provider_options or LMProviderOptions()
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            callbacks=callbacks,
            provider_options=merged_provider,
        )
        self.use_developer_role = use_developer_role

    @override
    def _get_initial_kwargs(
        self,
        *,
        temperature: float | None,
        max_tokens: int | None,
        provider_options: LMProviderOptions,
    ) -> dict[str, Any]:
        if is_openai_reasoning_model(self.model):
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise LMConfigurationError(
                    "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to `dspy.clients.lm.LM(...)`, e.g., `from dspy.clients.lm import LM; LM('openai/gpt-5', temperature=1.0, max_tokens=16000)`",
                    model=self.model,
                    provider=self._provider_name,
                )
            initial_kwargs = provider_options.to_kwargs()
            if temperature is not None:
                initial_kwargs["temperature"] = temperature
            if max_tokens is not None:
                initial_kwargs["max_tokens"] = max_tokens
            return validate_lm_kwargs(initial_kwargs)
        return super()._get_initial_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            provider_options=provider_options,
        )

    @property
    def _provider_name(self) -> str:
        return split_provider_model(self.model)[0]

    @property
    @override
    def supports_function_calling(self) -> bool:
        return _get_litellm().supports_function_calling(model=self.model)

    @property
    @override
    def supports_reasoning(self) -> bool:
        return _get_litellm().supports_reasoning(self.model)

    @property
    @override
    def reasoning_adaptation_mode(self) -> NativeAdaptationMode:
        if "gpt-5" in self.model and self.model_type == "chat":
            return NativeAdaptationMode.SKIP
        return NativeAdaptationMode.ADAPT

    @property
    @override
    def citations_adaptation_mode(self) -> NativeAdaptationMode:
        if self.model.startswith("anthropic/"):
            return NativeAdaptationMode.SKIP
        return NativeAdaptationMode.ADAPT

    @property
    @override
    def supports_response_schema(self) -> bool:
        return _get_litellm().supports_response_schema(model=self.model, custom_llm_provider=self._provider_name)

    @property
    @override
    def supported_params(self) -> set[str]:
        params = _get_litellm().get_supported_openai_params(model=self.model, custom_llm_provider=self._provider_name)
        return set(params) if params else set()

    def _wrap_litellm_exception(self, exc: Exception) -> LMError:
        if isinstance(exc, LMError):
            return exc
        status = _exception_status(exc)
        provider = getattr(exc, "llm_provider", None) or self._provider_name
        model = getattr(exc, "model", None) or self.model
        message = _exception_message(exc)
        metadata = {
            "model": model,
            "provider": provider,
            "provider_code": _exception_provider_code(exc),
            "status": status,
            "request_id": _exception_request_id(exc),
            "retry_after": _exception_retry_after(exc),
        }
        if is_litellm_context_window_error(exc):
            return ContextWindowExceededError(message=message or "Context window exceeded", **metadata)
        exc_cls = _lm_error_class_from_litellm_exception(exc) or _lm_error_class_from_status(status)
        error_kwargs: dict[str, Any] = {}
        for key, value in metadata.items():
            if key in {"status", "retry_after"}:
                if isinstance(value, int | float):
                    error_kwargs[key] = value
            elif isinstance(value, str | type(None)):
                error_kwargs[key] = value
        return exc_cls(message or "", **error_kwargs)

    async def aforward(self, request: LMRequest) -> LMResponse:
        provider_request = provider_request_for_model_type(self.model_type, request, self)
        litellm_cache_args = {"no-cache": True, "no-store": True}
        completion = completion_fn_for_model_type(self.model_type, lm=self)
        try:
            results = await completion(request=provider_request, num_retries=self.num_retries, cache=litellm_cache_args)
        except Exception as e:
            if isinstance(e, LMError):
                raise
            raise self._wrap_litellm_exception(e) from e
        self._check_truncation(results)
        return lm_response_for_model_type(self.model_type, results, request, self)

    @override
    def dump_state(self):
        state = super().dump_state()
        if self.use_developer_role:
            state["use_developer_role"] = self.use_developer_role
        return validate_lm_state(state)

    @classmethod
    @override
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False):
        state = validate_lm_state(dict(state))
        use_developer_role = state.pop("use_developer_role", False)
        state.pop("finetuning_model", None)
        state.pop("launch_kwargs", None)
        state.pop("train_kwargs", None)
        instance = super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)
        if not isinstance(instance, LM):
            raise TypeError(f"Expected LM instance from load_state, got {type(instance).__name__}.")
        instance.use_developer_role = use_developer_role
        return instance

    def _check_truncation(self, results) -> None:
        if self.model_type != "responses" and any(c.finish_reason == "length" for c in results["choices"]):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. You can inspect the latest LM interactions with `run.inspect_call_log()`. To avoid truncation, consider passing a larger max_tokens when setting up LM. You may also consider increasing the temperature (currently {self.kwargs['temperature']})  if the reason for truncation is repetition."
            )
