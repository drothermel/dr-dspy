import logging
import os
import re
import threading
from typing import Any, Literal

from typing_extensions import override

from dspy.clients._litellm import get_litellm, is_litellm_context_window_error
from dspy.clients.lm.errors import (
    _exception_message,
    _exception_provider_code,
    _exception_request_id,
    _exception_retry_after,
    _exception_status,
    _lm_error_class_from_litellm_exception,
    _lm_error_class_from_status,
)
from dspy.clients.lm.headers import _add_dspy_identifier_to_headers
from dspy.clients.lm.responses_compat import _convert_chat_request_to_responses_request
from dspy.clients.openai import OpenAIProvider
from dspy.clients.openai_format import (
    completion_to_lm_response,
    cost_from_response,
    responses_to_lm_response,
    to_openai_chat_request,
    to_openai_responses_request,
    to_openai_text_request,
    usage_from_response,
)
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.core.types import LMRequest, LMResponse
from dspy.core.types.config import merge_lm_request_config
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import ContextWindowExceededError, LMConfigurationError, LMError, LMUnsupportedFeatureError

from ..base_lm import BaseLM

logger = logging.getLogger(__name__)


def _get_litellm():
    return get_litellm(feature="dspy.clients.lm.LM")


def _is_openai_reasoning_model(model: str) -> bool:
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()
    return (
        re.match("^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\\d{4}-\\d{2}-\\d{2})?|gpt-5(?!-chat)(?:-.*)?)$", model_family)
        is not None
    )


class LM(BaseLM):
    __module__ = "dspy.clients.lm"

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        use_developer_role: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            callbacks=callbacks,
            **kwargs,
        )
        self.provider = provider or self.infer_provider()
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.use_developer_role = use_developer_role

    @override
    def _get_initial_kwargs(self, *, temperature, max_tokens, **kwargs) -> dict[str, Any]:
        if _is_openai_reasoning_model(self.model):
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise LMConfigurationError(
                    "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to `dspy.clients.lm.LM(...)`, e.g., `from dspy.clients.lm import LM; LM('openai/gpt-5', temperature=1.0, max_tokens=16000)`",
                    model=self.model,
                    provider=self._provider_name,
                )
            initial_kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        else:
            initial_kwargs = super()._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
        return initial_kwargs

    @property
    def _provider_name(self) -> str:
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return "openai"

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
        import dspy.clients.lm as lm_package

        provider_request = self._provider_request(request)
        litellm_cache_args = {"no-cache": True, "no-store": True}
        if self.model_type == "chat":
            completion = lm_package.alitellm_completion
        elif self.model_type == "text":
            completion = lm_package.alitellm_text_completion
        elif self.model_type == "responses":
            completion = lm_package.alitellm_responses_completion
        else:
            raise LMConfigurationError(
                f"Unsupported model_type {self.model_type!r} for `dspy.clients.lm.LM`.",
                model=self.model,
                provider=self._provider_name,
            )
        try:
            results = await completion(request=provider_request, num_retries=self.num_retries, cache=litellm_cache_args)
        except Exception as e:
            if isinstance(e, LMError):
                raise
            raise self._wrap_litellm_exception(e) from e
        self._check_truncation(results)
        return self._response_from_provider(results, request)

    def _provider_request(self, request: LMRequest) -> dict[str, Any]:
        if self.use_developer_role and self.model_type == "responses":
            request = request.model_copy(
                update={
                    "messages": [
                        message.model_copy(update={"role": "developer"}) if message.role == "system" else message
                        for message in request.messages
                    ]
                }
            )
        request = request.model_copy(update={"config": merge_lm_request_config(lm=self, config=request.config)})
        if self.model_type == "chat":
            provider_request = to_openai_chat_request(request)
        elif self.model_type == "text":
            provider_request = to_openai_text_request(request)
        elif self.model_type == "responses":
            provider_request = to_openai_responses_request(request)
        else:
            raise LMConfigurationError(
                f"Unsupported model_type {self.model_type!r} for `dspy.clients.lm.LM`.",
                model=self.model,
                provider=self._provider_name,
            )
        lm_defaults = {
            key: value
            for key, value in self.kwargs.items()
            if value is not None and key not in {"cache", "reasoning_effort", "reasoning"}
        }
        return {**lm_defaults, **provider_request}

    def _response_from_provider(self, response: Any, request: LMRequest) -> LMResponse:
        if self.model_type == "responses":
            lm_response = responses_to_lm_response(response=response, request=request)
        elif self.model_type in {"chat", "text"}:
            lm_response = completion_to_lm_response(response=response, request=request)
        else:
            raise LMConfigurationError(
                f"Unsupported model_type {self.model_type!r} for `dspy.clients.lm.LM`.",
                model=self.model,
                provider=self._provider_name,
            )
        return lm_response.model_copy(
            update={
                "model": getattr(response, "model", None) or lm_response.model,
                "usage": usage_from_response(response),
                "cost": cost_from_response(response),
                "provider_response": response,
            }
        )

    def launch(self, launch_kwargs: dict[str, Any] | None = None) -> None:
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs: dict[str, Any] | None = None) -> None:
        self.provider.kill(self, launch_kwargs)

    def finetune(
        self,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> TrainingJob:
        if not self.provider.finetunable:
            raise LMUnsupportedFeatureError(
                f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly setting `provider` when creating the `dspy.clients.lm.LM` instance. For example, `from dspy.clients.lm import LM; from dspy.clients.openai import OpenAIProvider; LM('openai/gpt-4.1-mini-2025-04-14', provider=OpenAIProvider())`.",
                model=self.model,
                provider=self._provider_name,
                features=["finetuning"],
            )

        def thread_function_wrapper():
            return self._run_finetune_job(job)

        thread = threading.Thread(target=thread_function_wrapper)
        train_kwargs = train_kwargs or self.train_kwargs
        model_to_finetune = self.finetuning_model or self.model
        job = self.provider.TrainingJob(
            thread=thread,
            model=model_to_finetune,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs,
        )
        thread.start()
        return job

    def reinforce(self, train_kwargs) -> ReinforceJob:
        if not self.provider.reinforceable:
            raise LMUnsupportedFeatureError(
                f"Provider {self.provider} does not implement the reinforcement learning interface.",
                model=self.model,
                provider=self._provider_name,
                features=["reinforce"],
            )
        job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
        job.initialize()
        return job

    def _run_finetune_job(self, job: TrainingJob) -> None:
        try:
            if job.model is None or job.train_data is None:
                raise ValueError("TrainingJob requires model and train_data before finetuning.")
            model = self.provider.finetune(
                job=job,
                model=job.model,
                train_data=job.train_data,
                train_data_format=job.train_data_format,
                train_kwargs=job.train_kwargs,
            )
            lm = self.copy(model=model)
            job.set_result(lm)
        except Exception as err:
            logger.exception(err)
            job.set_result(err)

    def infer_provider(self) -> Provider:
        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        return Provider()

    @override
    def dump_state(self):
        state = super().dump_state()
        state.update(
            {
                "finetuning_model": self.finetuning_model,
                "launch_kwargs": self.launch_kwargs,
                "train_kwargs": self.train_kwargs,
            }
        )
        if self.use_developer_role:
            state["use_developer_role"] = self.use_developer_role
        if _is_openai_reasoning_model(self.model) and "max_completion_tokens" in state:
            state["max_tokens"] = state.pop("max_completion_tokens")
        return state

    @classmethod
    @override
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False):
        state = dict(state)
        model = state.get("model")
        if isinstance(model, str) and _is_openai_reasoning_model(model) and ("max_completion_tokens" in state):
            if "max_tokens" not in state:
                state["max_tokens"] = state["max_completion_tokens"]
            state.pop("max_completion_tokens")
        return super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)

    def _check_truncation(self, results) -> None:
        if self.model_type != "responses" and any(c.finish_reason == "length" for c in results["choices"]):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. You can inspect the latest LM interactions with `lm.inspect_history()` or `dspy.clients.base_lm.inspect_history()`. To avoid truncation, consider passing a larger max_tokens when setting up LM. You may also consider increasing the temperature (currently {self.kwargs['temperature']})  if the reason for truncation is repetition."
            )


async def alitellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    return await _get_litellm().acompletion(
        cache=cache, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request
    )


async def alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    model = request.pop("model").split("/", 1)
    headers = request.pop("headers", None)
    provider, model = (model[0] if len(model) > 1 else "openai", model[-1])
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")
    prompt = request.pop("prompt")
    return await _get_litellm().atext_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def alitellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)
    return await _get_litellm().aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )
