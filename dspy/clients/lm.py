import logging
import os
import re
import threading
import warnings
from typing import Any, Literal

import pydantic
from typing_extensions import override

from dspy.__metadata__ import __version__
from dspy.clients._litellm import get_litellm, is_litellm_context_window_error
from dspy.clients.cache import request_cache
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
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMAuthError,
    LMBillingError,
    LMConfigurationError,
    LMError,
    LMInvalidRequestError,
    LMNotConfiguredError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
    LMUnexpectedError,
    LMUnsupportedFeatureError,
    LMUnsupportedModelError,
)

from .base_lm import BaseLM

logger = logging.getLogger(__name__)


def _get_litellm():
    return get_litellm(feature="dspy.clients.lm.LM")


def _is_openai_reasoning_model(model: str) -> bool:
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()
    return (
        re.match(
            r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
            model_family,
        )
        is not None
    )


class LM(BaseLM):
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        use_developer_role: bool = False,
        **kwargs,
    ) -> None:
        """Create a new language model instance for use with DSPy modules and programs.

        Args:
            model: The model to use. This should be a string of the form
                `"llm_provider/llm_name"` supported by LiteLLM. For example,
                `"openai/gpt-4o"`.
            model_type: The type of the model, such as `"chat"`, `"text"`, or
                `"responses"`.
            temperature: The sampling temperature to use when generating responses.
            max_tokens: The maximum number of tokens to generate per response.
            cache: Whether to cache the model responses for reuse to improve performance
                and reduce costs.
            callbacks: A list of callback functions to run before and after each request.
            num_retries: The number of times to retry a request if it fails transiently due to
                network error, rate limiting, etc. Requests are retried with exponential
                backoff.
            provider: The provider to use. If not specified, the provider will be inferred from the model.
            finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
                from the models available for inference.
            rollout_id: Optional integer used to differentiate cache entries for otherwise
                identical requests. Different values bypass DSPy's caches while still caching
                future calls with the same inputs and rollout ID. Note that `rollout_id`
                only affects generation when `temperature` is non-zero. This argument is
                stripped before sending requests to the provider.
        """
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
            callbacks=callbacks,
            **kwargs,
        )

        self.provider = provider or self.infer_provider()
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.use_developer_role = use_developer_role

        self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))

    @override
    def _get_initial_kwargs(self, *, temperature, max_tokens, **kwargs) -> dict[str, Any]:
        # Override BaseLM's default kwargs shape for LiteLLM/model-family-specific token parameters.
        if _is_openai_reasoning_model(self.model):
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise LMConfigurationError(
                    "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to "
                    "`dspy.clients.lm.LM(...)`, e.g., "
                    "`from dspy.clients.lm import LM; LM('openai/gpt-5', temperature=1.0, max_tokens=16000)`",
                    model=self.model,
                    provider=self._provider_name,
                )
            initial_kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        else:
            initial_kwargs = super()._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)

        if initial_kwargs.get("rollout_id") is None:
            initial_kwargs.pop("rollout_id", None)
        return initial_kwargs

    @property
    def _provider_name(self) -> str:
        """Extract the provider name from the model string (e.g., 'openai' from 'openai/gpt-4o')."""
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

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id) -> None:
        if not self._warned_zero_temp_rollout and rollout_id is not None and temperature == 0:
            warnings.warn(
                "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.",
                stacklevel=3,
            )
            self._warned_zero_temp_rollout = True

    def _get_cached_completion_fn(self, completion_fn, cache):
        ignored_args_for_cache_key = ["api_key", "api_base", "base_url"]
        if cache:
            completion_fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=ignored_args_for_cache_key,
            )(completion_fn)

        litellm_cache_args = {"no-cache": True, "no-store": True}

        return completion_fn, litellm_cache_args

    def _wrap_litellm_exception(self, exc: Exception) -> LMError:
        """Convert exceptions raised at the LiteLLM boundary into DSPy LM exceptions."""
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
        return exc_cls(message, **metadata)  # ty:ignore[invalid-argument-type]

    async def aforward(self, request: LMRequest) -> LMResponse:
        """Call the configured LM asynchronously."""
        rollout_id = request.config.cache.rollout_id if request.config.cache is not None else None
        temperature = (
            request.config.temperature if request.config.temperature is not None else self.kwargs.get("temperature")
        )
        self._warn_zero_temp_rollout(temperature, rollout_id)
        provider_request = self._provider_request(request)
        cache = self._cache_enabled(request)

        if self.model_type == "chat":
            completion = alitellm_completion
        elif self.model_type == "text":
            completion = alitellm_text_completion
        elif self.model_type == "responses":
            completion = alitellm_responses_completion
        else:
            raise LMConfigurationError(
                f"Unsupported model_type {self.model_type!r} for `dspy.clients.lm.LM`.",
                model=self.model,
                provider=self._provider_name,
            )

        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)

        try:
            results = await completion(
                request=provider_request,
                num_retries=self.num_retries,
                cache=litellm_cache_args,
            )
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

        if request.config.cache is not None and request.config.cache.rollout_id is not None:
            provider_request["rollout_id"] = request.config.cache.rollout_id

        lm_defaults = {key: value for key, value in self.kwargs.items() if value is not None}
        return {**lm_defaults, **provider_request}

    def _cache_enabled(self, request: LMRequest) -> bool:
        if request.config.cache is None or request.config.cache.enabled is None:
            return self.cache
        return request.config.cache.enabled

    def _response_from_provider(self, response: Any, request: LMRequest) -> LMResponse:
        if self.model_type == "responses":
            lm_response = responses_to_lm_response(response, request)
        elif self.model_type in {"chat", "text"}:
            lm_response = completion_to_lm_response(response, request)
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
                "cache_hit": bool(getattr(response, "cache_hit", False)),
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
                f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly "
                "setting `provider` when creating the `dspy.clients.lm.LM` instance. For example, "
                "`from dspy.clients.lm import LM; from dspy.clients.openai import OpenAIProvider; "
                "LM('openai/gpt-4.1-mini-2025-04-14', provider=OpenAIProvider())`.",
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
        # TODO(GRPO Team): Should we return an initialized job here?

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
        # TODO(enhance): We should listen for keyboard interrupts somewhere.
        # Requires TrainingJob.cancel() to be implemented for each provider.
        try:
            model = self.provider.finetune(
                job=job,
                model=job.model,  # ty:ignore[invalid-argument-type]
                train_data=job.train_data,  # ty:ignore[invalid-argument-type]
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
        """Return a sanitized reconstruction state for this LM.

        Returns:
            A dictionary that can be passed to `BaseLM.load_state` to
            reconstruct this `LM`. The state excludes API keys.
        """
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
        if isinstance(model, str) and _is_openai_reasoning_model(model) and "max_completion_tokens" in state:
            if "max_tokens" not in state:
                state["max_tokens"] = state["max_completion_tokens"]
            state.pop("max_completion_tokens")

        return super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)

    def _check_truncation(self, results) -> None:
        if self.model_type != "responses" and any(c.finish_reason == "length" for c in results["choices"]):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
                "You can inspect the latest LM interactions with `lm.inspect_history()` or "
                "`dspy.clients.base_lm.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
                " if the reason for truncation is repetition."
            )


async def alitellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    return await _get_litellm().acompletion(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=headers,
        **request,
    )


async def alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    model = request.pop("model").split("/", 1)
    headers = request.pop("headers", None)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

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
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)

    return await _get_litellm().aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def _convert_chat_request_to_responses_request(request: dict[str, Any]):
    """
    Convert a chat request to a responses request
    See https://platform.openai.com/docs/api-reference/responses/create for the responses API specification.
    Also see https://platform.openai.com/docs/api-reference/chat/create for the chat API specification.
    """
    request = dict(request)
    if "messages" in request:
        input_items = []
        for msg in request.pop("messages"):
            content_blocks = []
            c = msg.get("content")
            if isinstance(c, str):
                content_blocks.append({"type": "input_text", "text": c})
            elif isinstance(c, list):
                for item in c:
                    content_blocks.append(_convert_content_item_to_responses_format(item))  # noqa: PERF401 dynamic typing/lint migration for scoped ty adoption
            input_items.append({"role": msg.get("role", "user"), "content": content_blocks})
        request["input"] = input_items
    # Convert `reasoning_effort` to reasoning format supported by the Responses API
    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}

    # Convert `response_format` to `text.format` for Responses API
    if "response_format" in request:
        response_format = request.pop("response_format")
        if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
            response_format = {
                "name": response_format.__name__,
                "type": "json_schema",
                "schema": response_format.model_json_schema(),
            }
        text = request.pop("text", {})
        request["text"] = {**text, "format": response_format}

    return request


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a content item from Chat API format to Responses API format.

    For images, converts from:
        {"type": "image_url", "image_url": {"url": "..."}}
    To:
        {"type": "input_image", "image_url": "..."}

    For text, converts from:
        {"type": "text", "text": "..."}
    To:
        {"type": "input_text", "text": "..."}

    For other types, passes through as-is.
    """
    if item.get("type") == "image_url":
        image_url = item.get("image_url", {}).get("url", "")
        return {
            "type": "input_image",
            "image_url": image_url,
        }
    if item.get("type") == "text":
        return {
            "type": "input_text",
            "text": item.get("text", ""),
        }
    if item.get("type") == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }

    return item


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None):
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{__version__}",
        **headers,
    }


# --------
# Errors
# --------


def _safe_litellm_exception_class(name: str) -> type[Exception] | None:
    cls = getattr(_get_litellm(), name, None)
    return cls if isinstance(cls, type) and issubclass(cls, Exception) else None


def _lm_error_class_from_litellm_exception(exc: Exception) -> type[LMError] | None:
    message = _exception_message(exc).lower()
    class_name = type(exc).__name__.lower()
    if _exception_status(exc) is None and any(
        phrase in message for phrase in ("api key", "apikey", "credentials", "environment variable")
    ):
        return LMNotConfiguredError
    if "timeout" in class_name or "timed out" in message or "timeout" in message:
        return LMTimeoutError
    if "connection" in class_name or "network" in message or "connection" in message:
        return LMTransportError

    mappings = [
        ("AuthenticationError", LMAuthError),
        ("RateLimitError", LMRateLimitError),
        ("NotFoundError", LMUnsupportedModelError),
        ("UnsupportedParamsError", LMUnsupportedFeatureError),
        ("UnprocessableEntityError", LMInvalidRequestError),
        ("ContentPolicyViolationError", LMInvalidRequestError),
        ("BadRequestError", LMInvalidRequestError),
        ("InvalidRequestError", LMInvalidRequestError),
        ("InternalServerError", LMServerError),
        ("ServiceUnavailableError", LMServerError),
        ("APIConnectionError", LMTransportError),
        ("APIResponseValidationError", LMProviderError),
        ("BudgetExceededError", LMBillingError),
        ("RouterRateLimitError", LMRateLimitError),
    ]
    for litellm_name, dspy_cls in mappings:
        litellm_cls = _safe_litellm_exception_class(litellm_name)
        if litellm_cls is not None and isinstance(exc, litellm_cls):
            return dspy_cls
    return None


def _lm_error_class_from_status(status: int | None) -> type[LMError]:
    if status in (401, 403):
        return LMAuthError
    if status == 402:
        return LMBillingError
    if status == 404:
        return LMUnsupportedModelError
    if status == 408:
        return LMTimeoutError
    if status == 429:
        return LMRateLimitError
    if status is not None and 400 <= status < 500:
        return LMInvalidRequestError
    if status is not None and status >= 500:
        return LMServerError
    return LMUnexpectedError if status is None else LMProviderError


# Best-effort LiteLLM/provider exception metadata extraction.
#
# LiteLLM exception metadata is not exposed as a single stable typed shape across providers, exception classes, and
# LiteLLM versions. Keep the defensive getattr-based extraction localized here so the rest of DSPy sees structured
# DSPyError metadata.
def _exception_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _exception_message(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    if message is None:
        message = str(exc)
    return str(message)


def _exception_headers(exc: Exception):
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or getattr(exc, "headers", None) or {}


def _exception_header(exc: Exception, name: str) -> str | None:
    headers = _exception_headers(exc)
    if not headers:
        return None
    try:
        return headers.get(name) or headers.get(name.lower())
    except AttributeError:
        return None


def _exception_request_id(exc: Exception) -> str | None:
    return (
        _exception_header(exc, "x-request-id")
        or _exception_header(exc, "request-id")
        or _exception_header(exc, "x-amzn-requestid")
        or _exception_header(exc, "x-ms-request-id")
    )


def _exception_retry_after(exc: Exception) -> float | None:
    retry_after = _exception_header(exc, "retry-after")
    try:
        return float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        return None


def _exception_provider_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") is not None:
            return str(error["code"])
        if body.get("code") is not None:
            return str(body["code"])
    return None
