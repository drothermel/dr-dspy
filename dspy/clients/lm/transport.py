from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from dspy.clients.lm.headers import _add_dspy_identifier_to_headers
from dspy.clients.lm.litellm_access import _get_litellm
from dspy.clients.model_id import split_provider_model
from dspy.clients.openai_format import (
    completion_to_lm_response,
    cost_from_response,
    responses_to_lm_response,
    to_openai_chat_request,
    to_openai_responses_request,
    to_openai_text_request,
    usage_from_response,
)
from dspy.core.types import LMRequest, LMResponse
from dspy.core.types.config import merge_lm_request_config
from dspy.errors import LMConfigurationError

LitellmCompletionFn = Callable[..., Awaitable[Any]]
ToOpenAIRequestFn = Callable[[LMRequest], dict[str, Any]]
ToLMResponseFn = Callable[[Any, LMRequest], LMResponse]

_DEFAULT_LITELLM_CACHE = {"no-cache": True, "no-store": True}


class LitellmCompletionName(StrEnum):
    CHAT = "alitellm_completion"
    TEXT = "alitellm_text_completion"
    RESPONSES = "alitellm_responses_completion"


class ModelTypeRoute(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    completion_fn_name: LitellmCompletionName
    to_openai_request: ToOpenAIRequestFn
    to_lm_response: ToLMResponseFn


async def alitellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or _DEFAULT_LITELLM_CACHE
    request = dict(request)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    return await _get_litellm().acompletion(
        cache=cache, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request
    )


async def alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or _DEFAULT_LITELLM_CACHE
    request = dict(request)
    model = request.pop("model")
    headers = request.pop("headers", None)
    provider, model_name = split_provider_model(model)
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")
    prompt = request.pop("prompt")
    return await _get_litellm().atext_completion(
        cache=cache,
        model=f"text-completion-openai/{model_name}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def alitellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or _DEFAULT_LITELLM_CACHE
    request = dict(request)
    headers = request.pop("headers", None)
    return await _get_litellm().aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def _transport_module():
    # Resolve completion fns at call time so tests can patch dspy.clients.lm.transport.alitellm_*.
    import dspy.clients.lm.transport as transport_module

    return transport_module


_MODEL_TYPE_ROUTES: dict[str, ModelTypeRoute] = {
    "chat": ModelTypeRoute(
        completion_fn_name=LitellmCompletionName.CHAT,
        to_openai_request=to_openai_chat_request,
        to_lm_response=completion_to_lm_response,
    ),
    "text": ModelTypeRoute(
        completion_fn_name=LitellmCompletionName.TEXT,
        to_openai_request=to_openai_text_request,
        to_lm_response=completion_to_lm_response,
    ),
    "responses": ModelTypeRoute(
        completion_fn_name=LitellmCompletionName.RESPONSES,
        to_openai_request=to_openai_responses_request,
        to_lm_response=responses_to_lm_response,
    ),
}


def _unsupported_model_type_error(*, model_type: str, model: str, provider: str) -> LMConfigurationError:
    return LMConfigurationError(
        f"Unsupported model_type {model_type!r} for `dspy.clients.lm.LM`.",
        model=model,
        provider=provider,
    )


def _route_for_model_type(model_type: str, *, lm: Any) -> ModelTypeRoute:
    provider = split_provider_model(lm.model)[0]
    route = _MODEL_TYPE_ROUTES.get(model_type)
    if route is None:
        raise _unsupported_model_type_error(model_type=model_type, model=lm.model, provider=provider)
    return route


def completion_fn_for_model_type(model_type: str, *, lm: Any) -> LitellmCompletionFn:
    route = _route_for_model_type(model_type, lm=lm)
    return getattr(_transport_module(), route.completion_fn_name.value)


def provider_request_for_model_type(model_type: str, request: LMRequest, lm: Any) -> dict[str, Any]:
    route = _route_for_model_type(model_type, lm=lm)
    if lm.use_developer_role and model_type == "responses":
        request = request.model_copy(
            update={
                "messages": [
                    message.model_copy(update={"role": "developer"}) if message.role == "system" else message
                    for message in request.messages
                ]
            }
        )
    request = request.model_copy(update={"config": merge_lm_request_config(lm=lm, config=request.config)})
    provider_request = route.to_openai_request(request)
    lm_defaults = {
        key: value for key, value in lm.kwargs.items() if value is not None and key not in {"cache", "reasoning"}
    }
    return {**lm_defaults, **provider_request}


def lm_response_for_model_type(model_type: str, response: Any, request: LMRequest, lm: Any) -> LMResponse:
    route = _route_for_model_type(model_type, lm=lm)
    lm_response = route.to_lm_response(response, request)
    return lm_response.model_copy(
        update={
            "model": getattr(response, "model", None) or lm_response.model,
            "usage": usage_from_response(response),
            "cost": cost_from_response(response),
            "provider_response": response,
        }
    )
