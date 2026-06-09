from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from dr_llm.backends.models import BackendRequest, BackendResponse
from dr_llm.llm import CallMode, EffortSpec, Message, ProviderName, SamplingControls

from dspy.core.types import LMOutput, LMRequest, LMResponse, LMUsage
from dspy.core.types.config import LMConfig, merge_lm_request_config
from dspy.core.types.parts import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMImagePart,
    LMPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMVideoPart,
)
from dspy.utils.exceptions import LMUnsupportedFeatureError

if TYPE_CHECKING:
    from dr_llm.llm.response import LlmResponse

_MessageRole = Literal["system", "user", "assistant"]
_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})

_UNSUPPORTED_PART_TYPES = (
    LMImagePart,
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMVideoPart,
    LMToolCallPart,
)


def split_provider_model(model: str) -> tuple[str, str]:
    if "/" in model:
        provider, rest = model.split("/", 1)
        return provider, rest
    return ProviderName.OPENAI, model


def probe_backend_request(lm: Any, *, mode: CallMode = CallMode.api) -> BackendRequest:
    provider_name, model_name = split_provider_model(lm.model)
    try:
        provider = ProviderName(provider_name)
    except ValueError as exc:
        raise LMUnsupportedFeatureError(
            f"Unsupported dr-llm provider {provider_name!r}.",
            model=lm.model,
        ) from exc
    merged = merge_lm_request_config(lm, LMConfig())
    return BackendRequest(
        provider=provider,
        model=model_name,
        mode=mode,
        messages=[Message(role="user", content="")],
        max_tokens=merged.max_tokens,
        effort=_effort_from_config(merged),
        sampling=_sampling_from_config(merged),
    )


def _reject_unsupported_request(request: LMRequest) -> None:
    if request.tools:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support tool calling.",
            model=request.model,
            features=["tools"],
        )
    if request.config.response_format is not None:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support structured response_format.",
            model=request.model,
            features=["response_format"],
        )
    for message in request.messages:
        if message.role not in _ALLOWED_MESSAGE_ROLES:
            raise LMUnsupportedFeatureError(
                f"dr-llm backends v1 do not support message role {message.role!r}.",
                model=request.model,
                features=["role"],
            )
        _reject_unsupported_parts(message.parts, model=request.model)


def _unsupported_features_for_part(part: LMPart) -> list[str]:
    if isinstance(part, LMToolCallPart):
        return ["tools"]
    return ["multimodal"]


def _reject_unsupported_parts(parts: list[LMPart], *, model: str) -> None:
    for part in parts:
        if isinstance(part, _UNSUPPORTED_PART_TYPES):
            raise LMUnsupportedFeatureError(
                f"dr-llm backends v1 do not support message part type {type(part).__name__}.",
                model=model,
                features=_unsupported_features_for_part(part),
            )


def _parts_to_content(parts: list[LMPart], *, model: str) -> str:
    texts: list[str] = []
    for part in parts:
        if isinstance(part, (LMTextPart, LMThinkingPart)):
            if part.text:
                texts.append(part.text)
        else:
            _reject_unsupported_parts([part], model=model)
    return "".join(texts)


def _sampling_from_config(config: LMConfig) -> SamplingControls | None:
    temperature = config.temperature
    top_p = config.top_p
    if temperature is None and top_p is None:
        return None
    return SamplingControls(temperature=temperature, top_p=top_p)


def _effort_from_config(config: LMConfig) -> EffortSpec:
    reasoning = config.reasoning
    if reasoning is not None and reasoning.effort is not None:
        try:
            return EffortSpec(reasoning.effort.lower())
        except ValueError:
            pass
    return EffortSpec.NA


def lm_request_to_backend_request(
    request: LMRequest,
    *,
    lm: Any,
    mode: CallMode = CallMode.api,
) -> BackendRequest:
    _reject_unsupported_request(request)
    merged = merge_lm_request_config(lm, request.config)
    provider_name, model_name = split_provider_model(request.model)
    try:
        provider = ProviderName(provider_name)
    except ValueError as exc:
        raise LMUnsupportedFeatureError(
            f"Unsupported dr-llm provider {provider_name!r}.",
            model=request.model,
        ) from exc
    messages = [
        Message(
            role=cast("_MessageRole", message.role),
            content=_parts_to_content(message.parts, model=request.model),
        )
        for message in request.messages
    ]
    return BackendRequest(
        provider=provider,
        model=model_name,
        mode=mode,
        messages=messages,
        max_tokens=merged.max_tokens,
        effort=_effort_from_config(merged),
        reasoning=None,
        sampling=_sampling_from_config(merged),
        metadata=dict(request.metadata),
    )


def _usage_to_lm_usage(usage: Any) -> LMUsage:
    if usage is None:
        return LMUsage()
    if isinstance(usage, LMUsage):
        return usage
    return LMUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        reasoning_tokens=getattr(usage, "reasoning_tokens", None),
    )


def _cost_to_float(cost: Any) -> float | None:
    if cost is None:
        return None
    total = getattr(cost, "total_cost_usd", None)
    if total is not None:
        return float(total)
    return None


def backend_response_to_lm_response(
    response: BackendResponse,
    *,
    request: LMRequest,
) -> LMResponse:
    parts: list[LMPart] = []
    if response.text:
        parts.append(LMTextPart(text=response.text))
    if response.reasoning:
        parts.append(LMThinkingPart(text=response.reasoning))
    if response.reasoning_details:
        for item in response.reasoning_details:
            text = item.get("text") if isinstance(item, dict) else None
            if isinstance(text, str) and text:
                parts.append(LMThinkingPart(text=text))
    finish_reason = response.finish_reason
    provider_data: dict[str, Any] = {}
    if response.source is not None:
        provider_data["source"] = response.source
    if response.sample_id is not None:
        provider_data["sample_id"] = response.sample_id
    if response.request_fingerprint is not None:
        provider_data["request_fingerprint"] = response.request_fingerprint
    warnings = (
        [warning.model_dump(mode="json") for warning in response.warnings] if response.warnings else []
    )
    if warnings:
        provider_data["warnings"] = warnings
    return LMResponse(
        model=request.model,
        outputs=[
            LMOutput(
                parts=parts,
                finish_reason=finish_reason,
                truncated=finish_reason == "length",
                provider_data=provider_data,
            )
        ],
        usage=_usage_to_lm_usage(response.usage),
        cost=_cost_to_float(response.cost),
        provider_response=response,
    )


def llm_response_to_backend_response(response: LlmResponse, **meta: Any) -> BackendResponse:
    payload = response.model_dump()
    return BackendResponse(**payload, **meta)
