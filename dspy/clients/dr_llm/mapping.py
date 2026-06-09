from __future__ import annotations

from typing import Any, Literal, cast

from dr_llm.backends.models import BackendRequest, BackendResponse
from dr_llm.llm import CallMode, EffortSpec, Message, SamplingControls
from pydantic import ValidationError

from dspy.clients.dr_llm.contract import reject_unsupported_merged_config
from dspy.clients.dr_llm.controls import DR_LLM_EXTENSION_KEY, DrLlmProviderControls, parse_dr_llm_controls
from dspy.clients.dr_llm.provider_name import parse_dr_llm_provider
from dspy.clients.model_id import split_provider_model
from dspy.core.types import LMConfig, LMOutput, LMRequest, LMResponse, LMUsage, merge_lm_request_config
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
from dspy.errors import LMConfigurationError, LMUnsupportedFeatureError

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


def probe_backend_request(lm: Any, *, mode: CallMode = CallMode.api) -> BackendRequest:
    provider_name, model_name = split_provider_model(lm.model)
    provider = parse_dr_llm_provider(provider_name, model=lm.model)
    merged = merge_lm_request_config(lm, LMConfig())
    reject_unsupported_merged_config(merged, model=lm.model)
    controls = _dr_llm_controls_from_config(merged, lm=lm, model=lm.model)
    return BackendRequest(
        provider=provider,
        model=model_name,
        mode=mode,
        messages=[Message(role="user", content="")],
        max_tokens=merged.max_tokens,
        effort=controls.effort or _effort_from_config(merged, model=lm.model),
        reasoning=controls.reasoning,
        sampling=_sampling_from_merged_config(merged, controls=controls, model=lm.model),
    )


def _reject_unsupported_request(request: LMRequest) -> None:
    if request.tools:
        raise LMUnsupportedFeatureError(
            "dr-llm backends v1 do not support tool calling.",
            model=request.model,
            features=["tools"],
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


def _dr_llm_controls_from_config(config: LMConfig, *, lm: Any, model: str) -> DrLlmProviderControls:
    default_controls = getattr(lm, "_dr_llm_controls", None)
    data = (
        default_controls.model_dump(mode="json", exclude_none=True)
        if isinstance(default_controls, DrLlmProviderControls)
        else {}
    )
    if DR_LLM_EXTENSION_KEY in config.extensions:
        override = parse_dr_llm_controls(config.extensions[DR_LLM_EXTENSION_KEY])
        data.update(override.model_dump(mode="json", exclude_unset=True))
    try:
        return parse_dr_llm_controls(data)
    except ValidationError as exc:
        raise LMConfigurationError("Invalid dr_llm provider controls.", model=model) from exc


def _sampling_from_merged_config(
    config: LMConfig,
    *,
    controls: DrLlmProviderControls,
    model: str,
) -> SamplingControls | None:
    del model
    if controls.sampling is not None:
        return controls.sampling
    return _sampling_from_config(config)


def _effort_from_config(config: LMConfig, *, model: str) -> EffortSpec:
    reasoning = config.reasoning
    if reasoning is not None and reasoning.effort is not None:
        try:
            return EffortSpec(reasoning.effort.lower())
        except ValueError as exc:
            raise LMConfigurationError(
                f"Invalid reasoning effort {reasoning.effort!r} for dr-llm backend.",
                model=model,
            ) from exc
    return EffortSpec.NA


def lm_request_to_backend_request(
    request: LMRequest,
    *,
    lm: Any,
    mode: CallMode = CallMode.api,
) -> BackendRequest:
    _reject_unsupported_request(request)
    merged = merge_lm_request_config(lm, request.config)
    reject_unsupported_merged_config(merged, model=request.model)
    controls = _dr_llm_controls_from_config(merged, lm=lm, model=request.model)
    provider_name, model_name = split_provider_model(request.model)
    provider = parse_dr_llm_provider(provider_name, model=request.model)
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
        effort=controls.effort or _effort_from_config(merged, model=request.model),
        reasoning=controls.reasoning,
        sampling=_sampling_from_merged_config(merged, controls=controls, model=request.model),
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
    warnings = [warning.model_dump(mode="json") for warning in response.warnings] if response.warnings else []
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
