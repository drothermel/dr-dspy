from __future__ import annotations

import json
from typing import Any

from dspy.clients.openai_format.media import get_value, model_dump, split_data_uri
from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMImagePart,
    LMOutput,
    LMRefusalPart,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMUsage,
)


def completion_to_lm_response(response: Any, request: LMRequest) -> LMResponse:
    """Convert an OpenAI Chat or text completion response into `LMResponse`."""
    choices = get_value(response, "choices", []) or []
    if not isinstance(choices, list):
        choices = []

    model = get_value(response, "model")
    if not isinstance(model, str):
        model = request.model

    response_id = get_value(response, "id")
    if not isinstance(response_id, str):
        response_id = None

    outputs = [choice_to_lm_output(choice) for choice in choices] if choices else [LMOutput(parts=[])]
    return LMResponse(
        model=model,
        outputs=outputs,
        usage=usage_from_response(response),
        response_id=response_id,
        provider_response=response,
    )


def choice_to_lm_output(choice: Any) -> LMOutput:
    """Convert one completion choice into one DSPy output candidate."""
    message = get_value(choice, "message")
    parts = []
    if message is not None:
        reasoning = get_value(message, "reasoning_content")
        if reasoning:
            parts.append(LMThinkingPart(text=str(reasoning)))
        content = get_value(message, "content")
        if content:
            parts.extend(message_content_to_parts(content))
        for tool_call in get_value(message, "tool_calls") or []:
            parts.append(provider_tool_call_to_part(tool_call))  # noqa: PERF401 dynamic typing/lint migration for scoped ty adoption
        parts.extend(extract_citations_from_choice(choice))
    else:
        text = get_value(choice, "text")
        if text:
            parts.extend(message_content_to_parts(text))
    finish_reason = get_value(choice, "finish_reason")
    return LMOutput(
        parts=parts,
        finish_reason=finish_reason,
        truncated=finish_reason == "length",
        logprobs=get_value(choice, "logprobs"),
        provider_output=choice,
    )


def responses_to_lm_response(response: Any, request: LMRequest) -> LMResponse:
    """Convert an OpenAI Responses object into `LMResponse`.

    The Responses API represents one assistant answer as a sequence of output
    items: messages, function calls, reasoning, binary artifacts, images, and refusals.
    DSPy stores those as typed parts on one `LMOutput`.
    """
    parts = []
    output_items = get_value(response, "output", []) or []
    if not isinstance(output_items, list):
        output_items = []
    for output_item in output_items:
        output_type = get_value(output_item, "type")
        if output_type == "message":
            for content_item in get_value(output_item, "content", []) or []:
                parts.extend(response_content_item_to_parts(content_item))
                parts.extend(responses_annotations_to_citations(content_item))
        elif output_type == "function_call":
            parts.append(responses_function_call_to_part(output_item))
        elif output_type in {"image", "output_image", "image_generation_call"}:
            parts.append(output_image_to_part(output_item))
        elif output_type in {"audio", "output_audio"}:
            parts.append(output_audio_to_part(output_item))
        elif output_type in {"file", "output_file"}:
            parts.append(output_file_to_part(output_item))
        elif output_type == "refusal":
            parts.append(refusal_to_part(output_item))
        elif output_type == "reasoning":
            for item in get_value(output_item, "content") or get_value(output_item, "summary") or []:
                text = get_value(item, "text")
                if text:
                    parts.append(LMThinkingPart(text=text))

    model = get_value(response, "model")
    if not isinstance(model, str):
        model = request.model

    response_id = get_value(response, "id")
    if not isinstance(response_id, str):
        response_id = None

    return LMResponse(
        model=model,
        outputs=[LMOutput(parts=parts, provider_output=response)],
        usage=usage_from_response(response),
        response_id=response_id,
        provider_response=response,
    )


def message_content_to_parts(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [LMTextPart(text=content)]
    if not isinstance(content, list):
        return [LMTextPart(text=str(content))]
    parts = []
    for item in content:
        parts.extend(response_content_item_to_parts(item))
    return parts


def response_content_item_to_parts(item: Any) -> list[Any]:
    item_type = get_value(item, "type")
    text = get_value(item, "text")
    if item_type in {"text", "output_text", "input_text"} or (text is not None and item_type is None):
        return [LMTextPart(text=text)]
    if item_type in {"refusal", "output_refusal"}:
        return [refusal_to_part(item)]
    if item_type in {"image", "output_image", "image_url"}:
        return [output_image_to_part(item)]
    if item_type in {"audio", "output_audio", "input_audio"}:
        return [output_audio_to_part(item)]
    if item_type in {"file", "output_file", "input_file"}:
        return [output_file_to_part(item)]
    if item_type in {"tool_call", "function_call"}:
        return [provider_tool_call_to_part(item)]
    return []


def provider_tool_call_to_part(tool_call: Any) -> LMToolCallPart:
    """Convert an OpenAI-shaped tool call into a DSPy tool-call part."""
    function = get_value(tool_call, "function", {})
    name = get_value(function, "name") or get_value(tool_call, "name")
    arguments = get_value(function, "arguments", get_value(tool_call, "arguments", "{}"))
    provider_data = model_dump(tool_call)
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except Exception as error:
        args = {}
        provider_data["raw_arguments"] = arguments
        provider_data["arguments_parse_error"] = str(error)
    call_id = get_value(tool_call, "call_id") or get_value(tool_call, "id")
    return LMToolCallPart(id=call_id, name=name or "", args=args, provider_data=provider_data)


def responses_function_call_to_part(output_item: Any) -> LMToolCallPart:
    """Convert one Responses function_call item into a DSPy tool-call part."""
    args = get_value(output_item, "arguments", {})
    provider_data = model_dump(output_item)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception as error:
            provider_data["raw_arguments"] = args
            provider_data["arguments_parse_error"] = str(error)
            args = {}
    return LMToolCallPart(
        id=get_value(output_item, "call_id"),
        name=get_value(output_item, "name", ""),
        args=args,
        provider_data=provider_data,
    )


def citation_to_part(citation: Any) -> LMCitationPart:
    if hasattr(citation, "model_dump"):
        citation = model_dump(citation)
    if not isinstance(citation, dict):
        citation = {"text": str(citation)}
    citation_fields = {"cited_text", "text", "supported_text", "document_title", "title", "url"}
    return LMCitationPart(
        text=citation.get("cited_text") or citation.get("text") or citation.get("supported_text"),
        title=citation.get("document_title") or citation.get("title"),
        url=citation.get("url"),
        metadata={key: value for key, value in citation.items() if key not in citation_fields},
    )


def extract_citations_from_choice(choice: Any) -> list[LMCitationPart]:
    try:
        message = get_value(choice, "message")
        provider_specific_fields = get_value(message, "provider_specific_fields", {}) or {}
        citations_data = provider_specific_fields.get("citations")
        if isinstance(citations_data, list):
            citations = []
            for item in citations_data:
                citations.extend(item if isinstance(item, list) else [item])
            return [citation_to_part(citation) for citation in citations]
    except Exception:
        return []
    return []


def responses_annotations_to_citations(content_item: Any) -> list[LMCitationPart]:
    return [citation_to_part(annotation) for annotation in get_value(content_item, "annotations", []) or []]


def output_image_to_part(value: Any) -> LMImagePart:
    data = model_dump(value)
    image_url = data.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    source = image_url or data.get("url")
    b64_data = data.get("b64_json") or data.get("data")
    file_id = data.get("file_id")
    media_type = data.get("media_type") or data.get("mime_type") or "image/png"
    detail = data.get("detail")
    if b64_data is not None:
        if isinstance(b64_data, str) and b64_data.startswith("data:"):
            media_type, b64_data = split_data_uri(b64_data)
        return LMImagePart(data=b64_data, media_type=media_type, detail=detail)
    if source is not None:
        return LMImagePart(url=source, media_type=media_type, detail=detail)
    if file_id is not None:
        return LMImagePart(file_id=file_id, media_type=media_type, detail=detail)
    raise ValueError("Provider image output did not include data, url, or file_id.")


def output_audio_to_part(value: Any) -> LMAudioPart:
    data = model_dump(value)
    audio = data.get("audio") if isinstance(data.get("audio"), dict) else data
    source = audio.get("url")  # ty:ignore[unresolved-attribute]
    b64_data = audio.get("data") or audio.get("b64_json")  # ty:ignore[unresolved-attribute]
    file_id = audio.get("file_id")  # ty:ignore[unresolved-attribute]
    media_type = audio.get("media_type") or audio.get("mime_type") or "audio/wav"  # ty:ignore[unresolved-attribute]
    if b64_data is not None:
        if isinstance(b64_data, str) and b64_data.startswith("data:"):
            media_type, b64_data = split_data_uri(b64_data)
        return LMAudioPart(data=b64_data, media_type=media_type)
    if source is not None:
        return LMAudioPart(url=source, media_type=media_type)
    if file_id is not None:
        return LMAudioPart(file_id=file_id, media_type=media_type)
    raise ValueError("Provider audio output did not include data, url, or file_id.")


def output_file_to_part(value: Any) -> LMBinaryPart:
    data = model_dump(value)
    file = data.get("file") if isinstance(data.get("file"), dict) else data
    source = file.get("url")  # ty:ignore[unresolved-attribute]
    b64_data = file.get("file_data") or file.get("data")  # ty:ignore[unresolved-attribute]
    file_id = file.get("file_id") or file.get("id")  # ty:ignore[unresolved-attribute]
    filename = file.get("filename")  # ty:ignore[unresolved-attribute]
    media_type = file.get("media_type") or file.get("mime_type") or "application/octet-stream"  # ty:ignore[unresolved-attribute]
    if b64_data is not None:
        if isinstance(b64_data, str) and b64_data.startswith("data:"):
            media_type, b64_data = split_data_uri(b64_data)
        return LMBinaryPart(data=b64_data, media_type=media_type, filename=filename)
    if source is not None:
        return LMBinaryPart(url=source, media_type=media_type, filename=filename)
    if file_id is not None:
        return LMBinaryPart(file_id=file_id, media_type=media_type, filename=filename)
    raise ValueError("Provider file output did not include data, url, or file_id.")


def refusal_to_part(value: Any) -> LMRefusalPart:
    text = get_value(value, "refusal") or get_value(value, "text") or get_value(value, "content") or str(value)
    return LMRefusalPart(text=str(text))


def cost_from_response(response: Any) -> float | None:
    hidden = getattr(response, "_hidden_params", None) or {}
    return hidden.get("response_cost") if isinstance(hidden, dict) else None


def usage_from_response(response: Any) -> LMUsage | None:
    """Convert provider usage objects or dictionaries into `LMUsage`."""
    usage = get_value(response, "usage")
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = model_dump(usage)
    elif not isinstance(usage, dict):
        data = {}
        for key in dir(usage):
            if key.startswith("_"):
                continue
            try:
                value = getattr(usage, key)
            except Exception:  # noqa: S112 dynamic typing/lint migration for scoped ty adoption
                continue
            if value is not None and not callable(value):
                data[key] = value
        usage = data
    return LMUsage(**dict(usage))
