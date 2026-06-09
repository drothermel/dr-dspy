from __future__ import annotations

import json
import logging
import mimetypes
from typing import Any
from urllib.parse import urlparse

from dspy.clients.openai_format.media import get_value, model_dump, split_data_uri
from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMDocumentPart,
    LMImagePart,
    LMOpaquePart,
    LMPart,
    LMRefusalPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMVideoPart,
)
from dspy.core.types.parts.models import _coerce_part
from dspy.core.types.request import LMRequest
from dspy.core.types.response import LMOutput, LMResponse, LMUsage
from dspy.errors import LMInvalidRequestError

logger = logging.getLogger(__name__)


def completion_to_lm_response(response: Any, request: LMRequest) -> LMResponse:
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
    message = get_value(choice, "message")
    parts = []
    if message is not None:
        reasoning = get_value(message, "reasoning_content")
        if reasoning:
            parts.append(LMThinkingPart(text=str(reasoning)))
        content = get_value(message, "content")
        if content:
            parts.extend(message_content_to_parts(content))
        parts.extend(provider_tool_call_to_part(tool_call) for tool_call in (get_value(message, "tool_calls") or []))
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
    if isinstance(content, list):
        return parts_from_openai_content(content)
    return [LMTextPart(text=str(content))]


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
    message = get_value(choice, "message")
    if message is None:
        return []
    provider_specific_fields = get_value(message, "provider_specific_fields")
    if provider_specific_fields is None:
        return []
    if not isinstance(provider_specific_fields, dict):
        raise LMInvalidRequestError(
            "provider_specific_fields must be a dict when present on an OpenAI-format choice message."
        )
    citations_data = provider_specific_fields.get("citations")
    if citations_data is None:
        return []
    if not isinstance(citations_data, list):
        raise LMInvalidRequestError("citations must be a list when present in provider_specific_fields.")
    citations = []
    for item in citations_data:
        citations.extend(item if isinstance(item, list) else [item])
    return [citation_to_part(citation) for citation in citations]


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


def _mapping_from_dump(value: Any) -> dict[str, Any]:
    data = model_dump(value)
    return data if isinstance(data, dict) else {}


def output_audio_to_part(value: Any) -> LMAudioPart:
    data = _mapping_from_dump(value)
    audio_val = data.get("audio")
    audio = audio_val if isinstance(audio_val, dict) else data
    source = audio.get("url")
    b64_data = audio.get("data") or audio.get("b64_json")
    file_id = audio.get("file_id")
    media_type = audio.get("media_type") or audio.get("mime_type") or "audio/wav"
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
    data = _mapping_from_dump(value)
    file_val = data.get("file")
    file = file_val if isinstance(file_val, dict) else data
    source = file.get("url")
    b64_data = file.get("file_data") or file.get("data")
    file_id = file.get("file_id") or file.get("id")
    filename = file.get("filename")
    media_type = file.get("media_type") or file.get("mime_type") or "application/octet-stream"
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
            except Exception:
                logger.debug("Skipping usage attribute %r", key, exc_info=True)
                continue
            if value is not None and (not callable(value)):
                data[key] = value
        usage = data
    return LMUsage(**dict(usage))


def parts_from_openai_content(content: Any) -> list[LMPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [LMTextPart(text=content)]
    if not isinstance(content, list):
        return [_coerce_part(content)]
    parts: list[LMPart] = []
    for item in content:
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type == "text":
            parts.append(LMTextPart(text=item.get("text", ""), metadata=item.get("metadata", {}) or {}))
        elif item_type == "image_url":
            image = item.get("image_url", {})
            if not isinstance(image, dict):
                raise TypeError("Image content block must be a mapping.")
            url = image.get("url")
            if url is None:
                raise ValueError("Image content block requires url.")
            parts.append(_image_source_to_part(url))
        elif item_type == "input_audio":
            audio = item.get("input_audio", {})
            parts.append(_audio_dict_to_part(audio))
        elif item_type == "file":
            parts.append(_binary_dict_to_part(item.get("file", {})))
        elif item_type == "document":
            parts.append(_document_dict_to_part(item))
        elif item_type == "video":
            video = item.get("video", {})
            parts.append(_media_dict_to_video_part(video))
        elif isinstance(item, dict):
            parts.append(LMOpaquePart(block=item))
        else:
            parts.append(_coerce_part(item))
    return parts


def _image_source_to_part(source: str) -> LMImagePart:
    if not isinstance(source, str):
        raise TypeError("Image URL must be a string.")
    if source.startswith("data:"):
        media_type, data = split_data_uri(source)
        return LMImagePart(data=data, media_type=media_type)
    media_type = mimetypes.guess_type(urlparse(source).path)[0] or "image/png"
    return LMImagePart(url=source, media_type=media_type)


def _audio_dict_to_part(audio: dict[str, Any]) -> LMAudioPart:
    if not isinstance(audio, dict):
        raise TypeError("Audio content block must be a mapping.")
    audio_format = audio.get("format") or "wav"
    if not isinstance(audio_format, str):
        raise TypeError("Audio format must be a string.")
    media_type = audio_format if "/" in audio_format else f"audio/{audio_format}"
    if audio.get("data") is not None:
        data = audio["data"]
        if isinstance(data, str) and data.startswith("data:"):
            media_type, data = split_data_uri(data)
        return LMAudioPart(data=data, media_type=media_type)
    if audio.get("url") is not None:
        return LMAudioPart(url=audio["url"], media_type=media_type)
    if audio.get("file_id") is not None:
        return LMAudioPart(file_id=audio["file_id"], media_type=media_type)
    if audio.get("path") is not None:
        return LMAudioPart(path=audio["path"], media_type=media_type)
    raise ValueError("Audio content block requires data, url, file_id, or path.")


def _binary_dict_to_part(file: dict[str, Any]) -> LMBinaryPart:
    if file.get("file_data") is not None:
        media_type, data = split_data_uri(file["file_data"])
        return LMBinaryPart(data=data, media_type=media_type, filename=file.get("filename"))
    if file.get("data") is not None:
        media_type, data = split_data_uri(file["data"])
        return LMBinaryPart(data=data, media_type=media_type, filename=file.get("filename"))
    if file.get("file_id") is not None:
        return LMBinaryPart(file_id=file["file_id"], filename=file.get("filename"))
    raise ValueError("Binary content block requires data, file_data, or file_id.")


def _document_dict_to_part(item: dict[str, Any]) -> LMDocumentPart:
    title = item.get("title")
    context = item.get("context")
    citations = item.get("citations") or {}
    media_type = item.get("media_type") or "application/pdf"
    for source_key in ("data", "url", "file_id", "path"):
        if item.get(source_key) is not None:
            return LMDocumentPart(
                **{source_key: item[source_key]},
                media_type=media_type,
                title=title,
                context=context,
            )
    source = item.get("source")
    if isinstance(source, dict):
        return LMDocumentPart(
            source=source,
            citations=citations if isinstance(citations, dict) else {},
            title=title,
            context=context,
        )
    if isinstance(source, str):
        source_kwargs = _media_source_kwargs(source, default_media_type=media_type)
        return LMDocumentPart(
            data=source_kwargs.get("data"),
            url=source_kwargs.get("url"),
            file_id=source_kwargs.get("file_id"),
            media_type=source_kwargs.get("media_type", media_type),
            title=title,
            context=context,
        )
    raise ValueError("Document content block requires source.")


def _media_dict_to_video_part(video: dict[str, Any]) -> LMVideoPart:
    if video.get("data") is not None:
        data = video["data"]
        if isinstance(data, str) and data.startswith("data:"):
            media_type, data = split_data_uri(data)
        else:
            media_type = video.get("media_type") or "video/mp4"
        return LMVideoPart(data=data, media_type=media_type)
    if video.get("url") is not None:
        return LMVideoPart(url=video["url"], media_type=video.get("media_type") or "video/mp4")
    if video.get("file_id") is not None:
        return LMVideoPart(file_id=video["file_id"], media_type=video.get("media_type") or "video/mp4")
    if video.get("path") is not None:
        return LMVideoPart(path=video["path"], media_type=video.get("media_type") or "video/mp4")
    raise ValueError("Video content block requires data, url, file_id, or path.")


def _media_source_kwargs(source: str, *, default_media_type: str) -> dict[str, str]:
    if source.startswith("data:"):
        media_type, data = split_data_uri(source)
        return {"data": data, "media_type": media_type}
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        media_type = mimetypes.guess_type(parsed.path)[0] or default_media_type
        return {"url": source, "media_type": media_type}
    return {"file_id": source, "media_type": default_media_type}


def tool_calls_from_openai_chat(tool_calls: list[Any]) -> list[LMToolCallPart]:
    return [provider_tool_call_to_part(tool_call) for tool_call in tool_calls]
