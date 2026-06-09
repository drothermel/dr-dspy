# Internal: import from dspy.clients.openai_format.serialize only in tests and spine-adjacent code.
from __future__ import annotations

import os
from typing import Any, Literal

import pydantic

from dspy.clients.openai_format.binary import binary_to_openai
from dspy.clients.openai_format.media import (
    media_format,
    media_source,
    media_type_for_path,
    part_text,
    read_path_base64,
)
from dspy.clients.openai_format.reasoning_models import is_openai_reasoning_model
from dspy.clients.openai_format.tool_calls import tool_call_part_to_openai
from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMConfig,
    LMDocumentPart,
    LMImagePart,
    LMOpaquePart,
    LMRefusalPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMToolChoice,
    LMToolResultPart,
    LMToolSpec,
    LMVideoPart,
)
from dspy.errors import LMUnsupportedFeatureError


def parts_to_openai_content(parts: list[Any]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and isinstance(parts[0], LMTextPart):
        return parts[0].text
    blocks: list[dict[str, Any]] = []
    for part in parts:
        blocks.extend(part_to_openai_blocks(part))
    return blocks


def part_to_openai_blocks(part: Any) -> list[dict[str, Any]]:
    if isinstance(part, LMOpaquePart):
        return [dict(part.block)]
    if isinstance(part, LMTextPart):
        return [{"type": "text", "text": part.text}]
    if isinstance(part, LMImagePart):
        return [image_to_openai(part)]
    if isinstance(part, LMDocumentPart):
        return document_to_openai_blocks(part)
    if isinstance(part, LMAudioPart):
        return [audio_to_openai(part)]
    if isinstance(part, LMVideoPart):
        return [video_to_openai(part)]
    if isinstance(part, LMBinaryPart):
        return [binary_to_openai(part)]
    if isinstance(part, LMThinkingPart):
        return [{"type": "text", "text": part.text}]
    if isinstance(part, LMCitationPart):
        citation = " ".join(value for value in (part.title, part.text, part.url) if value)
        return [{"type": "text", "text": citation}]
    if isinstance(part, LMRefusalPart):
        return [{"type": "text", "text": part.text}]
    if isinstance(part, LMToolResultPart):
        return part_to_openai_blocks(LMTextPart(text="".join(part_text(value) for value in part.content)))
    if isinstance(part, LMToolCallPart):
        raise LMUnsupportedFeatureError(
            "OpenAI-format tool calls must be serialized at the message layer, not via part_to_openai_blocks.",
            features=["tools"],
        )
    raise LMUnsupportedFeatureError(
        f"OpenAI-format serialization does not support message part type {type(part).__name__}.",
        features=["part_type"],
    )


def image_to_openai(image: LMImagePart) -> dict[str, Any]:
    image_url: dict[str, Any] = {"url": media_source(image)}
    if image.detail is not None:
        image_url["detail"] = image.detail
    return {"type": "image_url", "image_url": image_url}


def audio_to_openai(audio: LMAudioPart) -> dict[str, Any]:
    if audio.data is not None:
        data = audio.data
        media_type = audio.media_type
    elif audio.path is not None:
        data = read_path_base64(audio.path)
        media_type = media_type_for_path(audio.path, fallback=audio.media_type)
    else:
        raise ValueError("OpenAI-format audio input requires base64 `data` or local `path`.")
    return {"type": "input_audio", "input_audio": {"data": data, "format": media_format(media_type)}}


def document_to_openai_blocks(document: LMDocumentPart) -> list[dict[str, Any]]:
    block: dict[str, Any] = {"type": "document"}
    if document.source is not None:
        block["source"] = document.source
    else:
        block["source"] = media_source(document)
        block["media_type"] = document.media_type
    if document.citations:
        block["citations"] = document.citations
    if document.title is not None:
        block["title"] = document.title
    if document.context is not None:
        block["context"] = document.context
    return [block]


def video_to_openai(video: LMVideoPart) -> dict[str, Any]:
    filename = os.path.basename(video.path) if video.path is not None else None
    return binary_to_openai(
        LMBinaryPart(
            data=video.data,
            url=video.url,
            file_id=video.file_id,
            path=video.path,
            media_type=video.media_type,
            filename=filename,
        )
    )


def tool_to_openai(tool: LMToolSpec) -> dict[str, Any]:
    data = {"type": "function", "function": {"name": tool.name, "parameters": tool.parameters}}
    if tool.description is not None:
        data["function"]["description"] = tool.description
    data.update(tool.provider_data)
    return data


def tool_choice_to_openai(choice: LMToolChoice) -> dict[str, Any]:
    if choice.allowed:
        if len(choice.allowed) != 1 or choice.mode not in {"required", "auto"}:
            raise ValueError(
                "OpenAI-format tool_choice only supports constraining to a single allowed tool with mode 'required' or 'auto'."
            )
        data: dict[str, Any] = {"tool_choice": {"type": "function", "function": {"name": choice.allowed[0]}}}
    else:
        data: dict[str, Any] = {"tool_choice": choice.mode}
    if choice.parallel is not None:
        data["parallel_tool_calls"] = choice.parallel
    return data


def assistant_tool_call_to_openai(call: LMToolCallPart) -> dict[str, Any]:
    return tool_call_part_to_openai(call, include_provider_data=True)


def tool_result_to_openai(result: LMToolResultPart) -> dict[str, Any]:
    return {"content": parts_to_openai_content(result.content)}


def config_to_provider_kwargs(
    config: LMConfig,
    *,
    model: str | None = None,
    endpoint: Literal["chat", "responses", "text"] = "chat",
) -> dict[str, Any]:
    if endpoint == "text":
        return text_config_kwargs(config)
    if endpoint == "responses":
        return responses_config_kwargs(config, model=model)
    return common_config_kwargs(config, model=model, endpoint=endpoint)


def common_config_kwargs(config: LMConfig, *, model: str | None = None, endpoint: str = "chat") -> dict[str, Any]:
    data = dict(config.extensions)
    _validate_openai_reasoning_temperature(config, model=model, endpoint=endpoint)
    for key in ("temperature", "top_p"):
        value = getattr(config, key)
        if value is not None:
            data[key] = value
    if config.max_tokens is not None:
        token_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
        data[token_key] = config.max_tokens
    if config.stop:
        data["stop"] = config.stop
    if config.logprobs is not None:
        data["logprobs"] = config.logprobs
    if config.n is not None:
        data["n"] = config.n
    if config.response_format is not None:
        data["response_format"] = config.response_format
    if config.reasoning is not None:
        data.update(reasoning_to_chat_kwargs(config.reasoning))
    if config.prompt_cache is not None:
        data.update(prompt_cache_to_kwargs(config.prompt_cache))
    return data


def responses_config_kwargs(config: LMConfig, *, model: str | None = None) -> dict[str, Any]:
    data = dict(config.extensions) if config.extensions else {}
    _validate_openai_reasoning_temperature(config, model=model, endpoint="responses")
    for key in ("temperature", "top_p"):
        value = getattr(config, key)
        if value is not None:
            data[key] = value
    if config.max_tokens is not None:
        data["max_output_tokens"] = config.max_tokens
    if config.n is not None:
        data["n"] = config.n
    if config.logprobs is not None:
        data["logprobs"] = config.logprobs
    if config.stop:
        data["stop"] = config.stop
    if config.reasoning is not None:
        data.update(reasoning_to_responses_kwargs(config.reasoning))
    if config.prompt_cache is not None:
        data.update(prompt_cache_to_kwargs(config.prompt_cache))
    if config.response_format is not None:
        text = data.pop("text", {})
        data["text"] = {**text, "format": response_format_to_responses(config.response_format)}
    return data


def text_config_kwargs(config: LMConfig) -> dict[str, Any]:
    data = dict(config.extensions)
    for key in ("temperature", "max_tokens", "top_p"):
        value = getattr(config, key)
        if value is not None:
            data[key] = value
    if config.stop:
        data["stop"] = config.stop
    if config.logprobs is not None:
        data["logprobs"] = config.logprobs
    if config.n is not None:
        data["n"] = config.n
    return data


def reasoning_to_chat_kwargs(reasoning: Any) -> dict[str, Any]:
    data = {}
    if reasoning.effort is not None:
        data["reasoning_effort"] = reasoning.effort
    return data


def reasoning_to_responses_kwargs(reasoning: Any) -> dict[str, Any]:
    data = {}
    if reasoning.effort is not None:
        data["effort"] = reasoning.effort
    if reasoning.summary is not None:
        data["summary"] = reasoning.summary
    return {"reasoning": data} if data else {}


def _validate_openai_reasoning_temperature(config: LMConfig, *, model: str | None, endpoint: str) -> None:
    if not is_openai_reasoning_model(model):
        return
    effort = getattr(config.reasoning, "effort", None) if config.reasoning is not None else None
    if effort in {None, "none"}:
        return
    if config.temperature in {None, 1}:
        return
    raise LMUnsupportedFeatureError(
        "OpenAI reasoning models only support the default temperature when reasoning effort is active. Use temperature=None or temperature=1, or set reasoning_effort='none'.",
        model=model,
        provider="openai",
        features=["temperature", "reasoning"],
        issues=[f"{endpoint} request used reasoning effort {effort!r} with temperature={config.temperature!r}."],
    )


def _uses_max_completion_tokens(model: str | None) -> bool:
    return is_openai_reasoning_model(model)


def prompt_cache_to_kwargs(cache: Any) -> dict[str, Any]:
    data = {}
    if cache.key is not None:
        data["prompt_cache_key"] = cache.key
    if cache.enabled is False:
        data["prompt_cache"] = False
    return data


def response_format_to_responses(value: Any) -> Any:
    if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
        return {"name": value.__name__, "type": "json_schema", "schema": value.model_json_schema()}
    return value


def responses_tool_output_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
            else str(block)
            for block in content
        )
    return str(content)
