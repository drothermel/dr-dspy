from __future__ import annotations

from typing import Annotated, Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMBasePart(BaseModel):
    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class LMSourcePart(LMBasePart):
    media_type: str
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> LMSourcePart:
        _validate_one_source(self)
        return self


class LMTextPart(LMBasePart):
    type: Literal["text"] = "text"
    text: str


class LMImagePart(LMSourcePart):
    type: Literal["image"] = "image"
    media_type: str = "image/png"
    detail: Literal["low", "high", "auto"] | None = None


class LMAudioPart(LMSourcePart):
    type: Literal["audio"] = "audio"
    media_type: str = "audio/wav"


class LMVideoPart(LMSourcePart):
    type: Literal["video"] = "video"
    media_type: str = "video/mp4"


class LMDocumentPart(LMBasePart):
    type: Literal["document"] = "document"
    media_type: str = "application/pdf"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None
    source: dict[str, Any] | None = None
    citations: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    context: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> LMDocumentPart:
        has_media_source = any(value is not None for value in (self.data, self.url, self.file_id, self.path))
        if self.source is not None and has_media_source:
            raise ValueError("LMDocumentPart accepts either source or one of data, url, file_id, or path, not both.")
        if self.source is None:
            _validate_one_source(self)
        elif not self.source:
            raise ValueError("LMDocumentPart.source must be non-empty when provided.")
        return self


class LMBinaryPart(LMSourcePart):
    type: Literal["binary"] = "binary"
    media_type: str = "application/octet-stream"
    filename: str | None = None


class LMToolCallPart(LMBasePart):
    type: Literal["tool_call"] = "tool_call"
    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)


class LMToolResultPart(LMBasePart):
    type: Literal["tool_result"] = "tool_result"
    call_id: str | None = None
    name: str | None = None
    content: list[LMPart] = Field(default_factory=list)
    is_error: bool = False
    provider_data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_content(cls, data: Any) -> Any:
        if isinstance(data, dict) and "content" in data:
            data = dict(data)
            content = data["content"]
            if content is None:
                data["content"] = []
            elif isinstance(content, list):
                data["content"] = [_coerce_part(item) for item in content]
            else:
                data["content"] = [_coerce_part(content)]
        return data


class LMThinkingPart(LMBasePart):
    type: Literal["thinking"] = "thinking"
    text: str
    redacted: bool = False


class LMCitationPart(LMBasePart):
    type: Literal["citation"] = "citation"
    text: str | None = None
    title: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def validate_has_content(self) -> LMCitationPart:
        if self.text is None and self.title is None and (self.url is None):
            raise ValueError("LMCitationPart requires at least one of text, title, or url.")
        return self


class LMRefusalPart(LMBasePart):
    type: Literal["refusal"] = "refusal"
    text: str


class LMOpaquePart(LMBasePart):
    type: Literal["opaque"] = "opaque"
    block: dict[str, Any]


LMPart = Annotated[
    LMTextPart
    | LMImagePart
    | LMAudioPart
    | LMVideoPart
    | LMDocumentPart
    | LMBinaryPart
    | LMToolCallPart
    | LMToolResultPart
    | LMThinkingPart
    | LMCitationPart
    | LMRefusalPart
    | LMOpaquePart,
    Field(discriminator="type"),
]


def _validate_one_source(part: Any) -> None:
    sources = {
        name: getattr(part, name) for name in ("data", "url", "file_id", "path") if getattr(part, name) is not None
    }
    class_name = type(part).__name__
    if len(sources) != 1:
        raise ValueError(f"{class_name} requires exactly one of data, url, file_id, or path.")
    name, value = next(iter(sources.items()))
    if isinstance(value, str) and (not value):
        raise ValueError(f"{class_name}.{name} must be non-empty.")


def _coerce_part(value: Any) -> LMPart:
    if isinstance(
        value,
        (
            LMTextPart,
            LMImagePart,
            LMAudioPart,
            LMVideoPart,
            LMDocumentPart,
            LMBinaryPart,
            LMToolCallPart,
            LMToolResultPart,
            LMThinkingPart,
            LMCitationPart,
            LMRefusalPart,
            LMOpaquePart,
        ),
    ):
        return value
    if isinstance(value, str):
        return LMTextPart(text=value)
    if isinstance(value, dict) and "type" in value:
        return pydantic.TypeAdapter(LMPart).validate_python(value)
    raise TypeError(f"Cannot convert {type(value)!r} to an LMPart.")
