"""Normalized LM content part models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMBasePart(BaseModel):
    """A single content item in an LM message or output."""

    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class LMSourcePart(LMBasePart):
    """Content from exactly one provider-addressable source."""

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
    """Text content."""

    type: Literal["text"] = "text"
    text: str


class LMImagePart(LMSourcePart):
    """Image content from data, a URL, a file ID, or a local path."""

    type: Literal["image"] = "image"
    media_type: str = "image/png"
    detail: Literal["low", "high", "auto"] | None = None


class LMAudioPart(LMSourcePart):
    """Audio content from data, a URL, a file ID, or a local path."""

    type: Literal["audio"] = "audio"
    media_type: str = "audio/wav"


class LMVideoPart(LMSourcePart):
    """Video content from data, a URL, a file ID, or a local path."""

    type: Literal["video"] = "video"
    media_type: str = "video/mp4"


class LMDocumentPart(LMBasePart):
    """Semantic source/document content, optionally citation-enabled.

    Documents are source material: text, PDFs, reports, contracts, or other
    provider-addressable evidence that benefits from title/context/citation
    semantics. Use `LMBinaryPart` for opaque attachments or arbitrary bytes.
    """

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
    """Opaque binary content from data, a URL, a file ID, or a local path."""

    type: Literal["binary"] = "binary"
    media_type: str = "application/octet-stream"
    filename: str | None = None


class LMToolCallPart(LMBasePart):
    """A model request to call a tool.

    Use `ToolCall(...)` as a shorter alias when constructing
    assistant messages by hand.

    Args:
        id: Provider call ID, when the backend uses one.
        name: Name of the tool to call.
        args: JSON-like arguments for the tool.
        provider_data: Raw provider fields to keep with the tool call.

    Examples:
        ```python
        from dspy.core.types import Assistant, ToolCall

        assistant = Assistant(
            ToolCall(
                id="call_1",
                name="search",
                args={"query": "DSPy"},
            )
        )
        ```
    """

    type: Literal["tool_call"] = "tool_call"
    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)


class LMToolResultPart(LMBasePart):
    """A tool execution result sent back to a model."""

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
    """Reasoning or thinking content returned by a model."""

    type: Literal["thinking"] = "thinking"
    text: str
    redacted: bool = False


class LMCitationPart(LMBasePart):
    """A source citation returned by a model."""

    type: Literal["citation"] = "citation"
    text: str | None = None
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_has_content(self) -> LMCitationPart:
        if self.text is None and self.title is None and self.url is None:
            raise ValueError("LMCitationPart requires at least one of text, title, or url.")
        return self


class LMRefusalPart(LMBasePart):
    """A model refusal."""

    type: Literal["refusal"] = "refusal"
    text: str


class LMOpaquePart(LMBasePart):
    """Provider-native content block preserved verbatim for round-trip."""

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
    if isinstance(value, str) and not value:
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
