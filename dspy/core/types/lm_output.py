from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dspy.core.types.parts import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMDocumentPart,
    LMImagePart,
    LMPart,
    LMRefusalPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMVideoPart,
)
from dspy.core.types.parts.models import _coerce_part
from dspy.core.types.parts.serialize import _part_to_value


class LMOutput(BaseModel):
    parts: list[LMPart] = Field(default_factory=list)
    finish_reason: str | None = None
    truncated: bool = False
    logprobs: Any | None = None
    provider_output: Any | None = Field(default=None, exclude=True)
    provider_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_parts(cls, data: Any) -> Any:
        if isinstance(data, dict) and "parts" in data:
            data = dict(data)
            data["parts"] = [_coerce_part(part) for part in data["parts"]]
        return data

    @property
    def text(self) -> str | None:
        texts = [part.text for part in self.parts if isinstance(part, LMTextPart)]
        return "".join(texts) if texts else None

    @property
    def reasoning_content(self) -> str | None:
        texts = [part.text for part in self.parts if isinstance(part, LMThinkingPart)]
        return "".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[LMToolCallPart]:
        return [part for part in self.parts if isinstance(part, LMToolCallPart)]

    @property
    def citations(self) -> list[LMCitationPart]:
        return [part for part in self.parts if isinstance(part, LMCitationPart)]

    @property
    def images(self) -> list[LMImagePart]:
        return [part for part in self.parts if isinstance(part, LMImagePart)]

    @property
    def audio(self) -> list[LMAudioPart]:
        return [part for part in self.parts if isinstance(part, LMAudioPart)]

    @property
    def videos(self) -> list[LMVideoPart]:
        return [part for part in self.parts if isinstance(part, LMVideoPart)]

    @property
    def documents(self) -> list[LMDocumentPart]:
        return [part for part in self.parts if isinstance(part, LMDocumentPart)]

    @property
    def binaries(self) -> list[LMBinaryPart]:
        return [part for part in self.parts if isinstance(part, LMBinaryPart)]

    @property
    def refusal(self) -> str | None:
        refusals = [part.text for part in self.parts if isinstance(part, LMRefusalPart)]
        return "".join(refusals) if refusals else None

    def to_value(self) -> Any:
        values = [_part_to_value(part) for part in self.parts]
        values = [value for value in values if value is not None]
        if len(values) == 1 and isinstance(values[0], str) and (self.logprobs is None):
            return values[0]
        return values

    def to_output_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text}
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            data["tool_calls"] = [
                {"name": call.name, "args": dict(call.args), "id": call.id} for call in self.tool_calls
            ]
        if self.citations:
            data["citations"] = [citation.model_dump(exclude_none=True) for citation in self.citations]
        if self.logprobs is not None:
            data["logprobs"] = self.logprobs
        return data


def requires_output_dict(output: LMOutput) -> bool:
    return bool(
        output.logprobs is not None or output.reasoning_content is not None or output.tool_calls or output.citations
    )
