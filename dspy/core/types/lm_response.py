from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dspy.core.types.lm_output import LMOutput, requires_output_dict
from dspy.core.types.parts import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMDocumentPart,
    LMImagePart,
    LMPart,
    LMTextPart,
    LMToolCallPart,
    LMVideoPart,
)
from dspy.core.types.usage import LMUsage


class LMResponse(BaseModel):
    model: str | None = None
    outputs: list[LMOutput] = Field(min_length=1)
    usage: LMUsage | dict[str, Any] | None = None
    cost: float | None = None
    response_id: str | None = None
    provider_response: Any | None = Field(default=None, exclude=True)
    provider_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_usage(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("usage"), dict):
            data = dict(data)
            data["usage"] = LMUsage(**data["usage"])
        return data

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        model: str | None = None,
        usage: LMUsage | dict[str, Any] | None = None,
        cost: float | None = None,
        **kwargs: Any,
    ) -> LMResponse:
        return cls(model=model, outputs=[LMOutput(parts=[LMTextPart(text=text)])], usage=usage, cost=cost, **kwargs)

    @property
    def output(self) -> LMOutput:
        return self.outputs[0]

    @property
    def parts(self) -> list[LMPart]:
        return self.output.parts

    @property
    def text(self) -> str | None:
        return self.output.text

    @property
    def reasoning_content(self) -> str | None:
        return self.output.reasoning_content

    @property
    def tool_calls(self) -> list[LMToolCallPart]:
        return self.output.tool_calls

    @property
    def citations(self) -> list[LMCitationPart]:
        return self.output.citations

    @property
    def images(self) -> list[LMImagePart]:
        return self.output.images

    @property
    def audio(self) -> list[LMAudioPart]:
        return self.output.audio

    @property
    def videos(self) -> list[LMVideoPart]:
        return self.output.videos

    @property
    def documents(self) -> list[LMDocumentPart]:
        return self.output.documents

    @property
    def binaries(self) -> list[LMBinaryPart]:
        return self.output.binaries

    def to_values(self) -> list[Any]:
        return [output.to_value() for output in self.outputs]

    def to_outputs(self) -> list[Any]:
        outputs: list[Any] = []
        for output in self.outputs:
            if requires_output_dict(output):
                outputs.append(output.to_output_dict())
            else:
                outputs.append(output.to_value())
        return outputs

    def usage_as_dict(self) -> dict[str, Any]:
        if self.usage is None:
            return {}
        if isinstance(self.usage, LMUsage):
            return self.usage.model_dump(exclude_none=True)
        return dict(self.usage)
