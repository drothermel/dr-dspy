from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types.lm_response import LMResponse
from dspy.core.types.parts import LMAudioPart, LMCitationPart, LMImagePart
from dspy.core.types.usage import LMUsage


class LMDelta(BaseModel):
    type: str


class LMTextDelta(LMDelta):
    type: Literal["text_delta"] = "text_delta"
    text: str


class LMThinkingDelta(LMDelta):
    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class LMToolCallDelta(LMDelta):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str | None = None
    name: str | None = None
    args_delta: str | None = None


class LMCitationDelta(LMDelta):
    type: Literal["citation_delta"] = "citation_delta"
    citation: LMCitationPart


class LMImageDelta(LMDelta):
    type: Literal["image_delta"] = "image_delta"
    image: LMImagePart


class LMAudioDelta(LMDelta):
    type: Literal["audio_delta"] = "audio_delta"
    audio: LMAudioPart


LMAnyDelta = Annotated[
    LMTextDelta | LMThinkingDelta | LMToolCallDelta | LMCitationDelta | LMImageDelta | LMAudioDelta,
    Field(discriminator="type"),
]


class LMStreamEvent(BaseModel):
    type: str
    model_config = ConfigDict(extra="forbid")

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


class LMStreamStartEvent(LMStreamEvent):
    type: Literal["start"] = "start"
    model: str | None = None


class LMStreamDeltaEvent(LMStreamEvent):
    type: Literal["delta"] = "delta"
    output_index: int = Field(default=0, ge=0)
    part_index: int = Field(ge=0)
    delta: LMAnyDelta


class LMStreamOutputEndEvent(LMStreamEvent):
    type: Literal["output_end"] = "output_end"
    output_index: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    truncated: bool = False


class LMStreamEndEvent(LMStreamEvent):
    type: Literal["end"] = "end"
    usage: LMUsage | dict[str, Any] | None = None
    cost: float | None = None
    response: LMResponse | None = None


class LMStreamErrorEvent(LMStreamEvent):
    type: Literal["error"] = "error"
    error: Exception
    model_config = ConfigDict(arbitrary_types_allowed=True)
