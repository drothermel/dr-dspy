from __future__ import annotations

import json
from pprint import pformat
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import override

from dspy.core.types.history import (
    _history_request_kwargs,
    _history_request_messages_as_openai,
    _history_request_prompt,
)
from dspy.core.types.messages import LMMessage
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
    _coerce_part,
    _part_to_value,
    _tool_call_to_provider_dict,
)
from dspy.core.types.request import LMRequest


class LMUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def fill_aliases(self) -> LMUsage:
        if self.input_tokens is None and self.prompt_tokens is not None:
            self.input_tokens = self.prompt_tokens
        if self.output_tokens is None and self.completion_tokens is not None:
            self.output_tokens = self.completion_tokens
        if self.prompt_tokens is None and self.input_tokens is not None:
            self.prompt_tokens = self.input_tokens
        if self.completion_tokens is None and self.output_tokens is not None:
            self.completion_tokens = self.output_tokens
        if self.total_tokens is None and self.input_tokens is not None and (self.output_tokens is not None):
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


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
            data["tool_calls"] = [_tool_call_to_provider_dict(call) for call in self.tool_calls]
        if self.citations:
            data["citations"] = [citation.model_dump(exclude_none=True) for citation in self.citations]
        if self.logprobs is not None:
            data["logprobs"] = self.logprobs
        return data


def _requires_output_dict(output: LMOutput) -> bool:
    return bool(
        output.logprobs is not None or output.reasoning_content is not None or output.tool_calls or output.citations
    )


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

    @override
    def __iter__(self):
        return iter(self.to_values())

    def __getitem__(self, index: int) -> Any:
        return self.to_values()[index]

    def __len__(self) -> int:
        return len(self.outputs)

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
            if _requires_output_dict(output):
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


class LMHistoryEntry(BaseModel):
    request: LMRequest
    response: LMResponse
    timestamp: str
    uuid: str
    model_type: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @property
    def outputs(self) -> list[Any]:
        return self.response.to_outputs()

    @property
    def usage(self) -> dict[str, Any]:
        return self.response.usage_as_dict()

    @property
    def cost(self) -> float | None:
        return self.response.cost

    @property
    def model(self) -> str:
        return self.request.model

    @property
    def prompt(self) -> str | None:
        return _history_request_prompt(self.request)

    @property
    def messages(self) -> list[LMMessage]:
        return self.request.messages

    @property
    def messages_as_openai(self) -> list[dict[str, Any]]:
        return _history_request_messages_as_openai(self.request)

    @property
    def kwargs(self) -> dict[str, Any]:
        return _history_request_kwargs(self.request)

    @property
    def response_model(self) -> str | None:
        return self.response.model

    @override
    def __repr__(self) -> str:
        formatted = pformat(self.model_dump(mode="python", exclude_none=True), width=100, sort_dicts=False)
        return f"LMHistoryEntry(\n{formatted}\n)"

    @override
    def __str__(self) -> str:
        return repr(self)

    def to_dict(self, *, mode: str = "python", exclude_none: bool = False, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            return self.model_dump(mode=mode, exclude_none=exclude_none, **kwargs)
        data = {
            **self.model_dump(mode="python", exclude_none=True),
            "outputs": self.outputs,
            "usage": self.usage,
            "cost": self.cost,
            "model": self.model,
            "prompt": self.prompt,
            "messages": self.messages_as_openai,
            "kwargs": self.kwargs,
            "response_model": self.response_model,
        }
        if mode != "python":
            data = json.loads(json.dumps(data, default=_json_default))
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return str(value)
