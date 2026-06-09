from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dspy.core.types.parts.models import LMPart, LMTextPart, LMToolResultPart, _coerce_part
from dspy.core.types.parts.openai import _parts_from_openai_content, _tool_calls_from_openai


class LMMessageRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LMMessage(BaseModel):
    role: LMMessageRole | str
    parts: list[LMPart]
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_parts(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            data = dict(data)
            if data.get("role") == "tool" and "parts" not in data:
                content = data.pop("content", None)
                call_id = data.pop("tool_call_id", None)
                name = data.pop("name", None)
                data["parts"] = [
                    LMToolResultPart(call_id=call_id, name=name, content=_parts_from_openai_content(content))
                ]
            elif "parts" not in data:
                parts = _parts_from_openai_content(data.pop("content", None)) if "content" in data else []
                if "tool_calls" in data:
                    parts.extend(_tool_calls_from_openai(data.pop("tool_calls") or []))
                data["parts"] = parts
            else:
                data["parts"] = [_coerce_part(part) for part in data["parts"]]
                if "tool_calls" in data:
                    data["parts"].extend(_tool_calls_from_openai(data.pop("tool_calls") or []))
        return data

    @property
    def text(self) -> str | None:
        texts = [part.text for part in self.parts if isinstance(part, LMTextPart)]
        return "".join(texts) if texts else None
