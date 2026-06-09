from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dspy.core.types.parts.models import LMPart, LMTextPart, _coerce_part


class LMMessageRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LMMessage(BaseModel):
    role: LMMessageRole
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
            role = data.get("role")
            if role is not None and not isinstance(role, LMMessageRole):
                try:
                    data["role"] = LMMessageRole(role)
                except ValueError as err:
                    raise ValueError(f"Invalid LMMessage role: {role!r}.") from err
            if "parts" in data:
                data["parts"] = [_coerce_part(part) for part in data["parts"]]
        return data

    @property
    def text(self) -> str | None:
        texts = [part.text for part in self.parts if isinstance(part, LMTextPart)]
        return "".join(texts) if texts else None
