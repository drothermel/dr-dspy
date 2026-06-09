from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TurnEvent(BaseModel):
    """Single agent turn log entry.

    Known agent fields are typed; task-specific input/output keys may appear as
    extra fields (e.g. ReActV2 pending inputs merged into the event).
    """

    model_config = ConfigDict(extra="allow")

    thought: str | None = None
    next_thought: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_calls: Any | None = None
    observation: Any | None = None
    generated_code: str | None = None
    code_output: str | None = None
    action: Any | None = None
    result: str | None = None
    reasoning: str | None = None
    code: str | None = None
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in type(self).model_fields if getattr(self, name) is not None}
        extra = getattr(self, "__pydantic_extra__", None) or {}
        for key, value in extra.items():
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnEvent:
        return cls.model_validate(data)
