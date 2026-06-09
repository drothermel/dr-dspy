from __future__ import annotations

from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, field_validator


class TurnLog(BaseModel):
    turns: tuple[dict[str, Any], ...] = ()
    model_config = ConfigDict(frozen=True)

    @field_validator("turns", mode="before")
    @classmethod
    def _coerce_turns(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return (value,)
        if isinstance(value, list):
            return tuple(value)
        return value

    @classmethod
    def empty(cls) -> TurnLog:
        return cls()

    def append_turn(self, event: dict[str, Any]) -> TurnLog:
        if not event:
            return self
        return TurnLog(turns=(*self.turns, dict(event)))


def is_turn_log_type(annotation: Any) -> bool:
    if annotation is TurnLog:
        return True
    origin = get_origin(annotation)
    if origin is not None:
        return any(is_turn_log_type(arg) for arg in get_args(annotation))
    return False
