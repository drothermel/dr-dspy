from __future__ import annotations

from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from dspy.history.turn_event import TurnEvent


class TurnLog(BaseModel):
    turns: tuple[TurnEvent | dict[str, Any], ...] = ()
    model_config = ConfigDict(frozen=True)

    @field_validator("turns", mode="after")
    @classmethod
    def _ensure_turn_events(cls, value: tuple[TurnEvent | dict[str, Any], ...]) -> tuple[TurnEvent, ...]:
        return tuple(turn if isinstance(turn, TurnEvent) else TurnEvent.model_validate(turn) for turn in value)

    @field_serializer("turns")
    def _serialize_turns(self, turns: tuple[TurnEvent, ...]) -> tuple[dict[str, Any], ...]:
        return tuple(turn.to_dict() for turn in turns)

    @field_validator("turns", mode="before")
    @classmethod
    def _coerce_turns(cls, value: Any) -> Any:
        if isinstance(value, TurnEvent):
            return (value,)
        if isinstance(value, dict):
            return (TurnEvent.model_validate(value),)
        if isinstance(value, list):
            return tuple(TurnEvent.model_validate(item) if isinstance(item, dict) else item for item in value)
        return value

    @classmethod
    def empty(cls) -> TurnLog:
        return cls()

    def append_turn(self, event: TurnEvent) -> TurnLog:
        if not event.to_dict():
            return self
        return TurnLog(turns=(*self.turns, event))


def is_turn_log_type(annotation: Any) -> bool:
    if annotation is TurnLog:
        return True
    origin = get_origin(annotation)
    if origin is not None:
        return any(is_turn_log_type(arg) for arg in get_args(annotation))
    return False
