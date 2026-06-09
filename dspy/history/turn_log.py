from __future__ import annotations

from typing import Any

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
            raise ValueError("Cannot append an empty TurnEvent; at least one field must be set.")
        return TurnLog(turns=(*self.turns, event))

    def truncate_oldest(self, n: int = 1) -> TurnLog:
        if len(self.turns) < n + 1:
            raise ValueError(
                "The turn log is too long so your prompt exceeded the context window, but the turn log cannot be truncated because it only has one turn."
            )
        return TurnLog(turns=self.turns[n:])
