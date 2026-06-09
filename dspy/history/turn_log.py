from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from dspy.history.turn_event import TurnEvent


class TurnLog(BaseModel):
    turns: tuple[TurnEvent, ...] = ()
    model_config = ConfigDict(frozen=True)

    @field_validator("turns", mode="before")
    @classmethod
    def _coerce_turns(cls, value: Any) -> tuple[TurnEvent, ...]:
        if isinstance(value, TurnEvent):
            return (value,)
        if isinstance(value, dict):
            return (TurnEvent.model_validate(value),)
        if isinstance(value, (list, tuple)):
            coerced: list[TurnEvent] = []
            for index, item in enumerate(value):
                if isinstance(item, TurnEvent):
                    coerced.append(item)
                elif isinstance(item, dict):
                    coerced.append(TurnEvent.model_validate(item))
                else:
                    raise TypeError(f"TurnLog turns[{index}] must be a TurnEvent or dict, got {type(item).__name__}.")
            return tuple(coerced)
        raise TypeError(f"TurnLog turns must be a TurnEvent, dict, list, or tuple, got {type(value).__name__}.")

    @field_serializer("turns")
    def _serialize_turns(self, turns: tuple[TurnEvent, ...]) -> tuple[dict[str, Any], ...]:
        return tuple(turn.model_dump(mode="json", exclude_none=True) for turn in turns)

    @classmethod
    def empty(cls) -> TurnLog:
        return cls()

    def append_turn(self, event: TurnEvent) -> TurnLog:
        if not event.model_dump(mode="json", exclude_none=True):
            raise ValueError("Cannot append an empty TurnEvent; at least one field must be set.")
        return TurnLog(turns=(*self.turns, event))

    def truncate_oldest(self, n: int = 1) -> TurnLog:
        if len(self.turns) < n + 1:
            raise ValueError(
                "The turn log is too long so your prompt exceeded the context window, but the turn log cannot be truncated because it only has one turn."
            )
        return TurnLog(turns=self.turns[n:])
