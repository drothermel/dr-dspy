from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from dspy.history.turn_event import TurnEvent


@runtime_checkable
class AgentHistory(Protocol):
    @classmethod
    def empty(cls) -> Self: ...

    def append_turn(self, event: TurnEvent) -> Self: ...


@runtime_checkable
class ConversationTurnLog(AgentHistory, Protocol):
    @property
    def turns(self) -> tuple[TurnEvent, ...]: ...
