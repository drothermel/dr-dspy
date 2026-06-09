from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from typing_extensions import Self


@runtime_checkable
class AgentHistory(Protocol):
    @classmethod
    def empty(cls) -> Self: ...

    def append_turn(self, event: dict[str, Any]) -> Self: ...


@runtime_checkable
class ConversationTurnLog(AgentHistory, Protocol):
    @property
    def turns(self) -> tuple[dict[str, Any], ...]: ...
