from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from dspy.core.types.call_options import ModuleCallOptions
    from dspy.history.repl_history import REPLHistory
    from dspy.history.turn_event import TurnEvent
    from dspy.history.turn_log import TurnLog
    from dspy.runtime.run_context import RunContext


@runtime_checkable
class AgentHistory(Protocol):
    @classmethod
    def empty(cls) -> Self: ...

    def append_turn(self, event: TurnEvent) -> Self: ...


@runtime_checkable
class ConversationTurnLog(AgentHistory, Protocol):
    @property
    def turns(self) -> tuple[TurnEvent, ...]: ...


class TurnLogModule(Protocol):
    async def __call__(
        self,
        *,
        turn_log: TurnLog,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **kwargs: Any,
    ) -> Any: ...


class REPLHistoryModule(Protocol):
    async def __call__(
        self,
        *,
        turn_log: REPLHistory,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **kwargs: Any,
    ) -> Any: ...
