from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar, runtime_checkable

from dspy.history.repl_history import REPLHistory
from dspy.history.turn_log import TurnLog

if TYPE_CHECKING:
    from dspy.history.turn_event import TurnEvent
    from dspy.runtime.call_options import ModuleCallOptions
    from dspy.runtime.run_context import RunContext


@runtime_checkable
class AgentHistory(Protocol):
    @classmethod
    def empty(cls) -> Self: ...

    def append_turn(self, event: TurnEvent) -> Self: ...


@runtime_checkable
class TruncatableHistory(AgentHistory, Protocol):
    def truncate_oldest(self, n: int = 1) -> Self: ...


@runtime_checkable
class ConversationTurnLog(AgentHistory, Protocol):
    @property
    def turns(self) -> tuple[TurnEvent, ...]: ...


H = TypeVar("H", bound=TruncatableHistory)


class HistoryModule(Protocol[H]):
    async def __call__(
        self,
        *,
        turn_log: H,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **kwargs: Any,
    ) -> Any: ...


TurnLogModule = HistoryModule[TurnLog]
REPLHistoryModule = HistoryModule[REPLHistory]
