"""dr-llm client protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext

__all__ = ["PoolSessionIdResolver"]


class PoolSessionIdResolver(Protocol):
    """Resolve a dr-llm pool acquire session id for a RunContext."""

    def __call__(self, run: RunContext, *, fallback: str | None = None) -> str: ...
