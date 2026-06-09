from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from dspy.primitives import Module
    from dspy.runtime.run_context import RunContext
    from dspy.runtime.usage_tracker import UsageTracker


@dataclass(frozen=True, slots=True)
class _AmbientFrame:
    run: RunContext
    caller: Module | None = None
    usage_tracker: UsageTracker | None = None


_AMBIENT_STACK: ContextVar[tuple[_AmbientFrame, ...]] = ContextVar("ambient_stack", default=())
ACTIVE_RUN: ContextVar[RunContext | None] = ContextVar("active_run", default=None)


def get_ambient_stack() -> tuple[_AmbientFrame, ...]:
    return _AMBIENT_STACK.get()


def get_active_run() -> RunContext | None:
    active = ACTIVE_RUN.get()
    if active is not None:
        return active
    stack = _AMBIENT_STACK.get()
    if not stack:
        return None
    return stack[-1].run


def get_caller_modules() -> tuple[Module, ...]:
    return tuple(frame.caller for frame in _AMBIENT_STACK.get() if frame.caller is not None)


@asynccontextmanager
async def call_scope(*, run: RunContext, caller: Module | None = None) -> AsyncIterator[None]:
    stack = _AMBIENT_STACK.get()
    frame = _AmbientFrame(run=run, caller=caller, usage_tracker=run.usage_tracker)
    stack_token = _AMBIENT_STACK.set(stack + (frame,))
    run_token = ACTIVE_RUN.set(run)
    try:
        yield
    finally:
        ACTIVE_RUN.reset(run_token)
        _AMBIENT_STACK.reset(stack_token)
