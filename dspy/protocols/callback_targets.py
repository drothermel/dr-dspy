from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LMCallbackTarget(Protocol):
    model: str
    history: list[Any]

    async def __call__(self, request: Any) -> Any: ...


@runtime_checkable
class AdapterCallbackTarget(Protocol):
    callbacks: list[Any]

    def format(self, *args: Any, **kwargs: Any) -> Any: ...

    def parse(self, *args: Any, **kwargs: Any) -> Any: ...

    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class EvaluateCallbackTarget(Protocol):
    callbacks: list[Any]

    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ToolCallbackTarget(Protocol):
    name: str | None

    def __call__(self, **kwargs: Any) -> Any: ...

    async def acall(self, **kwargs: Any) -> Any: ...
