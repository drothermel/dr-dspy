from __future__ import annotations

from typing import Protocol, TypeVar

__all__ = ["QueryRetriever"]

QueryT = TypeVar("QueryT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


class QueryRetriever(Protocol[QueryT, ResultT]):
    k: int

    async def __call__(self, query: QueryT, /) -> ResultT: ...

    async def aforward(self, query: QueryT, /) -> ResultT: ...
