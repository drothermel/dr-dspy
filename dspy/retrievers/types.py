from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

__all__ = ["QueryRetriever", "RetrievedPassage"]

QueryT = TypeVar("QueryT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


@dataclass
class RetrievedPassage:
    long_text: str
    score: float | None = None
    pid: int | None = None


class QueryRetriever(Protocol[QueryT, ResultT]):
    k: int

    async def __call__(self, query: QueryT, /) -> ResultT: ...

    async def aforward(self, query: QueryT, /) -> ResultT: ...
